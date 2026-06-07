import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { PrismaClient } from "@prisma/client";
import { withAccelerate } from "@prisma/extension-accelerate";

function databaseUrl() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL must be provided through the environment.");
  }
  return url;
}

function client() {
  const url = databaseUrl();
  const prisma = new PrismaClient({ datasources: { db: { url } } });
  if (url.startsWith("prisma+postgres://") || url.startsWith("prisma://")) {
    return prisma.$extends(withAccelerate());
  }
  return prisma;
}

function readArg(flag, fallback = null) {
  const index = process.argv.indexOf(flag);
  if (index === -1) return fallback;
  return process.argv[index + 1] || fallback;
}

const MIGRATIONS = [
  `create table if not exists real_benchmark_tasks (
    task_id text primary key,
    source text not null default 'OpenML',
    source_task_id text,
    data_source text not null default 'OpenML',
    openml_data_id integer,
    openml_name text,
    openml_version integer,
    openml_url text,
    dataset_name text not null,
    domain text,
    problem_statement text not null,
    prediction_goal text not null,
    problem_type text not null,
    objective text not null,
    payload jsonb not null,
    dataset_ref text not null,
    artifact_ref text not null,
    target_column text not null,
    primary_metric text not null,
    secondary_metrics jsonb not null default '[]'::jsonb,
    metric text not null,
    row_count integer not null,
    column_count integer not null,
    split text,
    synthetic boolean not null default false,
    status text not null default 'ready',
    priority integer not null default 100,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint real_benchmark_tasks_metadata_only check (synthetic = false),
    constraint real_benchmark_tasks_problem_type check (problem_type in ('binary_classification','multiclass_classification','regression','unknown'))
  )`,
  `alter table real_benchmark_tasks add column if not exists source text not null default 'OpenML'`,
  `alter table real_benchmark_tasks add column if not exists source_task_id text`,
  `alter table real_benchmark_tasks add column if not exists dataset_name text`,
  `alter table real_benchmark_tasks add column if not exists problem_statement text`,
  `alter table real_benchmark_tasks add column if not exists prediction_goal text`,
  `alter table real_benchmark_tasks add column if not exists problem_type text`,
  `alter table real_benchmark_tasks add column if not exists artifact_ref text`,
  `alter table real_benchmark_tasks add column if not exists primary_metric text`,
  `alter table real_benchmark_tasks add column if not exists secondary_metrics jsonb not null default '[]'::jsonb`,
  `alter table real_benchmark_tasks add column if not exists openml_data_id integer`,
  `alter table real_benchmark_tasks add column if not exists openml_name text`,
  `alter table real_benchmark_tasks add column if not exists openml_version integer`,
  `alter table real_benchmark_tasks add column if not exists openml_url text`,
  `alter table real_benchmark_tasks add column if not exists domain text`,
  `alter table real_benchmark_tasks add column if not exists objective text`,
  `alter table real_benchmark_tasks add column if not exists row_count integer`,
  `alter table real_benchmark_tasks add column if not exists column_count integer`,
  `alter table real_benchmark_tasks add column if not exists synthetic boolean not null default false`,
  `create table if not exists real_dataset_summaries (
    task_id text primary key references real_benchmark_tasks(task_id) on delete cascade,
    schema_json jsonb not null,
    summary_json jsonb not null,
    artifact_manifest jsonb not null,
    raw_rows_exposed boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint real_dataset_summaries_no_raw_rows check (raw_rows_exposed = false)
  )`,
  `alter table real_dataset_summaries add column if not exists raw_rows_exposed boolean not null default false`,
  `create table if not exists benchmark_run_status (
    run_id text primary key,
    task_id text references real_benchmark_tasks(task_id) on delete cascade,
    track text not null,
    policy_version text,
    status text not null,
    reward numeric,
    metrics jsonb,
    artifacts jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
  )`,
  `create table if not exists tool_call_log (
    id bigserial primary key,
    timestamp timestamptz not null default now(),
    task_id text,
    tool_name text not null,
    arguments jsonb,
    status text not null,
    output_ref text,
    metadata jsonb
  )`,
  `create index if not exists idx_real_benchmark_tasks_status_priority on real_benchmark_tasks(status, priority, task_id)`,
  `create index if not exists idx_tool_call_log_task_time on tool_call_log(task_id, timestamp desc)`
];

async function migrate(prisma) {
  for (const statement of MIGRATIONS) {
    await prisma.$executeRawUnsafe(statement);
  }
  return { ok: true, migration_count: MIGRATIONS.length };
}

async function seedOpenml(prisma) {
  const payloadPath = readArg("--payload", path.resolve("reports/openml_registry_payload.json"));
  const raw = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
  const payloads = raw.payloads || [];
  if (!Array.isArray(payloads) || payloads.length === 0) {
    throw new Error(`No payloads found in ${payloadPath}`);
  }
  const seeded = [];
  for (const entry of payloads) {
    const task = entry.task;
    const summary = entry.summary_json;
    if (entry.raw_rows_exposed !== false || summary.raw_rows_exposed !== false) {
      throw new Error(`Refusing to seed ${task.task_id}: payload indicates raw rows are exposed.`);
    }
    await prisma.$executeRawUnsafe(
      `insert into real_benchmark_tasks(
        task_id, source, source_task_id, data_source, openml_data_id, openml_name, openml_version, openml_url, dataset_name, domain,
        problem_statement, prediction_goal, problem_type, objective, payload, dataset_ref, artifact_ref, target_column,
        primary_metric, secondary_metrics, metric, row_count, column_count, split, synthetic, status, priority, updated_at
      ) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15::jsonb,$16,$17,$18,$19,$20::jsonb,$21,$22,$23,$24,false,'ready',100,now())
      on conflict (task_id) do update set
        source=excluded.source,
        source_task_id=excluded.source_task_id,
        data_source=excluded.data_source,
        openml_data_id=excluded.openml_data_id,
        openml_name=excluded.openml_name,
        openml_version=excluded.openml_version,
        openml_url=excluded.openml_url,
        dataset_name=excluded.dataset_name,
        domain=excluded.domain,
        problem_statement=excluded.problem_statement,
        prediction_goal=excluded.prediction_goal,
        problem_type=excluded.problem_type,
        objective=excluded.objective,
        payload=excluded.payload,
        dataset_ref=excluded.dataset_ref,
        artifact_ref=excluded.artifact_ref,
        target_column=excluded.target_column,
        primary_metric=excluded.primary_metric,
        secondary_metrics=excluded.secondary_metrics,
        metric=excluded.metric,
        row_count=excluded.row_count,
        column_count=excluded.column_count,
        split=excluded.split,
        synthetic=false,
        status='ready',
        updated_at=now()`,
      task.task_id,
      task.source || task.data_source || "OpenML",
      task.source_task_id || String(task.openml_data_id || task.task_id),
      task.data_source || task.source || "OpenML",
      Number(task.openml_data_id || 0),
      task.openml_name || task.dataset_name || null,
      task.openml_version == null ? null : Number(task.openml_version),
      task.openml_url || null,
      task.dataset_name || task.openml_name || task.task_id,
      task.domain || null,
      task.problem_statement,
      task.prediction_goal,
      task.problem_type,
      task.objective || task.problem_statement,
      JSON.stringify(task),
      task.dataset_ref,
      task.artifact_ref || task.dataset_ref,
      task.target_column,
      task.primary_metric || task.metric || "roc_auc",
      JSON.stringify(task.secondary_metrics || []),
      task.metric || task.primary_metric || "roc_auc",
      Number(summary.row_count),
      Number(summary.column_count),
      task.split || null
    );
    await prisma.$executeRawUnsafe(
      `insert into real_dataset_summaries(task_id, schema_json, summary_json, artifact_manifest, raw_rows_exposed, updated_at)
       values ($1,$2::jsonb,$3::jsonb,$4::jsonb,false,now())
       on conflict (task_id) do update set
        schema_json=excluded.schema_json,
        summary_json=excluded.summary_json,
        artifact_manifest=excluded.artifact_manifest,
        raw_rows_exposed=false,
        updated_at=now()`,
      task.task_id,
      JSON.stringify(entry.schema_json),
      JSON.stringify(entry.summary_json),
      JSON.stringify(entry.artifact_manifest)
    );
    seeded.push({ task_id: task.task_id, problem_type: task.problem_type, primary_metric: task.primary_metric, rows: Number(summary.row_count), columns: Number(summary.column_count), raw_rows_exposed: false });
  }
  return { ok: true, seeded_count: seeded.length, seeded, raw_rows_exposed: false };
}

async function verify(prisma) {
  const rows = await prisma.$queryRaw`select t.task_id, t.dataset_name, t.problem_statement, t.prediction_goal, t.problem_type, t.target_column, t.primary_metric, t.secondary_metrics, t.artifact_ref, t.row_count, t.column_count, s.raw_rows_exposed from real_benchmark_tasks t join real_dataset_summaries s using(task_id) order by t.priority, t.task_id`;
  return { ok: true, task_count: rows.length, tasks: rows };
}

async function ping(prisma) {
  const rows = await prisma.$queryRaw`select 1::int as ok`;
  return { ok: Array.isArray(rows) && Number(rows[0]?.ok) === 1 };
}

async function main() {
  const command = process.argv[2] || "help";
  const prisma = client();
  try {
    let result;
    if (command === "migrate") result = await migrate(prisma);
    else if (command === "seed-openml") result = await seedOpenml(prisma);
    else if (command === "verify") result = await verify(prisma);
    else if (command === "ping") result = await ping(prisma);
    else throw new Error(`Unknown command: ${command}`);
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await prisma.$disconnect().catch(() => {});
  }
}

main().catch((error) => {
  console.error(JSON.stringify({ ok: false, error_name: error?.name || null, error_message: String(error?.message || error).slice(0, 1000) }, null, 2));
  process.exit(1);
});
