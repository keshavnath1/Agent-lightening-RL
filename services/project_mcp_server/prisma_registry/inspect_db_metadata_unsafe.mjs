import { PrismaClient } from "@prisma/client";
import { withAccelerate } from "@prisma/extension-accelerate";
const prisma = new PrismaClient({ datasources: { db: { url: process.env.DATABASE_URL } } }).$extends(withAccelerate());
try {
  const current = await prisma.$queryRawUnsafe("select current_database() as database, current_user as user, current_schema() as schema");
  const visible_tables = await prisma.$queryRawUnsafe("select table_schema, table_name from information_schema.tables where table_schema not in (pg_catalog,information_schema) order by table_schema, table_name limit 100");
  const registry_tables = await prisma.$queryRawUnsafe("select table_schema, table_name from information_schema.tables where table_name in (real_benchmark_tasks,real_dataset_summaries,benchmark_run_status,tool_call_log) order by table_schema, table_name");
  const public_create = await prisma.$queryRawUnsafe("select has_schema_privilege(current_user, public, USAGE) as usage, has_schema_privilege(current_user, public, CREATE) as create_privilege");
  console.log(JSON.stringify({ok:true, current, public_create, visible_table_count: visible_tables.length, visible_tables, registry_tables}, null, 2));
} catch (err) {
  console.log(JSON.stringify({ok:false, error_name: err?.name ?? null, error_message: String(err?.message ?? err).slice(0, 1000)}, null, 2));
  process.exit(1);
} finally {
  await prisma.$disconnect().catch(() => {});
}
