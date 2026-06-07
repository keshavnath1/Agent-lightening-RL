import { PrismaClient } from "@prisma/client";
import { withAccelerate } from "@prisma/extension-accelerate";
const prisma = new PrismaClient({ datasources: { db: { url: process.env.DATABASE_URL } } }).$extends(withAccelerate());
try {
  const info = {};
  info.current = await prisma.$queryRaw`select current_database() as database, current_user as user, current_schema() as schema`;
  info.schema_privileges = await prisma.$queryRaw`select has_schema_privilege(current_user, public, USAGE) as usage, has_schema_privilege(current_user, public, CREATE) as create`;
  info.visible_tables = await prisma.$queryRaw`select table_schema, table_name from information_schema.tables where table_schema not in (pg_catalog,information_schema) order by table_schema, table_name limit 50`;
  console.log(JSON.stringify({ok:true, ...info}, null, 2));
} catch (err) {
  console.log(JSON.stringify({ok:false, error_name: err?.name ?? null, error_message: String(err?.message ?? err).slice(0, 800)}, null, 2));
  process.exit(1);
} finally {
  await prisma.$disconnect().catch(() => {});
}
