# Connecting to the PostgreSQL database

This project uses **PostgreSQL + pgvector** in production and **SQLite** for quick local
dev. This guide covers how to connect to Postgres from your **Django app**, from the
terminal, and from **VS Code** while you develop.

> The app automatically switches to PostgreSQL whenever `DATABASE_URL` is set (see
> `config/settings.py`). If it is not set, Django falls back to the local `db.sqlite3`.

---

## 1. Django app connection (what the backend needs)

Set a `DATABASE_URL` in your environment (Azure App Service "Application settings" or a
`.env` file, or the shell for local dev).

```
# Postgres (managed Azure Flexible Server or self-hosted VM both work)
DATABASE_URL=postgres://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
```

Notes:

- **TLS (SSL):** Azure Postgres requires SSL by default. Keep `?sslmode=require` (or
  `sslmode=verify-full` with a CA cert in production).
- **No `DATABASE_URL` → SQLite.** Leave it unset for pure local work.
- After switching engines, run migrations:
  ```bash
  python manage.py migrate
  ```

Then verify Django sees the right engine:

```bash
python manage.py shell -c "from django.db import connection; print(connection.vendor, connection.settings_dict['NAME'])"
# e.g. "postgresql fundzaai"
```

---

## 2. psql from the VS Code terminal

You can connect straight from the integrated terminal with the same URL:

```bash
psql "postgres://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
```

Useful quick commands while developing:

```sql
\dt                              -- list tables belonging to the current user
\d public.masteryrecord          -- describe one table
SELECT COUNT(*) FROM public.documentchunk;
SELECT id, title, status FROM public.syllabus_document LIMIT 10;
\q                               -- quit
```

---

## 3. VS Code extensions (browse + run queries from the editor)

Two easy options. Pick one:

### Option A — SQLTools (recommended)
1. Install **"SQLTools"** (by Matheus Teixeira) and the **"SQLTools PostgreSQL/Cockroach DB"** driver.
2. In the **Database** panel → *Add New Connection* → *PostgreSQL*.
3. Fill in:

   | Field | Value |
   |-------|-------|
   | Connection name | `Azure FundzaAI` |
   | Server/Host | your Postgres hostname |
   | Port | `5432` |
   | Database | your DB name (e.g. `fundzaai`) |
   | User | your user |
   | Password | your password (or "Ask on connect") |
   | SSL/TLS | **Enable** (Azure requires it) |

4. You can also **paste the connection string**: SQLTools accepts
   `postgresql://USER:PASS@HOST:5432/DBNAME?sslmode=require`.

### Option B — "PostgreSQL" extension (by Chris Kolkman)
1. Install **"PostgreSQL"**.
2. `Ctrl+Shift+P` → *PostgreSQL: Add Connection* → fill host/port/db/user/password, enable SSL.
3. Browse databases/tables and run queries from a query window.

> ⚠️ Do **not** use the "SQL Server (mssql)" extension — that is for SQL Server, not PostgreSQL.

---

## 4. Firewall / allow-list (required to connect from your dev machine)

Azure Database for PostgreSQL blocks external connections by default.

- **Azure Flexible Server:** Settings → **Networking** → add your **current public IP**
  (or enable "Allow public access from any Azure service" only if appropriate) and save.
- Your home/office IP can change; re-add it when you get "not accepting connections" /
  timeout errors.

After adding the IP, test from VS Code or `psql` once — then it will stay connected for
development.

---

## 5. pgvector

pgvector is a **built-in PostgreSQL extension** — you do **not** need a separate service.

Enable it once on your Postgres (the `vector` extension is supported by Azure Flexible
Server):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verify:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

> **Current status:** `DocumentChunk.embedding` is still a `JSONField` and retrieval
> computes cosine similarity in Python (`apps/rag/services/retriever.py`). A `VectorField`
> + HNSW index migration is the planned upgrade so queries use the vector index directly.
> Until that lands, pgvector won't do any work regardless of which Postgres you connect to.

---

## 6. Re-ingesting content after a fresh pull / new DB

`media/` (syllabus PDFs, past papers, figures) is **not** in git, so a fresh clone or a
new Postgres has an **empty RAG corpus**. After migrating:

1. Upload the source documents (through the app's document upload) or copy them into
   `media/`.
2. Re-embed the chunks with your chosen provider:
   ```bash
   python manage.py reembed_chunks
   ```
3. Confirm retrieval returns chunks:
   ```bash
   python manage.py shell -c "from apps.rag.services.retriever import retrieve; from apps.syllabus.models import Syllabus; print(len(retrieve(Syllabus.objects.first(), 'quadratic equations')))"
   ```

---

## 7. Example env for production (Azure App Service)

```
DATABASE_URL=postgres://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
DJANGO_SECRET_KEY=<new-long-random-string>
DJANGO_DEBUG=0
DJANGO_ALLOWED_HOSTS=<your-app-domain>
OPENROUTER_API_KEY=<your-key>
EMBEDDING_PROVIDER=local            # 'hash' | 'local' | 'openai'
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
REDIS_URL=redis://...
```

Run the server with `daphne` (Channels-aware), not `runserver`:

```bash
daphne -b 0.0.0.0 -p 8000 config.asgi:application
```