# Enabling pgvector search

`DocumentChunk.embedding` is stored as a `JSONField` (list of floats) so the app
works on SQLite too. On PostgreSQL you can add a real pgvector column and an HNSW
index so retrieval runs in SQL instead of loading up to 5000 rows into Python.

## 1. Connect the app to Postgres

Set in your environment / `.env`:

```
DATABASE_URL=postgres://USER:PASS@HOST:5432/fundzaai?sslmode=require
```

Then migrate (creates all app tables on Postgres):

```bash
python manage.py migrate
```

Verify Django uses Postgres:

```bash
python manage.py shell -c "from django.db import connection; print(connection.vendor)"
# -> postgresql
```

## 2. Enable the pgvector extension

On your Azure Postgres (or any Postgres), run once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Confirm:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

## 3. Add the vector column + HNSW index (suggested migration)

Add a `pgvector.django.VectorField` to `DocumentChunk` and a migration that adds
the column + an HNSW index. Example model field:

```python
from pgvector.django import VectorField

class DocumentChunk(models.Model):
    ...
    embedding_vec = VectorField(dimensions=384, null=True)   # 384 if using bge-small
```

Migration operations (Postgres only):

```python
from django.db import migrations
from pgvector.django import HnswIndex

class Migration(migrations.Migration):
    operations = [
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS vector;",
            state_operations=[
                migrations.AddField(
                    model_name="documentchunk",
                    name="embedding_vec",
                    field=pgvector.django.VectorField(dimensions=384, null=True),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="documentchunk",
            index=HnswIndex(
                name="docchunk_embedding_vec_hnsw",
                fields=["embedding_vec"],
                opclasses=["vector_cosine_ops"],
            ),
        ),
    ]
```

> **Dimension must match `EMBEDDING_DIM`** in your `.env` (384 for
> `BAAI/bge-small-en-v1.5`). If you change model/dim, re-embed (below) and pick a
> new column.

Apply it:

```bash
python manage.py migrate
```

## 4. Backfill `embedding_vec` from the JSON `embedding`

`embedding_vec` starts NULL. Populate it from the existing JSON, then re-embed
everything with your current provider so dimensions are consistent:

```bash
# one-time copy of JSON floats -> vector column
python manage.py shell -c "
from django.db.models import Func, F
from apps.rag.models import DocumentChunk
from django.db import connection
if connection.vendor == 'postgresql':
    DocumentChunk.objects.filter(embedding_vec__isnull=True).update(
        embedding_vec=Func(F('embedding'), function='ARRAY')
    )
"

# re-embed all chunks with the current provider/model (recommended)
python manage.py reembed_chunks
```

## 5. Switch the retriever to SQL vector search

Replace the Python cosine loop in `apps/rag/services/retriever.py` with a
Postgres `.order_by(CosineDistance("embedding_vec", qvec))` query when
`connection.vendor == "postgresql"`, falling back to the existing path on
SQLite. Example:

```python
from pgvector.django import CosineDistance

# inside retrieve(), after computing qvec, only on postgres:
qs = qs.filter(embedding_vec__isnull=False)\
       .order_by(CosineDistance("embedding_vec", qvec))[:top_k]
```

## 6. Verify

```bash
python manage.py shell -c "
from django.db import connection
print('pgvector ready:', connection.vendor == 'postgresql')
from pgvector.django import CosineDistance
print('import ok')
"
```

Then confirm retrieval returns chunks (non-empty) after ingestion.

---

## When this runs *locally* (SQLite)
The `VectorField`/HNSW migration is Postgres-only. On SQLite the JSON `embedding`
path in the retriever stays active, so local dev/tests keep working. Only enable
steps 3–5 on the Postgres instance where pgvector is installed.