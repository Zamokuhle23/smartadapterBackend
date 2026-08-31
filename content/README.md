# Syllabus Content Drop Folder 📚

Put your ECESWA syllabus PDFs, study notes and past papers here, organised per subject,
then ingest them in one shot:

```powershell
cd C:\work\FundzaAI\backend
python manage.py ingest_folder content --subject 6880     # EGCSE Mathematics
python manage.py ingest_folder content --subject 6873     # EGCSE English Language
```

Supported formats: `.pdf`, `.txt`, `.md`, `.docx` (subfolders are scanned recursively).

## Source priority: Cambridge IGCSE (primary) + ECESWA EGCSE (secondary)

The practice/exam generator weights provenance **70% Cambridge IGCSE / 30% ECESWA
EGCSE** (see `apps/syllabus/services/subject_map.py`), because IGCSE is ~18x larger
and EGCSE is its local derivative. Every question is tagged `source` (`igcse`/`egcse`)
so the app labels it clearly - Cambridge-sourced items show "Cambridge IGCSE".

### Batch-ingest the whole downloaded corpus (no copying needed)

```powershell
# Cambridge IGCSE = PRIMARY (tagged source=igcse)
python manage.py import_pastpapers --source igcse

# ECESWA EGCSE = SECONDARY (tagged source=egcse)
python manage.py import_pastpapers --source egcse

# Smoke-test a handful first
python manage.py import_pastpapers --source igcse --limit 20
```

Defaults read from `C:\work\FundzaAI\Resource\{IGCSE|EGCSE}_Papers`. Cambridge
codes are auto-mapped to the matching EGCSE subject (e.g. `0580 Mathematics`,
`0620/0625 -> Physical Science`); unmapped codes (ICT `3015`, French `0520`) are
skipped with a note. Only **past paper** and **mark scheme** PDFs are indexed.

> **Offline note:** ingestion embeds chunks for retrieval. If the `.env` sets
> `EMBEDDING_PROVIDER=local` but the model can't be downloaded (no network), run
> the import with the fast offline hasher to populate the DB, then re-embed later:
> ```powershell
> $env:EMBEDDING_PROVIDER='hash'; python manage.py import_pastpapers --source igcse
> ```

## Naming matters: past papers & mark schemes 📄

The generator reads provenance straight from filenames, so name files like this:

| Filename | Detected as |
|---|---|
| `6880_paper1_2022.pdf` | **Past paper**, Paper 1, year 2022 (MCQ-style source) |
| `Maths P2 2023.pdf` | **Past paper**, Paper 2, year 2023 (structured source) |
| `paper1_2022 mark scheme.pdf` | Mark scheme, Paper 1, year 2022 |
| `EGCSE-mathematics-syllabus.pdf` | Official syllabus (assessment weightings) |
| `0580_m24_qp_12.pdf` | **Cambridge IGCSE** paper, Paper 1, 2024 (qp = question paper) |
| `0620_w17_ms_21.pdf` | **Cambridge IGCSE** mark scheme, Paper 2, 2017 (ms = mark scheme) |
| `BIOLOGY 2.pdf` | ECESWA past paper, Paper 2 (bare "SUBJECT N" form) |

Cambridge conventions: `<code>_<m|s|w><yy>_<qp|ms>_<variant>.pdf` where the variant's
first digit is the paper number (`12` → Paper 1, `22` → Paper 2). EGCSE files: include
`mark scheme`/`MS` for marking guides, or a trailing paper digit.

Every ingested document is chunked, embedded and indexed against that subject, so the
tutor's answers AND generated quiz questions stay inside syllabus scope.

Priority subjects (from PLAN.md):
| Code | Subject | Level |
|---|---|---|
| 6880 | Mathematics | EGCSE |
| 6873 | English Language | EGCSE |
| 6884 | Biology | EGCSE |
| 6888 | Physical Science | EGCSE |
| 309 | Mathematics | JC |
| 101 | English Language | JC |
