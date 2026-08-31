"""
LLM-driven subject skill-tree seeder (two-level hierarchy).

Reads each subject's most recent syllabus document and asks the LLM to extract a
structured TWO-LEVEL skill tree: strands (e.g. "Number", "Shape, Position & Space")
-> subtopics (e.g. "16. Vectors") -> numbered learning objectives, each tagged
core/extended when the subject is tiered. This correctly separates interleaved
Core and Extended syllabus columns (which share numbering) into properly tier-tagged
objectives so Core students only see core/both, Extended students see all.

Persists as Topic rows (kind=strand/subtopic, nested via parent) + LearningObjective
rows (with code like "16.3" and tier). Sets Subject.tiers_available.

Far more reliable than regex heading-scraping for prose-style syllabi.

Usage:
    python manage.py seed_skill_trees --subject 6880          # one subject (replaces its tree)
    python manage.py seed_skill_trees --all --limit 3         # smoke test
    python manage.py seed_skill_trees --all                   # all EGCSE subjects
"""

from django.core.management.base import BaseCommand, CommandError

from apps.syllabus.models import LearningObjective, Subject, Topic
from apps.rag.models import DocumentChunk
from apps.rag.services.llm import get_chat_provider
from apps.syllabus.services.ingestion import extract_text

SYSTEM = (
    "You convert an official Eswatini/Cambridge EGCSE syllabus into a clean skill "
    "tree. Output ONLY valid JSON, no markdown fences."
)

TASK = """Here is the {level} {subject_name} ({code}) syllabus:

---SYLLABUS START---
{syllabus}
---SYLLABUS END---

Extract the curriculum as a structured TWO-LEVEL skill tree.

Rules:
1. "tiered": true ONLY if the syllabus offers separate Core and Extended curricula
   (distinct papers / content / grade ranges, e.g. "core curriculum" & "extended
   curriculum", "papers 1 and 2 core / papers 3 and 4 extended").
2. "strands": the subject's main content areas (e.g. Maths: Number, Algebra,
   Shape, Position & Space, Data Handling, Probability). Use the syllabus's real
   groupings. Exclude boilerplate ("Introduction", "Assessment", "Aims").
3. Each strand has "subtopics": a COMPLETE list of its numbered content sections,
   preserving the syllabus's section number in "code" (e.g. "16" for "16. Vectors").
   A subtopic may map to several strands via "topic_areas" (e.g. Vectors ->
   ["Algebra", "Shape, Position & Space"]).
4. Each subtopic has "objectives": the assessable skills, preserving the syllabus's
   sub-number in "code" (e.g. "16.3"). Keep the syllabus wording - only tidy it.
5. Tier: the syllabus interleaves Core and Extended under one numbering. Tag each
   objective "tier": "core", "extended", or "" (appears in both). ONLY items marked
   "Extended curriculum only" are "extended"; everything else is "core" or "".
6. "difficulty" 1-5 (1 recall ... 5 hard application) per objective.
7. Include EVERY numbered content section - do not drop any major strand/subtopic.

Return exactly:
{{"tiered": true/false, "strands": [
  {{"title": "Shape, Position & Space", "subtopics": [
     {{"code": "16", "title": "Vectors", "topic_areas": ["Algebra", "Shape, Position & Space"],
       "objectives": [
         {{"code": "16.3", "statement": "Add and subtract vectors", "tier": "core", "difficulty": 2}}
       ]}}
  ]}}
]}}"""
class Command(BaseCommand):
    help = "Seed Topic + LearningObjective skill trees from syllabi via the LLM"

    def add_arguments(self, parser):
        parser.add_argument("--subject", default=None, help="Subject code, e.g. 6880")
        parser.add_argument("--subjects", default="", help="Comma-separated codes to process")
        parser.add_argument("--all", action="store_true", help="Seed every EGCSE subject")
        parser.add_argument("--limit", type=int, default=0, help="Cap subjects processed")
        parser.add_argument("--dry-run", action="store_true", help="Ask LLM but do not save")
        parser.add_argument(
            "--model", default=None,
            help="OpenRouter model to use for extraction (overrides the app default). "
                 "Use a stronger model (e.g. a deepseek/qwen) for faithful full-syllabus trees.",
        )

    def handle(self, *args, **options):
        self.model = options["model"]
        qs = Subject.objects.filter(syllabus__level="EGCSE").order_by("code")
        if options["subjects"]:
            codes = [c.strip() for c in options["subjects"].split(",") if c.strip()]
            qs = qs.filter(code__in=codes)
        elif options["subject"]:
            qs = qs.filter(code=options["subject"])
        elif not options["all"]:
            raise CommandError("Pass --subject CODE, --subjects 'a,b', or --all")
        qs = list(qs)
        if options["limit"]:
            qs = qs[: options["limit"]]

        ok = failed = 0
        for subject in qs:
            try:
                info = self._extract(subject)
                if not options["dry_run"]:
                    self._save(subject, info)
                self.stdout.write(self.style.SUCCESS(
                    f"OK {subject.code} {subject.name}: tiered={info.get('tiered')} "
                    f"strands={len(info.get('strands', []))}"
                ))
                ok += 1
            except Exception as exc:  # noqa: BLE001 - keep going
                failed += 1
                self.stdout.write(self.style.ERROR(f"FAIL {subject.code}: {exc}"))
        self.stdout.write(self.style.SUCCESS(f"Done. {ok} ok, {failed} failed."))

    # ------------------------------------------------------------------
    def _extract(self, subject) -> dict:
        from apps.quiz.services.generator import _extract_json_object

        doc = self._latest_syllabus(subject)
        if doc is None:
            raise CommandError(f"{subject.name}: no syllabus document")
        chunks = list(DocumentChunk.objects.filter(document=doc).order_by("ordinal")
                      .values_list("text", flat=True)[:600])
        if chunks:
            syllabus_text = "\n".join(chunks)
        else:
            try:
                syllabus_text = extract_text(doc.file.path)[:100000]
            except Exception:  # noqa: BLE001 - corrupted PDF, nothing we can do
                raise CommandError(f"{subject.name}: syllabus text unreadable")
        syllabus_text = syllabus_text[:100000]

        provider = get_chat_provider()
        # Chunk the syllabus so the model cannot silently drop strands from a
        # very long document. Combine each chunk's strands into one tree.
        merged_strands: list[dict] = []
        title_seen = {}
        tiered = False
        for part in self._split_text(syllabus_text, size=8000):
            try:
                info = self._llm_extract(provider, subject, part)
            except Exception:  # noqa: BLE001 - a bad chunk shouldn't abort the subject
                continue
            if info.get("tiered"):
                tiered = True
            for strand in info.get("strands", []):
                title = str(strand.get("title", "")).strip()
                if not title:
                    continue
                if title in title_seen:
                    # Merge subtopics into the existing strand.
                    idx = title_seen[title]
                    for sub in strand.get("subtopics", []):
                        if sub.get("title"):
                            merged_strands[idx].setdefault("subtopics", []).append(sub)
                else:
                    title_seen[title] = len(merged_strands)
                    merged_strands.append(strand)
        if not merged_strands:
            raise CommandError(f"{subject.name}: LLM returned no strands")
        return {"tiered": tiered, "strands": merged_strands}

    @staticmethod
    def _split_text(text: str, size: int = 8000):
        """Split text into ~size-char pieces, cutting on paragraph boundaries."""
        lines = text.split("\n")
        parts, cur = [], []
        cur_len = 0
        for line in lines:
            line = line.strip()
            line_len = len(line) + 1
            if cur and cur_len + line_len > size:
                parts.append("\n".join(cur))
                cur, cur_len = [], 0
            cur.append(line)
            cur_len += line_len
        if cur:
            parts.append("\n".join(cur))
        return [p for p in parts if p.strip()]

    def _llm_extract(self, provider, subject, part) -> dict:
        from apps.quiz.services.generator import _extract_json_object

        raw = provider.chat([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK.format(
                level=subject.syllabus.get_level_display(),
                subject_name=subject.name,
                code=subject.code,
                syllabus=part,
            )},
        ], model=self.model, max_tokens=8000)
        if "[offline mode" in raw:
            raise CommandError("LLM not configured")
        return _extract_json_object(raw)

    @staticmethod
    def _latest_syllabus(subject):
        """
        Pick the subject's latest EDITION syllabus (highest first year in the title),
        preferring a ready one with chunks over a corrupt/incomplete newer-by-upload.
        E.g. EGCSE Mathematics 2027-2029 Syllabus should beat EGCSE Mathematics 2024-2026.
        """
        import re

        docs = list(
            subject.documents.filter(doc_type="syllabus")
        )
        if not docs:
            return None

        def year_of(d):
            m = re.search(r"(20\d{2})\s*[-–]\s*(20\d{2})", d.title)
            return int(m.group(1)) if m else 0

        docs.sort(key=lambda d: (-year_of(d), -d.created_at.timestamp()))
        # Prefer a ready doc; but a ready doc from an older edition should still lose
        # to a newer edition that is ready. We already sorted by year, so just take the
        # first one that is ready & chunked, else the first overall.
        for d in docs:
            if d.status == "ready" and d.chunk_count > 0:
                return d
        return docs[0]

    @staticmethod
    def _save(subject, info: dict):
        tiered = bool(info.get("tiered"))
        subject.tiers_available = ["core", "extended"] if tiered else []
        subject.save(update_fields=["tiers_available"])

        # Replace the subject's whole tree so only the latest, correct hierarchy remains.
        subject.topics.all().delete()

        for strand_order, strand in enumerate(info.get("strands", []), start=1):
            strand_title = str(strand.get("title", "")).strip()
            if not strand_title:
                continue
            strand_topic = Topic.objects.create(
                subject=subject,
                title=strand_title[:300],
                kind=Topic.Kind.STRAND,
                order=strand_order,
            )
            for sub_order, sub in enumerate(strand.get("subtopics", []), start=1):
                sub_title = str(sub.get("title", "")).strip()
                if not sub_title:
                    continue
                code = str(sub.get("code", "")).strip()
                sub_topic = Topic.objects.create(
                    subject=subject,
                    parent=strand_topic,
                    title=sub_title[:300],
                    kind=Topic.Kind.SUBTOPIC,
                    code=code[:20],
                    topic_areas=[str(a).strip() for a in sub.get("topic_areas", []) if str(a).strip()],
                    order=sub_order,
                )
                for o in sub.get("objectives", [])[:30]:
                    statement = str(o.get("statement", "")).strip()
                    if not statement:
                        continue
                    tier = str(o.get("tier", "")).strip()
                    if tier not in ("core", "extended"):
                        tier = ""
                    try:
                        dif = max(1, min(5, int(o.get("difficulty", 2) or 2)))
                    except (TypeError, ValueError):
                        dif = 2
                    oc = str(o.get("code", "")).strip()
                    LearningObjective.objects.create(
                        topic=sub_topic,
                        code=oc[:20],
                        statement=statement[:500],
                        difficulty=dif,
                        tier=tier,
                    )