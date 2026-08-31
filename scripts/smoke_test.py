"""End-to-end smoke test: ingest -> retrieve -> attempt/BKT -> tutor reply -> dashboard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hermetic test: never hit the real LLM even when an API key is configured.
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()


from django.contrib.auth import get_user_model
from apps.accounts.models import LearnerProfile
from apps.progress.services.bkt import update_mastery
from apps.rag.models import DocumentChunk
from apps.syllabus.models import LearningObjective, Subject, SyllabusDocument, Topic
from apps.syllabus.services.ingestion import process_document

User = get_user_model()

# --- setup ---
user, _ = User.objects.get_or_create(
    username="smoke_student",
    defaults={"level": "EGCSE", "form_level": 4},
)
user.set_password("SmokeTest123!")
user.save()
LearnerProfile.for_user(user)
syllabus = __import__("apps.syllabus.models", fromlist=["Syllabus"]).Syllabus.objects.get(level="EGCSE")
maths = Subject.objects.get(syllabus=syllabus, code="6880")

# --- 1. ingestion ---
from django.core.files.base import ContentFile

src_path = os.path.join(os.path.dirname(__file__), "_smoke_syllabus_doc.txt")
with open(src_path, "r", encoding="utf-8") as fh:
    doc_text = fh.read()

# Clean up leftovers from previous runs so the corpus stays consistent.
SyllabusDocument.objects.filter(title="EGCSE Maths sample notes").delete()

doc = SyllabusDocument.objects.create(
    syllabus=syllabus, subject=maths, title="EGCSE Maths sample notes"
)
doc.file.save("smoke_syllabus_doc.txt", ContentFile(doc_text.encode("utf-8")), save=True)
n_chunks = process_document(doc)

chunks = DocumentChunk.objects.filter(document=doc)
assert n_chunks > 0 and all(c.embedding for c in chunks), "ingestion failed"
print(f"[1] Ingestion OK - {n_chunks} chunks embedded")

# --- 2. objectives + attempts -> BKT ---
topic = Topic.objects.create(subject=maths, title="Algebraic fractions")
obj_weak = LearningObjective.objects.create(topic=topic, statement="Simplify algebraic fractions")
obj_ok = LearningObjective.objects.create(topic=topic, statement="Factorise quadratic trinomials")

from apps.progress.models import MasteryEvent, MasteryRecord

m0 = update_mastery(None, True)
m1 = update_mastery(m0, True)
m2 = update_mastery(m1, False)
assert m1 > m0 >= 0.30 and m2 < m1, f"BKT monotonicity broken: {m0} {m1} {m2}"
print(f"[2] BKT OK - mastery path after correct/correct/wrong: {m0} -> {m1} -> {m2}")

for correct in (False, False, True):
    MasteryEvent.objects.create(student=user, objective=obj_weak, correct=correct)
    rec, _ = MasteryRecord.objects.get_or_create(student=user, objective=obj_weak)
    rec.attempts += 1
    rec.mastery = update_mastery(rec.mastery, correct)
    rec.save()
MasteryEvent.objects.create(student=user, objective=obj_ok, correct=True)
rec, _ = MasteryRecord.objects.get_or_create(student=user, objective=obj_ok)
rec.mastery = update_mastery(rec.mastery, True)
rec.save()

# --- 3. personalized tutor reply (offline provider) ---
from apps.tutoring.models import ChatSession
from apps.tutoring.services.orchestrator import generate_reply

session = ChatSession.objects.create(student=user, syllabus=syllabus, subject=maths)
reply, meta = generate_reply(session, "How do I simplify an algebraic fraction like 6x/9?")
assert "[offline mode" in reply and "algebraic" in reply.lower(), reply[:200]
assert meta["provider"] == "OfflineTutorProvider"
print(f"[3] Tutor reply OK - retrieved {len(meta['retrieved_chunk_ids'])} chunks, "
      f"{len(reply)} chars, weaknesses woven into prompt")

# --- 4. dashboard analytics ---
from apps.progress.services.dashboard import study_recommendations, subject_summary

summary = subject_summary(user)
recs = study_recommendations(user)
assert summary and summary[0]["subject"] == "Mathematics", summary
assert recs and recs[0]["objective_statement"].startswith("Simplify"), recs[0]
print(f"[4] Dashboard OK - weakest subject: Mathematics @ {summary[0]['avg_mastery']:.0%}; "
      f"top recommendation: '{recs[0]['recommended_action']}' on '{recs[0]['objective_statement'][:40]}...'")

print("\nALL SMOKE TESTS PASSED")
