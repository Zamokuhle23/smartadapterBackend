from django.db import models

from apps.accounts.models import User
from apps.syllabus.models import LearningObjective, Subject, SyllabusDocument


class QuizQuestion(models.Model):
    """
    An LLM-generated, syllabus-grounded question tagged to a learning objective.

    Provenance fields record whether the item is a variation of a real
    past-paper question (preferred!) or freshly generated from the syllabus,
    plus which ECESWA paper its format mirrors (Paper 1 = MCQ style,
    Paper 2 = structured/long-form, etc.).
    """

    class Format(models.TextChoices):
        MCQ = "mcq", "Multiple choice"
        STRUCTURED = "structured", "Structured / free response"

    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="questions")
    objective = models.ForeignKey(
        LearningObjective, null=True, blank=True, on_delete=models.SET_NULL, related_name="questions"
    )
    topic_title = models.CharField(max_length=300, blank=True)
    difficulty = models.PositiveSmallIntegerField(default=2)  # 1..5
    format = models.CharField(
        max_length=12, choices=Format.choices, default=Format.MCQ
    )
    question_text = models.TextField()
    options = models.JSONField(default=list)  # list[str], index 0..n (empty for structured)
    correct_index = models.PositiveSmallIntegerField(null=True, blank=True)  # null for structured
    explanation = models.TextField(blank=True)
    marks = models.PositiveSmallIntegerField(default=1)
    marking_guidance = models.TextField(blank=True)  # model answer + mark allocation for grading
    paper_label = models.CharField(max_length=40, blank=True)  # e.g. "Paper 2"
    source_year = models.PositiveIntegerField(null=True, blank=True)
    source = models.CharField(
        max_length=10, choices=SyllabusDocument.Source.choices, blank=True
    )  # "igcse" (Cambridge, primary) or "egcse" (ECESWA, secondary)
    adapted_from_past_paper = models.BooleanField(default=False)
    source_chunk_ids = models.JSONField(null=True, blank=True)
    figures = models.ManyToManyField(
        "rag.DocumentFigure", blank=True, related_name="questions"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Q<{self.subject.code} d{self.difficulty} {self.get_format_display()}> {self.question_text[:60]}"


class ExamBlueprint(models.Model):
    """
    Cached exam structure for one subject + paper: topic weightings, question
    counts and formats, derived by the LLM from the syllabus's assessment
    scheme so simulations mirror the real paper's proportions.
    """

    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="blueprints")
    paper_number = models.PositiveSmallIntegerField(default=1)
    data = models.JSONField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("subject", "paper_number")

    def __str__(self):
        return f"Blueprint<{self.subject.code} paper {self.paper_number}>"


class ExamSession(models.Model):
    """One simulated exam sitting. Questions are generated lazily, following the blueprint queue."""

    class Status(models.TextChoices):
        IN_PROGRESS = "in_progress", "In progress"
        COMPLETED = "completed", "Completed"

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="exam_sessions")
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="exam_sessions")
    paper_number = models.PositiveSmallIntegerField(default=1)
    title = models.CharField(max_length=200, blank=True)  # e.g. "Mathematics (6880) - Paper 1"
    duration_minutes = models.PositiveIntegerField(null=True, blank=True)
    total_questions = models.PositiveSmallIntegerField(default=0)
    plan = models.JSONField(default=dict)  # blueprint snapshot incl. flattened question queue
    question_ids = models.JSONField(default=list)  # generated so far, in order
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.IN_PROGRESS)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"Exam<{self.student} {self.title or self.subject.code}>"


class QuizAttempt(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="quiz_attempts")
    question = models.ForeignKey(QuizQuestion, on_delete=models.CASCADE, related_name="attempts")
    selected_index = models.PositiveSmallIntegerField(null=True, blank=True)  # MCQ answers
    answer_text = models.TextField(blank=True)  # structured / free-response answers
    awarded_marks = models.FloatField(null=True, blank=True)  # LLM-rubric graded structured answers
    feedback = models.TextField(blank=True)
    correct = models.BooleanField()
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
