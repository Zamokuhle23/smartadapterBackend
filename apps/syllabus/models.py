from django.db import models

from apps.accounts.models import User


class Syllabus(models.Model):
    """
    A versioned syllabus ("textbook on the shelf") for one level.

    Dynamic architecture: adding a new syllabus = creating one of these +
    uploading its documents. The RAG pipeline indexes everything against it,
    so no model retraining is ever needed.
    """

    class Level(models.TextChoices):
        PRIMARY = "PRIMARY", "Primary (EPC)"
        JC = "JC", "Junior Certificate (Forms 1-3)"
        EGCSE = "EGCSE", "Eswatini GCSE (Forms 4-5)"
        AS_LEVEL = "AS", "AS Level"
        A_LEVEL = "A2", "A Level"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        PUBLISHED = "published", "Published"
        ARCHIVED = "archived", "Archived"

    level = models.CharField(max_length=10, choices=Level.choices)
    name = models.CharField(max_length=200)
    version = models.CharField(max_length=20, default="1.0")
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.DRAFT
    )
    description = models.TextField(blank=True)

    class Meta:
        unique_together = ("level", "name", "version")
        ordering = ("level", "name")

    def __str__(self):
        return f"{self.name} ({self.level} v{self.version})"


class Subject(models.Model):
    """A subject within a syllabus, e.g. EGCSE Mathematics (6880)."""

    class Tier(models.TextChoices):
        NONE = "", "No tiering"
        CORE = "core", "Core curriculum"
        EXTENDED = "extended", "Extended curriculum"

    syllabus = models.ForeignKey(
        Syllabus, on_delete=models.CASCADE, related_name="subjects"
    )
    code = models.CharField(max_length=10, help_text="ECESWA subject code, e.g. 6880")
    name = models.CharField(max_length=200)
    # Whether this subject is offered at two tiers (core/extended) that need separate
    # curricula, papers and teaching. Populated from the syllabus's assessment scheme.
    tiers_available = models.JSONField(
        default=list,
        blank=True,
        help_text='List of tiers, e.g. ["core", "extended"]. Empty = single/un-tiered.',
    )

    class Meta:
        unique_together = ("syllabus", "code")
        ordering = ("code",)

    def __str__(self):
        return f"{self.name} ({self.code})"

    def has_tiers(self) -> bool:
        return len(self.tiers_available) >= 2


class Topic(models.Model):
    """
    A topic/subtopic node in the syllabus tree.
    Parent is null for top-level topics; nested topics form subtopics.
    """

    class Kind(models.TextChoices):
        STRAND = "strand", "Strand"
        SUBTOPIC = "subtopic", "Subtopic"
        UNKNOWN = "unknown", "Unknown"

    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="topics")
    parent = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.CASCADE, related_name="children"
    )
    title = models.CharField(max_length=300)
    kind = models.CharField(
        max_length=10, choices=Kind.choices, default=Kind.SUBTOPIC
    )  # "strand" (top-level grouping) vs "subtopic" (leaf with objectives)
    # Cross-strand topic areas this subtopic belongs to, e.g. Vectors ->
    # ["Algebra", "Shape, Position & Space"].
    topic_areas = models.JSONField(default=list, blank=True)
    code = models.CharField(max_length=20, blank=True)  # syllabus numbering, e.g. "16"
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ("order", "id")

    def __str__(self):
        return self.title


class LearningObjective(models.Model):
    """An assessable objective under a topic - the atomic skill unit tracked per student."""

    topic = models.ForeignKey(
        Topic, on_delete=models.CASCADE, related_name="objectives"
    )
    statement = models.TextField()
    code = models.CharField(max_length=20, blank=True)  # syllabus sub-number, e.g. "16.3"
    difficulty = models.PositiveSmallIntegerField(default=1)  # 1..5
    # For tiered subjects: "core" / "extended" / "" (both). Extended-only objectives
    # are not served to Core-tier students.
    tier = models.CharField(
        max_length=10, choices=Subject.Tier.choices, blank=True, default=""
    )
    prerequisites = models.ManyToManyField("self", symmetrical=False, blank=True)

    def __str__(self):
        return self.statement[:80]


class SyllabusDocument(models.Model):
    """
    An uploaded source document (syllabus PDF, notes, past paper) that feeds RAG.
    Ingestion: extract text -> chunk -> embed -> store DocumentChunk rows.
    """

    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    class DocType(models.TextChoices):
        SYLLABUS = "syllabus", "Official syllabus"
        NOTES = "notes", "Study notes"
        PAST_PAPER = "past_paper", "Past exam paper"
        MARK_SCHEME = "mark_scheme", "Mark scheme"

    class Source(models.TextChoices):
        IGCSE = "igcse", "Cambridge IGCSE"  # Primary source (5,300+ papers)
        EGCSE = "egcse", "Eswatini GCSE"   # Secondary source (local papers)

    syllabus = models.ForeignKey(Syllabus, on_delete=models.CASCADE, related_name="documents")
    subject = models.ForeignKey(
        Subject, null=True, blank=True, on_delete=models.SET_NULL, related_name="documents"
    )
    title = models.CharField(max_length=300)
    doc_type = models.CharField(
        max_length=12, choices=DocType.choices, default=DocType.NOTES
    )
    paper_number = models.PositiveSmallIntegerField(null=True, blank=True)  # 1, 2, 3...
    year = models.PositiveIntegerField(null=True, blank=True)  # exam sitting year
    source = models.CharField(
        max_length=10, choices=Source.choices, default=Source.EGCSE
    )  # IGCSE (primary) or EGCSE (secondary) source flag
    file = models.FileField(upload_to="syllabus_docs/%Y/%m/")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.UPLOADED)
    chunk_count = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True)
    uploaded_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return self.title


class Enrollment(models.Model):
    """
    A subject the student has enrolled in - their persistent subject workspace
    (like a ChatGPT "Project"). All chats, mastery and recommendations for that
    subject are grouped under this single entry.
    """

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="enrollments")
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="enrollments")
    # For tiered subjects (core/extended), which curriculum this student follows.
    # Empty for un-tiered subjects. Drives which papers/topics are served.
    tier = models.CharField(
        max_length=10, choices=Subject.Tier.choices, blank=True, default=""
    )
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("student", "subject")
        ordering = ("subject__code",)

    def __str__(self):
        tier = f" ({self.tier})" if self.tier else ""
        return f"{self.student.username} → {self.subject}{tier}"


def user_is_student(user) -> bool:
    return bool(user and user.is_authenticated)
