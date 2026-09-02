from django.db import models

from apps.accounts.models import User
from apps.syllabus.models import Subject, Syllabus, Topic


class ChatSession(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_sessions")
    syllabus = models.ForeignKey(Syllabus, on_delete=models.PROTECT, related_name="chat_sessions")
    subject = models.ForeignKey(
        Subject, null=True, blank=True, on_delete=models.PROTECT, related_name="chat_sessions"
    )
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self):
        return f"Chat<{self.student.username} | {self.syllabus.name}{(' / ' + self.subject.name) if self.subject else ''}>"

    def save(self, *args, **kwargs):
        if not self.title:
            self.title = f"{self.syllabus.name}" + (f" - {self.subject.name}" if self.subject else "")
        super().save(*args, **kwargs)


class Message(models.Model):
    class Role(models.TextChoices):
        USER = "user", "Student"
        TUTOR = "tutor", "Tutor"

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=8, choices=Role.choices)
    content = models.TextField()
    # The subtopic this message belongs to. None = root "main chat". Messages are
    # grouped by this to form auto-created subtopic threads for the subject.
    topic = models.ForeignKey(
        Topic, null=True, blank=True, on_delete=models.SET_NULL, related_name="messages"
    )
    meta = models.JSONField(null=True, blank=True)  # retrieved chunk ids, provider info
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_at",)

    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:60]}"


class MemoryEntry(models.Model):
    """
    A durable fact remembered about a student, persisted across conversations.

    Facts are written by the LLM during tutoring (extracted from what the student
    says), embedded for similarity, and retrieved on-demand when a new message is
    topically relevant. `kind` separates:

      - always_on: global, high-value traits (e.g. "student is weak at Maths") that
        are retrieved at a LOW threshold so they are usually injected.
      - situational: specific facts (e.g. "struggles with fractions", "exam in Feb")
        retrieved only when the current message is similar enough (threshold-gated).
    """

    class Kind(models.TextChoices):
        ALWAYS_ON = "always_on", "Always-on trait"
        SITUATIONAL = "situational", "Situational fact"

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="memory_entries")
    fact = models.TextField()
    kind = models.CharField(max_length=12, choices=Kind.choices, default=Kind.SITUATIONAL)
    importance = models.PositiveSmallIntegerField(default=5)  # 1..10 (>=8 promoted to always_on)
    embedding = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-importance", "-updated_at")

    def __str__(self):
        return f"Mem<{self.student.username} [{self.kind}] {self.fact[:50]}>"

