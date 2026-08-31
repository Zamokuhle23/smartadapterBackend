from django.db import models

from apps.accounts.models import User
from apps.syllabus.models import Subject, Syllabus


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
    meta = models.JSONField(null=True, blank=True)  # retrieved chunk ids, provider info
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_at",)

    def __str__(self):
        return f"{self.get_role_display()}: {self.content[:60]}"

