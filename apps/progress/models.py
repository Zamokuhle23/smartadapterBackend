from django.db import models

from apps.accounts.models import User
from apps.syllabus.models import LearningObjective


class MasteryEvent(models.Model):
    """One raw attempt by a student on an objective - the evidence stream."""

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="mastery_events")
    objective = models.ForeignKey(
        LearningObjective, on_delete=models.CASCADE, related_name="events"
    )
    correct = models.BooleanField()
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    hints_used = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name_plural = "Mastery events"


class MasteryRecord(models.Model):
    """Current BKT mastery estimate for (student, objective)."""

    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name="mastery_records")
    objective = models.ForeignKey(
        LearningObjective, on_delete=models.CASCADE, related_name="mastery_records"
    )
    mastery = models.FloatField(default=0.30)  # BKT P_INIT
    attempts = models.PositiveIntegerField(default=0)
    correct_count = models.PositiveIntegerField(default=0)
    last_reviewed_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("student", "objective")

    def __str__(self):
        return f"Mastery<{self.student.username} obj={self.objective_id} {self.mastery:.2f}>"
