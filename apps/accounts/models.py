from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Student/teacher account with Eswatini education-system context."""

    class Level(models.TextChoices):
        JC = "JC", "Junior Certificate (Forms 1-3)"
        EGCSE = "EGCSE", "Eswatini GCSE (Forms 4-5)"
        AS_LEVEL = "AS", "AS Level"
        A_LEVEL = "A2", "A Level"
        OTHER = "OTHER", "Other"

    phone_number = models.CharField(max_length=20, blank=True)
    level = models.CharField(
        max_length=10, choices=Level.choices, default=Level.EGCSE
    )
    form_level = models.PositiveIntegerField(null=True, blank=True)
    school_name = models.CharField(max_length=200, blank=True)


class LearnerProfile(models.Model):
    """
    Personalization preferences for one learner.

    The tutor agent reads this before every reply so it can adapt:
    - preferred_language: English, siSwati, or code-switching mix
    - learning_style: Socratic questioning vs direct explanation (AUTO learns it)
    - pace: how much scaffolding per explanation
    """

    class Language(models.TextChoices):
        ENGLISH = "en", "English"
        SISWATI = "ss", "siSwati"
        MIXED = "mix", "English + siSwati"

    class Style(models.TextChoices):
        SOCRATIC = "socratic", "Socratic (guided questions)"
        DIRECT = "direct", "Direct explanation"
        AUTO = "auto", "Let the tutor decide"

    class Pace(models.TextChoices):
        GENTLE = "gentle", "Gentle, lots of scaffolding"
        NORMAL = "normal", "Normal"
        FAST = "fast", "Fast, challenge me"

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    preferred_language = models.CharField(
        max_length=4, choices=Language.choices, default=Language.ENGLISH
    )
    learning_style = models.CharField(
        max_length=10, choices=Style.choices, default=Style.AUTO
    )
    pace = models.CharField(max_length=8, choices=Pace.choices, default=Pace.NORMAL)
    diagnostic_complete = models.BooleanField(default=False)
    daily_goal_minutes = models.PositiveIntegerField(default=30)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Learner profile"
        verbose_name_plural = "Learner profiles"

    def __str__(self):
        return f"Profile<{self.user.username}>"

    @classmethod
    def for_user(cls, user) -> "LearnerProfile":
        profile, _ = cls.objects.get_or_create(user=user)
        return profile
