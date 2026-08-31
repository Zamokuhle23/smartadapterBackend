from django.contrib import admin

from .models import ExamBlueprint, ExamSession, QuizAttempt, QuizQuestion


@admin.register(QuizQuestion)
class QuizQuestionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "subject",
        "topic_title",
        "difficulty",
        "format",
        "marks",
        "paper_label",
        "source",
        "adapted_from_past_paper",
        "short_question",
        "created_at",
    )
    list_filter = ("subject", "difficulty", "format", "source", "adapted_from_past_paper")

    @admin.display(description="Question")
    def short_question(self, obj):
        return obj.question_text[:70]


@admin.register(QuizAttempt)
class QuizAttemptAdmin(admin.ModelAdmin):
    list_display = (
        "student",
        "question",
        "selected_index",
        "awarded_marks",
        "correct",
        "created_at",
    )
    list_filter = ("correct",)


@admin.register(ExamBlueprint)
class ExamBlueprintAdmin(admin.ModelAdmin):
    list_display = ("subject", "paper_number", "updated_at")


@admin.register(ExamSession)
class ExamSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "student", "title", "total_questions", "status", "created_at")
    list_filter = ("status",)
