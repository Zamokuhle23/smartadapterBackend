from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import LearnerProfile, User


@admin.register(User)
class FundzaUserAdmin(UserAdmin):
    list_display = ("username", "email", "level", "form_level", "school_name")
    fieldsets = UserAdmin.fieldsets + (
        ("FundzaAI", {"fields": ("phone_number", "level", "form_level", "school_name")}),
    )


@admin.register(LearnerProfile)
class LearnerProfileAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "preferred_language",
        "learning_style",
        "pace",
        "diagnostic_complete",
    )
    list_filter = ("preferred_language", "learning_style", "pace")
