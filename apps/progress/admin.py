from django.contrib import admin

from .models import MasteryEvent, MasteryRecord


@admin.register(MasteryRecord)
class MasteryRecordAdmin(admin.ModelAdmin):
    list_display = ("student", "objective", "mastery", "attempts", "correct_count")
    list_filter = ("objective__topic__subject",)


@admin.register(MasteryEvent)
class MasteryEventAdmin(admin.ModelAdmin):
    list_display = ("student", "objective", "correct", "latency_ms", "hints_used", "created_at")
    list_filter = ("correct",)
