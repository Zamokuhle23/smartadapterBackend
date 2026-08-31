from django.contrib import admin

from .models import ChatSession, Message


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "student", "syllabus", "subject", "title", "updated_at")
    search_fields = ("student__username", "title")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "role", "short_content", "created_at")
    list_filter = ("role",)

    @admin.display(description="Content")
    def short_content(self, obj):
        return obj.content[:80]
