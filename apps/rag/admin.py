from django.contrib import admin

from .models import DocumentChunk, DocumentFigure


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "syllabus", "document", "subject", "ordinal", "page_number")
    search_fields = ("text",)


@admin.register(DocumentFigure)
class DocumentFigureAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "page_number", "ordinal")
    list_filter = ("document",)
