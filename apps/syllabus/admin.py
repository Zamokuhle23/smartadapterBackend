from django.contrib import admin

from .models import Enrollment, LearningObjective, Subject, Syllabus, SyllabusDocument, Topic


@admin.register(Syllabus)
class SyllabusAdmin(admin.ModelAdmin):
    list_display = ("name", "level", "version", "status")
    list_filter = ("level", "status")


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "syllabus")
    list_filter = ("syllabus",)
    search_fields = ("name", "code")


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ("title", "subject", "parent", "order")
    list_filter = ("subject",)
    search_fields = ("title",)


@admin.register(LearningObjective)
class LearningObjectiveAdmin(admin.ModelAdmin):
    list_display = ("short_statement", "topic", "difficulty")
    search_fields = ("statement",)

    @admin.display(description="Objective")
    def short_statement(self, obj):
        return obj.statement[:90]


@admin.register(SyllabusDocument)
class SyllabusDocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "subject", "doc_type", "source", "paper_number", "year", "status", "chunk_count")
    list_filter = ("status", "source", "doc_type", "syllabus")


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    list_display = ("student", "subject", "joined_at")
    list_filter = ("subject__syllabus",)

