from rest_framework import permissions, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import Enrollment, LearningObjective, Subject, Syllabus, SyllabusDocument, Topic
from .tasks import ingest_document_task


class SyllabusSerializer(serializers.ModelSerializer):
    subjects_count = serializers.IntegerField(source="subjects.count", read_only=True)

    class Meta:
        model = Syllabus
        fields = (
            "id",
            "level",
            "name",
            "version",
            "status",
            "description",
            "subjects_count",
        )


class SubjectSerializer(serializers.ModelSerializer):
    has_tiers = serializers.BooleanField(read_only=True)
    tiers_available = serializers.SerializerMethodField()

    class Meta:
        model = Subject
        fields = ("id", "syllabus", "code", "name", "has_tiers", "tiers_available")

    def get_tiers_available(self, obj):
        return obj.tiers_available


class TopicSerializer(serializers.ModelSerializer):
    objectives = serializers.SerializerMethodField()

    class Meta:
        model = Topic
        fields = ("id", "subject", "parent", "title", "order", "objectives")

    def get_objectives(self, obj):
        return [
            {"id": o.id, "statement": o.statement, "difficulty": o.difficulty}
            for o in obj.objectives.all()
        ]


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = SyllabusDocument
        fields = (
            "id",
            "syllabus",
            "subject",
            "title",
            "file",
            "status",
            "chunk_count",
            "error",
            "created_at",
            "doc_type",
            "year",
            "paper_number",
            "session",
        )
        read_only_fields = ("status", "chunk_count", "error", "created_at")


class SyllabusViewSet(viewsets.ReadOnlyModelViewSet):
    """List/retrieve syllabuses; filter with ?level=EGCSE|JC|..."""

    queryset = Syllabus.objects.filter(status=Syllabus.Status.PUBLISHED).prefetch_related(
        "subjects"
    )
    serializer_class = SyllabusSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        if self.request.query_params.get("all") == "1" and self.request.user.is_staff:
            return Syllabus.objects.all()
        level = self.request.query_params.get("level")
        if level:
            qs = qs.filter(level=level)
        return qs


class SubjectViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Subject.objects.select_related("syllabus")
    serializer_class = SubjectSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        syllabus = self.request.query_params.get("syllabus")
        if syllabus:
            qs = qs.filter(syllabus_id=syllabus)
        return qs


class TopicViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Topic.objects.select_related("subject").prefetch_related("objectives")
    serializer_class = TopicSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        subject = self.request.query_params.get("subject")
        parent = self.request.query_params.get("parent")
        if subject:
            qs = qs.filter(subject_id=subject)
        if parent:
            qs = qs.filter(parent_id=parent)
        return qs


class EnrollRequestSerializer(serializers.Serializer):
    subject_id = serializers.IntegerField()
    tier = serializers.ChoiceField(
        choices=Subject.Tier.choices, required=False, allow_blank=True
    )


class EnrollmentSerializer(serializers.ModelSerializer):
    subject = SubjectSerializer(read_only=True)

    class Meta:
        model = Enrollment
        fields = ("id", "subject", "tier", "joined_at")


class EnrollmentViewSet(viewsets.ModelViewSet):
    """
    The student's subject workspaces ("projects").
    GET    /api/my-subjects/          -> enrolled subjects
    POST   /api/my-subjects/ {"subject_id": N}
    DELETE /api/my-subjects/{id}/
    """

    serializer_class = EnrollmentSerializer
    http_method_names = ["get", "post", "delete", "head", "options"]

    def get_queryset(self):
        return Enrollment.objects.filter(student=self.request.user).select_related(
            "subject__syllabus"
        )

    def create(self, request, *args, **kwargs):
        payload = EnrollRequestSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        try:
            subject = Subject.objects.select_related("syllabus").get(
                pk=payload.validated_data["subject_id"]
            )
        except Subject.DoesNotExist:
            return Response({"detail": "Unknown subject_id"}, status=status.HTTP_400_BAD_REQUEST)
        tier = (payload.validated_data.get("tier") or "").strip()
        # Validate the tier is allowed for this subject (if subject is tiered).
        if tier and subject.tiers_available and tier not in subject.tiers_available:
            return Response(
                {"detail": f"Tier '{tier}' not offered for {subject.name}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if subject.has_tiers() and not tier:
            return Response(
                {"detail": f"{subject.name} requires a tier (core or extended)"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        enrollment, _created = Enrollment.objects.update_or_create(
            student=request.user, subject=subject, defaults={"tier": tier}
        )
        return Response(self.get_serializer(enrollment).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, instance):
        if instance.student_id != self.request.user.id:
            raise permissions.PermissionDenied()
        instance.delete()


class DocumentUploadViewSet(viewsets.ModelViewSet):
    """
    Plug-and-play syllabus ingestion:
    POST a document (PDF/TXT/DOCX) tied to a syllabus (+optional subject);
    it is chunked and embedded asynchronously, then available to the tutor RAG.
    """

    queryset = SyllabusDocument.objects.select_related("syllabus", "subject")
    serializer_class = DocumentSerializer
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        subject = self.request.query_params.get("subject")
        if subject:
            qs = qs.filter(subject_id=subject)
        doc_type = self.request.query_params.get("doc_type")
        if doc_type:
            qs = qs.filter(doc_type=doc_type)
        return qs

    def get_permissions(self):
        # Only staff may upload content; everyone authenticated may list/read.
        if self.action in ("create", "update", "partial_update", "destroy"):
            return [permissions.IsAdminUser()]
        return super().get_permissions()

    def perform_create(self, serializer):
        document = serializer.save(uploaded_by=self.request.user)
        try:
            ingest_document_task.delay(document.pk)
        except Exception:
            # No broker configured (dev): process inline so it still works.
            from .services.ingestion import process_document

            process_document(document)


    @action(detail=True, methods=["post"])
    def reingest(self, request, pk=None):
        document = self.get_object()
        ingest_document_task.delay(document.pk)
        return Response({"status": "queued"})
