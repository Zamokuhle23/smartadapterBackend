from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import ChatSession, Message
from .services.routing import thread_list


class MessageSerializer(serializers.ModelSerializer):
    topic_id = serializers.IntegerField(read_only=True, allow_null=True)

    class Meta:
        model = Message
        fields = ("id", "role", "content", "topic_id", "meta", "created_at")


class ChatSessionSerializer(serializers.ModelSerializer):
    syllabus_name = serializers.CharField(
        source="syllabus.name", read_only=True, allow_null=True, required=False
    )
    subject_name = serializers.CharField(
        source="subject.name", read_only=True, allow_null=True, required=False
    )

    class Meta:
        model = ChatSession
        fields = (
            "id",
            "syllabus",
            "subject",
            "title",
            "syllabus_name",
            "subject_name",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("title", "created_at", "updated_at")


class ChatSessionViewSet(viewsets.ModelViewSet):
    """
    Tutoring chat sessions. Sessions are always scoped to the requesting student.
    Real-time conversation happens over WebSocket at ws/chat/<id>/;
    this REST viewset is for listing/history and offline-friendly clients.
    """

    serializer_class = ChatSessionSerializer

    def get_queryset(self):
        return ChatSession.objects.filter(student=self.request.user).select_related(
            "syllabus", "subject"
        )

    def perform_create(self, serializer):
        serializer.save(student=self.request.user)

    def retrieve(self, request, *args, **kwargs):
        session = self.get_object()
        serializer = self.get_serializer(session)
        # Optional thread scoping: ?topic=main -> root chat, ?topic=<id> -> one subtopic.
        topic_id = request.query_params.get("topic")
        if topic_id == "main":
            msgs = session.messages.filter(topic__isnull=True).order_by("created_at", "id")
        elif topic_id:
            try:
                tid = int(topic_id)
            except (TypeError, ValueError):
                from rest_framework import status

                return Response(
                    {"detail": "Invalid topic id."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            msgs = session.messages.filter(topic_id=tid).order_by("created_at", "id")
        else:
            msgs = session.messages.all().order_by("created_at", "id")
        messages = MessageSerializer(msgs, many=True).data
        return Response({**serializer.data, "messages": messages})

    @action(detail=False, methods=["post"])
    def open(self, request):
        """Get-or-create the student's single chat session for a subject.

        Body: {"syllabus": <id>, "subject": <id|null>}. One session per
        (student, syllabus, subject); subtopic threads live inside it, so
        clients open this instead of spawning "new chats".
        """
        syllabus_id = request.data.get("syllabus")
        subject_id = request.data.get("subject")
        if not syllabus_id:
            from rest_framework import status

            return Response({"detail": "syllabus is required."},
                            status=status.HTTP_400_BAD_REQUEST)
        from apps.syllabus.models import Subject, Syllabus

        try:
            syllabus = Syllabus.objects.get(pk=syllabus_id)
        except Syllabus.DoesNotExist:
            from rest_framework import status

            return Response({"detail": "Unknown syllabus."},
                            status=status.HTTP_400_BAD_REQUEST)
        subject = None
        if subject_id is not None:
            try:
                subject = Subject.objects.get(pk=subject_id, syllabus=syllabus)
            except Subject.DoesNotExist:
                from rest_framework import status

                return Response({"detail": "Unknown subject for this syllabus."},
                                status=status.HTTP_400_BAD_REQUEST)
        session, _ = ChatSession.objects.get_or_create(
            student=request.user, syllabus=syllabus, subject=subject,
        )
        return Response(self.get_serializer(session).data)

    @action(detail=True, methods=["get"])
    def threads(self, request, pk=None):
        """Ordered list of main chat + subtopic threads for this session."""
        session = self.get_object()
        return Response(thread_list(session))
