from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import ChatSession, Message
from .services.routing import thread_list


class MessageSerializer(serializers.ModelSerializer):
    topic_id = serializers.IntegerField(source="topic_id", read_only=True)

    class Meta:
        model = Message
        fields = ("id", "role", "content", "topic_id", "meta", "created_at")


class ChatSessionSerializer(serializers.ModelSerializer):
    syllabus_name = serializers.CharField(source="syllabus.name", read_only=True)
    subject_name = serializers.CharField(source="subject.name", read_only=True)

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
            msgs = session.messages.filter(topic__isnull=True)
        elif topic_id:
            msgs = session.messages.filter(topic_id=topic_id)
        else:
            msgs = session.messages.all()
        messages = MessageSerializer(msgs, many=True).data
        return Response({**serializer.data, "messages": messages})

    @action(detail=True, methods=["get"])
    def threads(self, request, pk=None):
        """Ordered list of main chat + subtopic threads for this session."""
        session = self.get_object()
        return Response(thread_list(session))
