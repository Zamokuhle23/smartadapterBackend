from rest_framework import serializers, viewsets
from rest_framework.response import Response

from .models import ChatSession, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ("id", "role", "content", "meta", "created_at")


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
        messages = MessageSerializer(session.messages.all(), many=True).data
        return Response({**serializer.data, "messages": messages})
