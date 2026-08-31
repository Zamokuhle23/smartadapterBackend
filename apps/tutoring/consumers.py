import json

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer


class ChatConsumer(AsyncWebsocketConsumer):
    """
    Real-time tutoring chat.

    Connect: ws/chat/<session_id>/
    Send:    {"content": "explain factorising trinomials"}
    Receive: {"role": "user", "content": ...} then {"role": "tutor", "content": ..., "meta": {...}}
    """

    async def connect(self):
        self.session_id = int(self.scope["url_route"]["kwargs"]["session_id"])
        user = self.scope.get("user")
        if not user or not user.is_authenticated:
            await self.close(code=4401)
            return
        self.session = await self._get_session(user)
        if self.session is None:
            await self.close(code=4404)
            return
        self.group_name = f"chat_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        try:
            payload = json.loads(text_data or "{}")
            content = str(payload.get("content", "")).strip()
        except json.JSONDecodeError:
            content = ""
        if not content:
            return

        await self._save_message("user", content)
        await self.send_json({"role": "user", "content": content})

        reply, meta = await database_sync_to_async(self._generate)(content)
        await self._save_message("tutor", reply, meta)
        await self.send_json({"role": "tutor", "content": reply, "meta": meta})

    def _generate(self, content: str):
        from .services.orchestrator import generate_reply

        return generate_reply(self.session, content)

    @database_sync_to_async
    def _get_session(self, user):
        from .models import ChatSession

        return (
            ChatSession.objects.filter(pk=self.session_id, student=user).first()
        )

    @database_sync_to_async
    def _save_message(self, role: str, content: str, meta=None):
        from .models import Message

        Message.objects.create(session=self.session, role=role, content=content, meta=meta)
        # Auto-title the session from the first student message.
        default_title = f"{self.session.syllabus.name}" + (
            f" - {self.session.subject.name}" if self.session.subject else ""
        )
        if role == "user" and (self.session.title or "") == default_title:
            self.session.title = content[:60]
            self.session.save(update_fields=["title"])

    async def send_json(self, data: dict):
        await self.send(text_data=json.dumps(data))
