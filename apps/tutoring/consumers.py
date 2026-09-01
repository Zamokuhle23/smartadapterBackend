import asyncio
import json
import queue
import threading

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer


class ChatConsumer(AsyncWebsocketConsumer):
    """Live tutoring channel: text + voice.

    Text in:  {"content": "..."}                    -> classic grounded reply
    Voice in: binary PCM 16k/mono frames, then {"kind":"voice_end"}
              ({"kind":"voice_cancel"} clears the buffer - barge-in)
    Voice out: {"kind":"transcript"|"token"|"audio"|"done"|"error", ...}
    """

    async def connect(self):
        self.session_id = int(self.scope["url_route"]["kwargs"]["session_id"])
        if not self.scope.get("user") or not self.scope["user"].is_authenticated:
            await self.close(code=4401)
            return
        self.session = await self._get_session(self.scope["user"])
        if self.session is None:
            await self.close(code=4404)
            return
        self.group_name = f"chat_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        self.audio_buffer = []
        self.voice_worker_alive = False

    async def disconnect(self, code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Accumulate voice audio (16k mono signed 16-bit PCM).
            self.audio_buffer.append(bytes_data)
            return

        try:
            payload = json.loads(text_data or "{}")
        except json.JSONDecodeError:
            return

        kind = payload.get("kind", "text")
        if kind == "voice_end":
            await self._handle_voice()
            return
        if kind == "voice_cancel":
            self.audio_buffer = []  # barge-in: drop the incomplete utterance
            return

        # -------- classic text chat --------
        content = str(payload.get("content", "")).strip()
        if not content:
            return
        await self._save_message("user", content)
        await self.send_json({"role": "user", "content": content})
        reply, meta = await database_sync_to_async(self._generate)(content)
        await self._save_message("tutor", reply, meta)
        await self.send_json({"role": "tutor", "content": reply, "meta": meta})

    async def _handle_voice(self):
        pcm = b"".join(self.audio_buffer)
        self.audio_buffer = []
        if not pcm or self.voice_worker_alive:
            return
        self.voice_worker_alive = True
        out: queue.SimpleQueue = queue.SimpleQueue()

        def producer():
            try:
                from .services.voice import transcribe
                from .services.orchestrator import answer_stream
                text = transcribe(pcm)
            except Exception as exc:
                out.put(("error", str(exc)))
                return
            out.put(("transcript", text))
            out.put(("save_user", text))
            try:
                parts = []
                for ev in answer_stream(self.session, text):
                    if ev["kind"] == "token":
                        parts.append(ev["text"])
                    out.put(("event", ev))
                out.put(("save_tutor", "".join(parts)))
            except Exception as exc:
                out.put(("error", str(exc)))
            finally:
                out.put(("done", None))

        threading.Thread(target=producer, name="voice-worker", daemon=True).start()
        await self._forward(out)

    async def _forward(self, out: queue.SimpleQueue):
        while True:
            kind, data = await asyncio.to_thread(out.get)
            if kind == "done":
                break
            if kind == "error":
                await self.send_json({"kind": "error", "text": data})
            elif kind == "save_user":
                await self._save_message("user", data)
            elif kind == "save_tutor":
                await self._save_message("tutor", data)
            elif kind == "transcript":
                await self.send_json({"kind": "transcript", "text": data})
            elif kind == "event":
                ev = data
                if ev["kind"] == "audio":
                    await self.send_json({"kind": "audio", "wav_base64": ev["wav_base64"]})
                else:
                    await self.send_json(ev)
        self.voice_worker_alive = False

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
