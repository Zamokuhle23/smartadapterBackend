import asyncio
import json
import logging
import queue
import threading

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)


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
        # Subtopic the current user turn is routed to (None = main chat).
        self.current_topic_id = None

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

        # -------- classic text chat (streamed) --------
        content = str(payload.get("content", "")).strip()
        if not content:
            return
        # The thread the student is viewing (None = main chat). Used as the
        # fallback when the message itself carries no clear subtopic signal
        # (follow-ups like "show me the steps" continue in place).
        client_topic = await self._client_topic(payload.get("topic_id"))
        self.current_topic_id = client_topic.id if client_topic else None
        topic = await self._route_topic(content, self.current_topic_id)
        self.current_topic_id = topic.id if topic else None
        await self._save_message("user", content, topic=topic)
        # Instant feedback: typing indicator first (LLM + retrieval take a
        # while), then token frames as they generate, then text_done.
        # The client already echoed the user's own message locally, so no
        # user-echo frame is sent (it used to linger as a permanent ⏳).
        await self.send_json({
            "kind": "typing", "typing": True,
            "topic_id": topic.id if topic else None,
        })
        await self._stream_text_reply(content)
        return

    async def _stream_text_reply(self, content: str):
        """Stream one chat turn: tokens live, then save + text_done frame."""
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()
        topic_id = self.current_topic_id

        def producer():
            try:
                from .services.orchestrator import stream_chat
                topic = self._topic_obj_sync(topic_id)
                deltas, chunk_ids = stream_chat(self.session, content, topic=topic)
                loop.call_soon_threadsafe(out.put_nowait, ("chunks", chunk_ids))
                for delta in deltas:
                    if delta:
                        loop.call_soon_threadsafe(out.put_nowait, ("token", delta))
            except Exception as exc:  # noqa: BLE001 - forward, then apologise below
                loop.call_soon_threadsafe(out.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(out.put_nowait, ("done", None))

        threading.Thread(target=producer, name="chat-worker", daemon=True).start()

        parts: list[str] = []
        chunk_ids: list = []
        while True:
            kind, data = await out.get()
            if kind == "done":
                break
            if kind == "chunks":
                chunk_ids = list(data or [])
            elif kind == "token":
                parts.append(data)
                await self.send_json({"kind": "token", "text": data})
            elif kind == "error":
                logger.warning("Chat stream error: %s", data)
        reply = "".join(parts)
        if not reply.strip():
            reply = ("Sorry, I could not generate a reply just now - "
                     "please try again.")
            await self.send_json({"kind": "token", "text": reply})
        from .services.orchestrator import _split_key_terms
        reply, key_terms = _split_key_terms(reply)
        topic = await database_sync_to_async(self._topic_obj)()
        meta = {"retrieved_chunk_ids": chunk_ids, "key_terms": key_terms}
        await self._save_message("tutor", reply, meta, topic=topic)
        await self.send_json({
            "kind": "text_done",
            "topic_id": topic.id if topic else None,
            "key_terms": key_terms,
        })

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
                from .services.routing import classify_topic
                text = transcribe(pcm)
            except Exception as exc:
                out.put(("error", str(exc)))
                return
            out.put(("transcript", text))
            try:
                fallback = self._topic_obj_sync(self.current_topic_id)
                topic = classify_topic(self.session, text, fallback=fallback)
            except Exception:
                topic = None
            out.put(("topic", topic.id if topic else None))
            out.put(("save_user", text))
            try:
                parts = []
                for ev in answer_stream(self.session, text, topic):
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
            if kind == "topic":
                self.current_topic_id = data
            elif kind == "error":
                await self.send_json({"kind": "error", "text": data})
            elif kind == "save_user":
                topic = await database_sync_to_async(self._topic_obj)()
                await self._save_message("user", data, topic=topic)
            elif kind == "save_tutor":
                topic = await database_sync_to_async(self._topic_obj)()
                await self._save_message("tutor", data, topic=topic)
            elif kind == "transcript":
                await self.send_json({"kind": "transcript", "text": data})
            elif kind == "event":
                ev = data
                if ev["kind"] == "audio":
                    await self.send_json({"kind": "audio", "wav_base64": ev["wav_base64"]})
                else:
                    await self.send_json(ev)
        self.voice_worker_alive = False

    @database_sync_to_async
    def _get_session(self, user):
        from .models import ChatSession

        return (
            ChatSession.objects.filter(pk=self.session_id, student=user).first()
        )

    @database_sync_to_async
    def _route_topic(self, content: str, fallback_id=None):
        from .services.routing import classify_topic
        fallback = self._topic_obj_sync(fallback_id)
        return classify_topic(self.session, content, fallback=fallback)

    @database_sync_to_async
    def _client_topic(self, topic_id):
        return self._topic_obj_sync(topic_id)

    def _topic_obj_sync(self, topic_id):
        """Resolve a topic id to a Topic (or None) for DB write."""
        if not topic_id:
            return None
        try:
            tid = int(topic_id)
        except (TypeError, ValueError):
            return None
        from apps.syllabus.models import Topic
        return Topic.objects.filter(pk=tid).first()

    def _topic_obj(self):
        """Resolve the stashed current_topic_id to a Topic (or None) for DB write."""
        if not self.current_topic_id:
            return None
        from apps.syllabus.models import Topic
        return Topic.objects.filter(pk=self.current_topic_id).first()

    @database_sync_to_async
    def _save_message(self, role: str, content: str, meta=None, topic=None):
        from .models import Message

        Message.objects.create(session=self.session, role=role, content=content, meta=meta, topic=topic)
        # Auto-title the session from the first student message.
        default_title = f"{self.session.syllabus.name}" + (
            f" - {self.session.subject.name}" if self.session.subject else ""
        )
        if role == "user" and (self.session.title or "") == default_title:
            self.session.title = content[:60]
            self.session.save(update_fields=["title"])

    async def send_json(self, data: dict):
        await self.send(text_data=json.dumps(data))
