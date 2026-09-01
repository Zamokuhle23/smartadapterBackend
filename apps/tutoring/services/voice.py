"""
Voice services: speech-to-text (faster-whisper) and text-to-speech (Piper).

Both are free / open-source and fast on CPU. Imports are lazy so the Django app
still boots even if the (heavy) audio dependencies are not yet installed -
callers get a clear error instead of an import crash.

Piper voices are downloaded on first use (e.g. "en_US-lessac-medium"). Override
via PIPER_VOICE / PIPER_VOICE_DIR in settings if you want a different or bundled
voice file.
"""
import base64
import io
import wave

from django.conf import settings


VOICE_SYSTEM_TEMPLATE = (
    "You are FundzaAI, a warm, encouraging tutor speaking aloud to an Eswatini "
    "student. You are having a spoken conversation.\n\n"
    "SPEAKING RULES - follow these strictly:\n"
    "- Speak in short, natural sentences. One idea per sentence.\n"
    "- NO markdown, no bullet points, no numbered lists, no headings, no asterisks.\n"
    "- Write the way you would say it out loud, including contractions "
    "(\"it's\", \"you'll\").\n"
    "- Keep explanations concrete and use an everyday example when it helps.\n"
    "- Pause between ideas so it sounds spoken, not written.\n"
    "- End with ONE short check-understanding question.\n"
    "- Stay within the student's syllabus scope and tier; keep it warm.\n"
)


def _whisper_model():
    from faster_whisper import WhisperModel

    name = getattr(settings, "WHISPER_MODEL", "base")
    # 8 threads, int8 on CPU = fast enough for interactive dictation.
    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=8)


def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe 16kHz mono, signed 16-bit PCM audio to text."""
    import numpy as np

    model = _whisper_model()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _info = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=True,  # built-in silero VAD ignores silence between utterances
    )
    return " ".join(seg.text for seg in segments).strip()


def _load_voice():
    from piper import PiperVoice

    voice_name = getattr(settings, "PIPER_VOICE", "") or ""
    if voice_name:
        return PiperVoice.load(voice_name)
    # Default: prefer a bundled voice in backend/voices if present, else download.
    import os as _os

    candidates = [
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "voices",
                      "en_US-lessac-medium.onnx"),
    ]
    for path in candidates:
        if _os.path.exists(path):
            return PiperVoice.load(path)
    return PiperVoice.load("en_US-lessac-medium", download_dir=getattr(settings, "PIPER_VOICE_DIR", None))


def synthesize(text: str) -> bytes:
    """Synthesize text to a 16-bit PCM WAV (mono). Returns the WAV bytes."""
    if not text.strip():
        return b""
    voice = _load_voice()
    out = io.BytesIO()
    synthesize_wav = getattr(voice, "synthesize_wav", None)
    if synthesize_wav is not None:
        with wave.open(out, "wb") as wav:
            voice.synthesize_wav(text, wav)
    else:
        # Fallback: piper.synthesize writes raw PCM; wrap it in a WAV container.
        raw = io.BytesIO()
        voice.synthesize(text, raw)
        raw.seek(0)
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(raw.read())
    out.seek(0)
    return out.getvalue()


def synthesize_base64(text: str) -> str:
    return base64.b64encode(synthesize(text)).decode("ascii")