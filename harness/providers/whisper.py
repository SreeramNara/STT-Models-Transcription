import time
from faster_whisper import WhisperModel

from .provider import Provider, Transcription


class WhisperProvider(Provider):

    def __init__(self, model_size: str):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
        self._name = f"whisper-{model_size}"

    @property
    def name(self):
        return self._name

    def transcribe(self, audio_path: str, audio_seconds: float) -> Transcription:

        start = time.perf_counter()

        segments, info = self.model.transcribe(audio_path)

        text = " ".join(segment.text.strip() for segment in segments)

        elapsed = time.perf_counter() - start

        return Transcription(
            text=text,
            duration=elapsed,
            audio_seconds=audio_seconds
        )