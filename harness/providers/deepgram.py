import os
import time
import requests

from .provider import Provider, Transcription


class DeepgramProvider(Provider):

    def __init__(self, model="nova-2", cap_minutes=30):

        self.api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()

        if not self.api_key:
            raise ValueError("Missing DEEPGRAM_API_KEY")

        self.model = model
        self.cap_seconds = cap_minutes * 60

        self.url = f"https://api.deepgram.com/v1/listen?model={model}&smart_format=true"

        self.used_seconds = 0

    @property
    def name(self):
        return f"deepgram-{self.model}"

    def transcribe(self, audio_path: str, audio_seconds: float) -> Transcription:

        with open(audio_path, "rb") as f:
            audio = f.read()

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav"
        }

        start = time.perf_counter()

        response = requests.post(
            self.url,
            headers=headers,
            data=audio
        )

        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            raise RuntimeError(response.text)

        result = response.json()

        text = result["results"]["channels"][0]["alternatives"][0]["transcript"]

        return Transcription(
            text=text,
            duration=elapsed,
            audio_seconds=audio_seconds
        )