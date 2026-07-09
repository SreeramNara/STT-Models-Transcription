import os
import time
import requests

from .provider import Provider, Transcription

UPLOAD_URL     = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
POLL_INTERVAL  = 1.5   # seconds between status polls
MIN_DURATION   = 0.2   # AssemblyAI rejects clips shorter than ~200ms


class AssemblyAIProvider(Provider):

    def __init__(self, model="best"):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()

        if not self.api_key:
            raise ValueError("Missing ASSEMBLYAI_API_KEY")

        # AssemblyAI speech model: "best" (most accurate) or "nano" (faster/cheaper)
        self.model = model

        self.headers = {
            "Authorization": self.api_key,
        }

    @property
    def name(self):
        return f"assemblyai-{self.model}"

    def _upload(self, audio_path: str) -> str:
        """Upload audio file to AssemblyAI and return the upload URL."""
        with open(audio_path, "rb") as f:
            response = requests.post(
                UPLOAD_URL,
                headers=self.headers,
                data=f,
            )
        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.text}")
        return response.json()["upload_url"]

    def _submit(self, upload_url: str) -> str:
        """Submit a transcription job and return the transcript ID."""
        payload = {
            "audio_url":    upload_url,
            "speech_model": self.model,
        }
        response = requests.post(
            TRANSCRIPT_URL,
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Transcription submit failed: {response.text}")
        return response.json()["id"]

    def _poll(self, transcript_id: str) -> str:
        """Poll until transcription is complete and return the transcript text."""
        url = f"{TRANSCRIPT_URL}/{transcript_id}"
        while True:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                raise RuntimeError(f"Poll failed: {response.text}")
            result = response.json()
            status = result["status"]
            if status == "completed":
                return result["text"] or ""
            elif status == "error":
                error_msg = result.get("error", "")
                # AssemblyAI rejects clips that are too short — treat as empty
                # transcript rather than a fatal error so the benchmark continues.
                if "too short" in error_msg.lower():
                    return ""
                raise RuntimeError(f"Transcription error: {error_msg}")
            time.sleep(POLL_INTERVAL)

    def transcribe(self, audio_path: str, audio_seconds: float) -> Transcription:
        start = time.perf_counter()

        # Skip clips below AssemblyAI's minimum duration entirely —
        # saves an upload + API round trip for clips we know will be rejected.
        if audio_seconds < MIN_DURATION:
            return Transcription(
                text="",
                duration=time.perf_counter() - start,
                audio_seconds=audio_seconds,
            )

        upload_url    = self._upload(audio_path)
        transcript_id = self._submit(upload_url)
        text          = self._poll(transcript_id)

        elapsed = time.perf_counter() - start

        return Transcription(
            text=text,
            duration=elapsed,
            audio_seconds=audio_seconds,
        )