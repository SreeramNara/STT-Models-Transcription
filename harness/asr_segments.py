"""Timestamped transcript segments, used to align text with diarization speakers.

Granite returns a single block of text with no timestamps, so the Granite provider
uses this segmenter to get (start, end, text) segments it can match against
pyannote's speaker turns.
"""

from faster_whisper import WhisperModel


class ASRSegmenter:

    def __init__(self, model_size: str = "small"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_with_timestamps(self, audio_path: str) -> list[dict]:
        """Return [{"start": float, "end": float, "text": str}, ...] in seconds."""
        segments, _info = self.model.transcribe(audio_path)
        return [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
            if seg.text.strip()
        ]
