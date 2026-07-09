from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class Transcription:
    text: str
    duration: float        # seconds spent transcribing
    audio_seconds: float   # length of audio file

    # 🆕 optional diarization output
    speaker_segments: Optional[List[Dict]] = None


class Provider(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, audio_seconds: float) -> Transcription:
        pass