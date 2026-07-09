import os
import torch
import numpy as np
import soundfile as sf
from pyannote.audio import Pipeline

TARGET_SR = 16000


class Diarizer:

    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")

        if hf_token is None:
            raise ValueError("HF_TOKEN not set")

        print("Loading diarization pipeline...")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )

        print("Diarization pipeline loaded.")

    def diarize(self, audio_path: str, num_speakers: int = None, min_speakers: int = 1, max_speakers: int = 4) -> list[dict]:
        """
        Returns a list of dicts: [{"start": float, "end": float, "speaker": str}, ...]

        num_speakers: set to exact number if known (overrides min/max)
        min_speakers: minimum number of speakers to detect (default 1)
        max_speakers: maximum number of speakers to detect (default 4)

        Passing min_speakers=2 helps pyannote find a second speaker on
        short clips where automatic clustering fails.
        """
        samples, sr = sf.read(audio_path, dtype="float32", always_2d=False)

        # stereo → mono
        if samples.ndim == 2:
            samples = samples.mean(axis=1)

        # resample to 16kHz if needed
        if sr != TARGET_SR:
            ratio      = TARGET_SR / sr
            new_length = int(len(samples) * ratio)
            indices    = np.linspace(0, len(samples) - 1, new_length)
            samples    = np.interp(indices, np.arange(len(samples)), samples)

        waveform   = torch.from_numpy(samples).unsqueeze(0)
        audio_dict = {"waveform": waveform, "sample_rate": TARGET_SR}

        # Pass speaker count hints to pyannote.
        # num_speakers pins it to an exact count.
        # min/max_speakers gives it a range to work within.
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["min_speakers"] = min_speakers
            kwargs["max_speakers"] = max_speakers

        output = self.pipeline(audio_dict, **kwargs)

        return output.serialize()["exclusive_diarization"]