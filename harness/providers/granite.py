import time
import torch
import soundfile as sf
import torchaudio.functional as F

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .provider import Provider, Transcription
from harness.diarization import Diarizer
from harness.asr_segments import ASRSegmenter


MODEL_ID = "ibm-granite/granite-4.0-1b-speech"
PROMPT = "<|audio|>can you transcribe the speech into a written format?"


class GraniteProvider(Provider):

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Granite speech model on {self._device} (first run downloads ~4GB)...")

        self.diarizer = Diarizer()
        self.asr_segmenter = ASRSegmenter()

        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._tokenizer = self._processor.tokenizer

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )

        chat = [{"role": "user", "content": PROMPT}]
        self._prompt_str = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        print("Granite model loaded.")

    @property
    def name(self):
        return "granite-4.0-1b-speech"

    def _format_segments(self, segments):
        return "\n".join(
            f"{s['speaker']}: {s['text']}"
            for s in segments
        )

    def transcribe(self, audio_path: str, audio_seconds: float):

        audio, sr = sf.read(audio_path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = torch.tensor(audio, dtype=torch.float32)

        if sr != 16000:
            audio = F.resample(audio.unsqueeze(0), sr, 16000).squeeze(0)

        start = time.perf_counter()

        # -------------------------
        # 1. GRANITE FULL TEXT
        # Used for WER — clean transcript with no speaker labels.
        # -------------------------
        model_inputs = self._processor(
            self._prompt_str,
            audio,
            device=self._device,
            return_tensors="pt",
        ).to(self._device)

        model_outputs = self._model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)

        full_text = self._tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )[0].strip()

        # -------------------------
        # 2. REAL TIMESTAMP SEGMENTS
        # -------------------------
        asr_segments = self.asr_segmenter.transcribe_with_timestamps(audio_path)

        # -------------------------
        # 3. DIARIZATION (time-based)
        # -------------------------
        diar_segments = self.diarizer.diarize(audio_path)

        def get_speaker(t):
            for d in diar_segments:
                if d["start"] <= t <= d["end"]:
                    return d["speaker"]
            return "UNKNOWN"

        # -------------------------
        # 4. ALIGN ASR ↔ SPEAKERS
        # -------------------------
        speaker_lines = []

        for seg in asr_segments:
            mid = (seg["start"] + seg["end"]) / 2
            speaker = get_speaker(mid)
            speaker_lines.append({
                "speaker": speaker,
                "text": seg["text"]
            })

        # -------------------------
        # 5. FORMAT DIARIZED OUTPUT
        # Stored separately — used for display only, not WER.
        # -------------------------
        diarized_text = "\n".join(
            f"{s['speaker']}: {s['text']}"
            for s in speaker_lines
        )

        elapsed = time.perf_counter() - start

        # text = full_text (clean, no speaker labels) → used for WER
        # diarized_text attached dynamically → used for display in inspect_transcripts
        result = Transcription(
            text=full_text,
            duration=elapsed,
            audio_seconds=audio_seconds,
        )
        result.diarized_text = diarized_text   # display-only, not in dataclass schema

        return result