# STT Model Benchmarking

**A benchmarking harness that compares speech-to-text models on real multi-speaker meeting audio, measuring accuracy and speed.**

Most STT benchmarks use clean, single-speaker audio. This harness tests models on conversational meeting recordings from the [AMI Meeting Corpus](https://huggingface.co/datasets/edinburghcstr/ami), with overlapping speakers, crosstalk, and far-field microphones. Every model sits behind the same interface, so they're all scored on identical clips with identical text normalization.

## Models compared

| Model | Type | Runs on |
|---|---|---|
| Whisper Small | Open source (via `faster-whisper`) | Local CPU, int8 |
| Whisper Large-v3 | Open source (via `faster-whisper`) | Local CPU, int8 |
| Deepgram Nova-2 | Hosted API | Deepgram cloud |
| AssemblyAI Best | Hosted API | AssemblyAI cloud |
| IBM Granite 4.0 1B Speech | Open source (via `transformers`) | Local GPU or CPU |

## Metrics

For every clip and every model, the harness records:

- **WER (word error rate):** the share of words the model got wrong compared with the reference transcript. Lower is better.
- **CER (character error rate):** the same idea at the character level.
- **RTF (real-time factor):** processing time divided by audio length. An RTF of 0.5 means a 10-minute recording takes 5 minutes to transcribe.

WER is reported at the corpus level (total word errors / total reference words) rather than as an average of per-clip rates, so short clips don't skew the result. Before scoring, both the reference and the model output are lowercased, stripped of punctuation, and stripped of filler words like "um" and "uh". That way models aren't penalized for formatting choices.

## Speaker diarization

The Granite provider also labels **who spoke when**. It runs [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) to find speaker turns, then assigns each timestamped transcript segment to the speaker active at its midpoint. Example output format:

```
SPEAKER_00: so the main thing we need to decide today is the remote's case material
SPEAKER_01: right, I looked into rubber versus plastic
```

The diarized text is used for display only. WER is always computed on the clean transcript, so speaker labels never affect the score.

## How it works

```
manifest.jsonl ──► for each clip ──► each provider ──► normalize text ──► WER / CER / RTF
                                                                                 │
                                                       results/per_clip.csv ◄────┘
                                                       summary per provider
```

Each provider implements one method:

```python
class Provider(ABC):
    def transcribe(self, audio_path: str, audio_seconds: float) -> Transcription: ...
```

To add a new model, write a class that implements `transcribe()` and register it in `build_provider()` in `harness/run.py`.

## Setup

**Requirements:** Python 3.10+. A GPU is optional, but it makes Granite much faster.

```bash
git clone https://github.com/SreeramNara/STT-Models-Transcription.git
cd STT-Models-Transcription

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API keys

Copy `.env.example` to `.env` and fill in the keys you need, then load them into your shell (for example `set -a; source .env; set +a` on macOS/Linux).

| Variable | Needed for |
|---|---|
| `DEEPGRAM_API_KEY` | Deepgram Nova-2 |
| `ASSEMBLYAI_API_KEY` | AssemblyAI |
| `HF_TOKEN` | Speaker diarization (accept the terms on the [pyannote model page](https://huggingface.co/pyannote/speaker-diarization-3.1) first) |

Whisper doesn't need any keys. To skip a hosted provider, remove it from your config.

## Running a benchmark

**1. Download AMI clips and build a manifest:**

```bash
python -m scripts.build_ami_manifest --config sdm --split test --limit 50
```

This saves the clips as WAV files under `data/` and writes `data/manifest.jsonl`, a JSON Lines file with one clip per line:

```json
{"audio": "data/ami_sdm_test/AMI_ES2004a_SDM_00012345.wav", "text": "so the main thing we need to decide today", "duration": 4.2}
```

You can also benchmark your own audio by writing a manifest in the same format.

**2. Choose providers** in a config file (see [`configs/example.yaml`](configs/example.yaml)):

```yaml
providers:
  - type: whisper
    model: small
  - type: deepgram
    model: nova-2
```

**3. Run it:**

```bash
python -m harness.run --manifest data/manifest.jsonl --config configs/example.yaml
```

Results are written to `results/`:

| File | Contents |
|---|---|
| `per_clip.csv` | WER, CER and RTF for every clip and model |
| `summary.csv` | One row per model |
| `report.md` | The summary as a Markdown table, also printed to the terminal |

## Results

<!-- TODO: paste the table from results/report.md after a run -->

## Project structure

```
harness/
├── run.py                 # Entry point: loads manifest + config, runs every provider, writes results
├── normalizer.py          # Text normalization applied before scoring
├── metrics.py             # WER, CER and real-time factor
├── report.py              # Per-model summary and Markdown report
├── diarization.py         # pyannote speaker diarization
├── asr_segments.py        # Timestamped segments for aligning text with speakers
└── providers/
    ├── provider.py        # Shared Provider interface and Transcription result
    ├── whisper.py         # faster-whisper (Small, Large-v3)
    ├── deepgram.py        # Deepgram REST API
    ├── assemblyai.py      # AssemblyAI upload, submit and poll
    └── granite.py         # IBM Granite speech model + diarization alignment
scripts/
└── build_ami_manifest.py  # Downloads AMI clips and writes the manifest
configs/
└── example.yaml           # Example provider configuration
```

## Implementation notes

- **AMI loading on Windows:** audio is loaded through Hugging Face `datasets` with `Audio(decode=False)` and decoded manually with `soundfile`. This avoids the `torchcodec` dependency, which doesn't install cleanly on Windows.
- **SDM instead of IHM:** the benchmark uses AMI's single distant microphone recordings rather than individual headset mics. That captures all speakers on one channel, which is closer to a real meeting recording and much harder to transcribe.
- **Diarization hints:** pyannote often finds only one speaker on short clips. The diarizer accepts `min_speakers` and `max_speakers` to keep clustering in a sensible range.
