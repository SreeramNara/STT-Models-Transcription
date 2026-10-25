"""Download AMI meeting clips from Hugging Face and write a benchmark manifest.

Each line of the manifest is: {"audio": path, "text": reference, "duration": seconds}

Audio is loaded with Audio(decode=False) and decoded with soundfile, which avoids
the torchcodec dependency that doesn't install cleanly on Windows.

Example:
    python -m scripts.build_ami_manifest --config sdm --split test --limit 50
"""

import argparse
import io
import json
import os

import soundfile as sf
from datasets import Audio, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="sdm", choices=["sdm", "ihm"],
                        help="sdm = single distant mic (all speakers, harder); ihm = individual headset mics")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50, help="number of clips to keep")
    parser.add_argument("--min-seconds", type=float, default=2.0, help="skip clips shorter than this")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    audio_dir = os.path.join(args.out_dir, f"ami_{args.config}_{args.split}")
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")

    dataset = load_dataset("edinburghcstr/ami", args.config, split=args.split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    written = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for example in dataset:
            text = (example.get("text") or "").strip()
            if not text:
                continue

            audio = example["audio"]
            raw = audio.get("bytes")
            if raw is None:
                with open(audio["path"], "rb") as f:
                    raw = f.read()

            samples, sr = sf.read(io.BytesIO(raw), dtype="float32")
            duration = len(samples) / sr
            if duration < args.min_seconds:
                continue

            wav_path = os.path.join(audio_dir, f"{example['audio_id']}.wav")
            sf.write(wav_path, samples, sr)

            manifest.write(json.dumps({"audio": wav_path, "text": text, "duration": round(duration, 3)}) + "\n")
            written += 1
            if written >= args.limit:
                break

    print(f"Wrote {written} clips to {manifest_path}")


if __name__ == "__main__":
    main()
