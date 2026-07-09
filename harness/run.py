import argparse
import json
import os
import yaml
import pandas as pd

from tqdm import tqdm

from harness.normalizer import normalize
from harness.metrics import compute_metrics, compute_rtf
from harness.report import write_summary, write_report

from harness.providers.whisper import WhisperProvider


def build_provider(config):

    if config["type"] == "whisper":
        return WhisperProvider(config["model"])

    if config["type"] == "deepgram":
        from harness.providers.deepgram import DeepgramProvider
        return DeepgramProvider(
            model=config["model"],
            cap_minutes=config.get("cap_minutes", 30)
        )

    if config["type"] == "assemblyai":
        from harness.providers.assemblyai import AssemblyAIProvider
        return AssemblyAIProvider(
            model=config.get("model", "best")
        )

    if config["type"] == "granite":
        from harness.providers.granite import GraniteProvider
        return GraniteProvider()

    raise ValueError(f"Unknown provider {config['type']}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    providers = [build_provider(p) for p in config["providers"]]

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f]

    rows = []

    for clip in tqdm(manifest):

        reference = normalize(clip["text"])
        audio_path = clip["audio"]
        duration = clip["duration"]

        for provider in providers:

            result = provider.transcribe(audio_path, duration)

            hypothesis = normalize(result.text)

            metrics = compute_metrics(reference, hypothesis)

            rows.append({
                "provider": provider.name,
                "audio": audio_path,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "rtf": compute_rtf(result.duration, duration),
                "cost": 0.0,
                "word_errors": metrics["word_errors"],
                "ref_words": metrics["ref_words"]
            })

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(rows)

    df.to_csv("results/per_clip.csv", index=False)

    summary = write_summary(df)

    write_report(summary)


if __name__ == "__main__":
    main()