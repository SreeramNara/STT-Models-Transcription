"""Aggregate per-clip results into a per-provider summary."""

import os

import pandas as pd

RESULTS_DIR = "results"


def write_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize results per provider and save them to results/summary.csv.

    WER is corpus-level (total word errors / total reference words), the standard
    way to report it. CER and RTF are averaged across clips.
    """
    grouped = df.groupby("provider", sort=False)

    summary = pd.DataFrame({
        "clips": grouped.size(),
        "wer": grouped["word_errors"].sum() / grouped["ref_words"].sum().clip(lower=1),
        "cer": grouped["cer"].mean(),
        "rtf": grouped["rtf"].mean(),
    }).reset_index().sort_values("wer")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    return summary


def write_report(summary: pd.DataFrame) -> None:
    """Print the summary table and save it as Markdown to results/report.md."""
    lines = [
        "| Model | Clips | WER | CER | RTF |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.provider} | {row.clips} | {row.wer:.1%} | {row.cer:.1%} | {row.rtf:.2f} |"
        )
    table = "\n".join(lines)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("# STT benchmark results\n\n")
        f.write(table + "\n\n")
        f.write("WER is corpus-level (total errors / total reference words). "
                "RTF below 1.0 means faster than real time.\n")

    print("\n" + table + "\n")
