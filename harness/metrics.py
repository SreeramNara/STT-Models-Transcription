"""Accuracy and speed metrics for a single clip."""

import jiwer


def compute_metrics(reference: str, hypothesis: str) -> dict:
    """Word and character error rates for one normalized reference/hypothesis pair.

    Also returns the raw error and reference word counts, so the report can
    compute corpus-level WER (total errors / total words) instead of averaging
    per-clip rates, which over-weights short clips.
    """
    ref_words = len(reference.split())
    hyp_words = len(hypothesis.split())

    # jiwer can't score an empty reference: every hypothesis word is an insertion.
    if ref_words == 0:
        return {
            "wer": 0.0 if hyp_words == 0 else 1.0,
            "cer": 0.0 if not hypothesis else 1.0,
            "word_errors": hyp_words,
            "ref_words": 0,
        }

    out = jiwer.process_words(reference, hypothesis)
    word_errors = out.substitutions + out.deletions + out.insertions

    return {
        "wer": word_errors / ref_words,
        "cer": jiwer.cer(reference, hypothesis) if hypothesis else 1.0,
        "word_errors": word_errors,
        "ref_words": ref_words,
    }


def compute_rtf(processing_seconds: float, audio_seconds: float) -> float:
    """Real-time factor: processing time / audio length. Below 1.0 is faster than real time."""
    if audio_seconds <= 0:
        return 0.0
    return processing_seconds / audio_seconds
