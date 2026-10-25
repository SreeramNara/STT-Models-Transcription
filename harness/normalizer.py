"""Text normalization applied to both reference and model output before scoring.

Different STT providers format text differently (casing, punctuation, filler words).
Normalizing both sides the same way means WER measures recognition errors,
not formatting differences.
"""

import re

# Hesitation sounds. AMI references include many of these, and providers handle
# them inconsistently (some drop them, some transcribe them), so they are removed
# from both sides.
FILLERS = {"um", "uh", "uhm", "erm", "er", "ah", "eh", "hmm", "hm", "mm", "mm-hmm", "uh-huh"}

_PUNCTUATION = re.compile(r"[^\w\s'-]")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and filler words, and collapse whitespace."""
    if not text:
        return ""

    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)

    words = []
    for word in text.split():
        word = word.strip("'-")
        if word and word not in FILLERS:
            words.append(word)

    return _WHITESPACE.sub(" ", " ".join(words)).strip()
