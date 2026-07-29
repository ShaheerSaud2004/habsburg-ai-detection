"""
detector.py -- Habsburg AI Detection: a heuristic synthetic-text detector.

WHAT THIS IS
------------
A RESEARCH PROTOTYPE. It scores text 0..100 for how "synthetic / collapsed"
it looks, using six interpretable statistical signals. It also estimates the
synthetic-contamination ratio of a whole dataset.

HONEST CAVEAT
-------------
These six signals are heuristic PROXIES. They are demonstrated here under
*controlled* synthetic-vs-human conditions (see data_generator.py). They are
NOT a validated production classifier, and they CANNOT audit real frontier
models out of the box. Treat the numbers as a measurement *framework*, not a
ground-truth oracle. See NEXT_STEPS.md for what real validation requires.

Standard library only -- runs with zero pip install.
"""

import re
import hashlib
import statistics
from collections import Counter

# score >= SYNTHETIC_THRESHOLD  ->  predicted synthetic
SYNTHETIC_THRESHOLD = 50.0

# Canned filler phrases that templated / "collapsed" LLM text over-uses.
FILLER_PHRASES = [
    "it is important to note",
    "it is worth noting",
    "in conclusion",
    "in summary",
    "as we can see",
    "a wide range of",
    "plays a crucial role",
    "plays a vital role",
    "it is essential to",
    "delve into",
    "navigate the",
    "in today's world",
    "in the realm of",
    "a testament to",
    "rich tapestry",
    "ever-evolving",
    "ever-changing",
    "shed light on",
    "when it comes to",
    "first and foremost",
    "needless to say",
    "in essence",
    "furthermore",
    "moreover",
    "additionally",
    "overall",
]

_WORD_RE = re.compile(r"[a-zA-Z']+")
_SENT_RE = re.compile(r"[.!?]+")


def _tokens(text):
    return _WORD_RE.findall(text.lower())


def _sentences(text):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _shingles(text, k=5):
    """Hashed word-level k-shingles as a set, for Jaccard similarity."""
    toks = _tokens(text)
    grams = [tuple(toks)] if len(toks) < k else _ngrams(toks, k)
    return {hashlib.md5((" ".join(g)).encode("utf-8")).hexdigest() for g in grams}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


# Signal weights (sum to 1.0). Three signals are "human-high" and are inverted
# below, so a higher contribution always means "more synthetic".
WEIGHTS = {
    "repetition": 0.18,
    "filler_density": 0.20,
    "lexical_diversity": 0.16,
    "ngram_diversity": 0.16,
    "burstiness": 0.15,
    "duplicate_similarity": 0.15,
}
INVERTED = {"lexical_diversity", "ngram_diversity", "burstiness"}


def combine_signals(signals, drop=()):
    """Combine raw 0..1 signals into a 0..100 synthetic score.

    `drop` is an iterable of signal names to exclude (used by the ablation in
    evaluate.py). Remaining weights are renormalized so scores stay comparable.
    """
    drop = set(drop)
    acc = 0.0
    total_w = 0.0
    for key, w in WEIGHTS.items():
        if key in drop:
            continue
        contrib = (1.0 - signals[key]) if key in INVERTED else signals[key]
        acc += w * contrib
        total_w += w
    if total_w == 0.0:
        return 0.0
    return round(100.0 * _clamp01(acc / total_w), 2)


class SyntheticTextProbe:
    def __init__(self, reference_synthetic=None):
        # Optional corpus of KNOWN synthetic / model outputs. Used to catch
        # recycling: text that closely matches these is likely fed-back output.
        self.reference_synthetic = list(reference_synthetic) if reference_synthetic else []
        self._ref_shingles = [_shingles(t) for t in self.reference_synthetic]

    # ----- individual signals, each in 0..1 -----

    def _repetition(self, toks):
        tris = _ngrams(toks, 3)
        if not tris:
            return 0.0
        counts = Counter(tris)
        repeated = sum(c for c in counts.values() if c > 1)
        return _clamp01(repeated / len(tris))

    def _filler_density(self, text):
        low = text.lower()
        toks = _tokens(text)
        if not toks:
            return 0.0
        hits = sum(low.count(ph) for ph in FILLER_PHRASES)
        return _clamp01(hits / max(1.0, len(toks) / 50.0))

    def _lexical_diversity(self, toks):
        if not toks:
            return 0.0
        return _clamp01(len(set(toks)) / len(toks))  # type-token ratio

    def _ngram_diversity(self, toks):
        tris = _ngrams(toks, 3)
        if not tris:
            return 0.0
        return _clamp01(len(set(tris)) / len(tris))

    def _burstiness(self, text):
        lengths = [len(_tokens(s)) for s in _sentences(text)]
        lengths = [l for l in lengths if l > 0]
        if len(lengths) < 2:
            return 0.0
        mean = statistics.mean(lengths)
        if mean == 0:
            return 0.0
        cv = statistics.pstdev(lengths) / mean  # coefficient of variation
        return _clamp01(cv)  # human prose ~0.5-1.0; uniform AI text ~0.1

    def _duplicate_similarity(self, text):
        if not self._ref_shingles:
            return 0.0
        sh = _shingles(text)
        return _clamp01(max((_jaccard(sh, ref) for ref in self._ref_shingles), default=0.0))

    # ----- public API -----

    def raw_signals(self, text):
        """Return the six raw signals (each 0..1), unrounded."""
        text = text or ""
        toks = _tokens(text)
        return {
            "repetition": self._repetition(toks),
            "filler_density": self._filler_density(text),
            "lexical_diversity": self._lexical_diversity(toks),
            "ngram_diversity": self._ngram_diversity(toks),
            "burstiness": self._burstiness(text),
            "duplicate_similarity": self._duplicate_similarity(text),
        }

    def score_text(self, text):
        sig = self.raw_signals(text)
        likelihood = combine_signals(sig)

        reasons = []
        if sig["duplicate_similarity"] > 0.4:
            reasons.append("high overlap with known synthetic outputs (recycling)")
        if sig["filler_density"] > 0.4:
            reasons.append("dense canned filler phrasing")
        if sig["lexical_diversity"] < 0.45:
            reasons.append("low lexical diversity")
        if sig["burstiness"] < 0.35:
            reasons.append("uniform sentence lengths")
        if sig["repetition"] > 0.2:
            reasons.append("repeated trigrams")
        if not reasons:
            reasons.append("varied vocabulary and sentence structure")

        return {
            "synthetic_likelihood": likelihood,
            "is_synthetic": likelihood >= SYNTHETIC_THRESHOLD,
            "signals": {k: round(v, 3) for k, v in sig.items()},
            "explanation": "; ".join(reasons),
        }

    def detect_dataset(self, texts):
        texts = list(texts)
        scores, predicted = [], 0
        for t in texts:
            r = self.score_text(t)
            scores.append(r["synthetic_likelihood"])
            if r["is_synthetic"]:
                predicted += 1
        total = len(texts)
        return {
            "total": total,
            "predicted_synthetic": predicted,
            "estimated_contamination": round(predicted / total, 4) if total else 0.0,
            "scores": scores,
        }


# Backward-compatible alias (the detector was previously named HabsburgDetector).
HabsburgDetector = SyntheticTextProbe
