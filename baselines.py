"""
baselines.py -- simple, honest baselines to compare SyntheticTextProbe against
on REAL data. A method is only interesting if it beats these.

PerplexityBaseline: a word-level bigram language model with add-k smoothing,
fit on a reference corpus. The classic "AI text is more predictable" idea uses a
fixed pretrained LM under which AI text has LOWER perplexity. Here we fit our own
small LM and let evaluate.py calibrate the sign on a training split so that a
higher score means "more synthetic."

Standard library only.
"""

import math
import re
from collections import defaultdict, Counter

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _toks(text):
    return _WORD_RE.findall((text or "").lower())


class PerplexityBaseline:
    def __init__(self, k=0.5):
        self.k = k
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.vocab = set()
        self._fitted = False

    def fit(self, texts):
        for t in texts:
            toks = ["<s>"] + _toks(t) + ["</s>"]
            for w in toks:
                self.unigram[w] += 1
                self.vocab.add(w)
            for a, b in zip(toks, toks[1:]):
                self.bigram[a][b] += 1
        self._fitted = True
        return self

    def _neg_logprob(self, a, b):
        V = max(1, len(self.vocab))
        num = self.bigram[a][b] + self.k
        den = self.unigram[a] + self.k * V
        return -math.log2(num / den)

    def cross_entropy(self, text):
        """Mean per-token cross-entropy in bits (a perplexity proxy)."""
        toks = ["<s>"] + _toks(text) + ["</s>"]
        if len(toks) < 2:
            return 0.0
        total = sum(self._neg_logprob(a, b) for a, b in zip(toks, toks[1:]))
        return total / (len(toks) - 1)

    def score(self, text):
        # Raw cross-entropy; evaluate.py flips the sign if needed (decided on train).
        return self.cross_entropy(text)


class MeanWordLengthBaseline:
    """Trivial baseline: average word length. Sanity floor for AUC."""

    def fit(self, texts):
        return self

    def score(self, text):
        toks = _toks(text)
        if not toks:
            return 0.0
        return sum(len(w) for w in toks) / len(toks)
