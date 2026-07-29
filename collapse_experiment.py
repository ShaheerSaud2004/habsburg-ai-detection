"""
collapse_experiment.py -- recursive MODEL COLLAPSE with a small generative model.

This is the experiment that ties the detector back to model collapse itself.

A small word-level trigram language model (with bigram/unigram backoff) is trained
on a human seed corpus. It samples a new corpus; the NEXT generation's model is
trained on THAT corpus; and so on -- Shumailov-style recursion. Because each
generation is a finite sample, rare n-grams (the tail) get dropped and can never
return, so the model's distribution narrows generation after generation. A
trigram LM also literally cannot emit a word it never saw, so the vocabulary is
monotonically non-increasing -- collapse is guaranteed and measurable.

We instrument every generation with the Habsburg detector and watch its estimated
contamination climb as the corpus collapses.

Standard library only.

    python3 collapse_experiment.py
"""

import json
import os
import random
import re
from collections import defaultdict, Counter

from detector import SyntheticTextProbe
import fetch_multimodel as fm   # reused only for its human seed corpus

SEED = 1234
GENERATIONS = 6
N_DOCS = 150
DOC_LEN = 80
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "collapse_results.json")

_TOK = re.compile(r"[a-zA-Z']+|[.!?]")


def toks(t):
    return _TOK.findall((t or "").lower())


class TrigramModel:
    """Word-level trigram model with bigram/unigram backoff; samples documents."""

    def __init__(self):
        self.tri = defaultdict(Counter)
        self.bi = defaultdict(Counter)
        self.uni = Counter()
        self.starts = []

    def fit(self, docs):
        for d in docs:
            ws = toks(d)
            if len(ws) >= 2:
                self.starts.append((ws[0], ws[1]))
            for w in ws:
                self.uni[w] += 1
            for a, b in zip(ws, ws[1:]):
                self.bi[a][b] += 1
            for a, b, c in zip(ws, ws[1:], ws[2:]):
                self.tri[(a, b)][c] += 1
        return self

    @staticmethod
    def _sample(counter, rng):
        items = list(counter.items())
        total = sum(c for _, c in items)
        r = rng.random() * total
        acc = 0
        for w, c in items:
            acc += c
            if r <= acc:
                return w
        return items[-1][0]

    def gen_doc(self, rng, length):
        if not self.starts:
            return ""
        w1, w2 = rng.choice(self.starts)
        out = [w1, w2]
        for _ in range(length - 2):
            if (w1, w2) in self.tri:
                nxt = self._sample(self.tri[(w1, w2)], rng)
            elif w2 in self.bi:
                nxt = self._sample(self.bi[w2], rng)
            else:
                nxt = self._sample(self.uni, rng)
            out.append(nxt)
            w1, w2 = w2, nxt
        s = " ".join(out)
        return re.sub(r"\s+([.!?])", r"\1", s)   # tidy spacing before punctuation

    def sample_corpus(self, rng, n, length):
        return [self.gen_doc(rng, length) for _ in range(n)]


def diversity(docs):
    words = [w for d in docs for w in toks(d) if w.isalpha()]
    vocab = len(set(words))
    tris = list(zip(words, words[1:], words[2:]))
    ttr = (vocab / len(words)) if words else 0.0
    dtr = (len(set(tris)) / len(tris)) if tris else 0.0
    return vocab, round(ttr, 3), round(dtr, 3)


def main():
    rng = random.Random(SEED)
    detector = SyntheticTextProbe()

    human = fm.load("human")
    rng.shuffle(human)
    corpus = human[:N_DOCS]

    rows = []
    print("Recursive collapse: a trigram LM retrained on its own output each generation")
    print("=" * 72)
    for g in range(GENERATIONS + 1):
        vocab, ttr, dtr = diversity(corpus)
        scores = [detector.score_text(d)["synthetic_likelihood"] for d in corpus]
        mean_score = sum(scores) / len(scores)
        est = detector.detect_dataset(corpus)["estimated_contamination"]
        rows.append({
            "generation": g, "vocab": vocab, "ttr": ttr, "trigram_diversity": dtr,
            "mean_synth_score": round(mean_score, 1), "est_contamination": est,
        })
        tag = "(human seed)" if g == 0 else ""
        print("  gen %d %-12s vocab=%5d  ttr=%.3f  tri-div=%.3f  detector=%4.1f  est_contam=%3.0f%%"
              % (g, tag, vocab, ttr, dtr, mean_score, 100 * est))
        if g == GENERATIONS:
            break
        corpus = TrigramModel().fit(corpus).sample_corpus(rng, N_DOCS, DOC_LEN)

    g0, gN = rows[0], rows[-1]
    summary = {
        "seed": "RAID human",
        "n_docs": N_DOCS, "doc_len": DOC_LEN, "generations": rows,
        "vocab_drop_pct": round(100 * (1 - gN["vocab"] / g0["vocab"]), 1),
        "detector_rise": round(gN["mean_synth_score"] - g0["mean_synth_score"], 1),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Collapse over %d generations: vocabulary -%.0f%%, detector score +%.1f, "
          "estimated contamination %.0f%% -> %.0f%%"
          % (GENERATIONS, summary["vocab_drop_pct"], summary["detector_rise"],
             100 * g0["est_contamination"], 100 * gN["est_contamination"]))
    print("Wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
