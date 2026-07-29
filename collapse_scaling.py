"""
collapse_scaling.py -- does the collapse finding survive scale and a second dataset?

Reviewer concern: the headline collapse run uses only 150 documents. Here we vary
the per-generation corpus size over {150, 1500, 15000} and run on TWO seed corpora
(RAID human and HC3 human), holding generation-0 fixed (the full real pool) so that
size is the only thing that changes. We track the same instruments: distribution-
level metrics (vocabulary, trigram diversity) and the per-document detector.

Expected and reported honestly: larger corpora collapse more SLOWLY (a bigger
sample keeps the tail alive longer), but the qualitative finding is invariant --
distribution metrics fall while the per-document detector stays flat / sub-threshold.

Uses an optimized (cached, bisect) trigram sampler so 15k is tractable. Stdlib only.

    python3 collapse_scaling.py
"""

import bisect
import json
import os
import random
from collections import defaultdict, Counter

from detector import SyntheticTextProbe, SYNTHETIC_THRESHOLD
from collapse_experiment import toks, diversity
import fetch_multimodel as fm
import fetch_corpora

SEED = 1234
GENERATIONS = 5
DOC_LEN = 80
SIZES = [150, 1500, 15000]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "collapse_scaling_results.json")


class FastTrigram:
    """Trigram LM with bigram/unigram backoff; cached cumulative dists + bisect sampling."""

    def __init__(self):
        self.tri = defaultdict(Counter)
        self.bi = defaultdict(Counter)
        self.uni = Counter()
        self.starts = []
        self._cache = {}

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

    def _cum(self, key, counter):
        c = self._cache.get(key)
        if c is None:
            pop = list(counter.keys())
            cum, s = [], 0
            for w in pop:
                s += counter[w]; cum.append(s)
            c = (pop, cum, s)
            self._cache[key] = c
        return c

    def _draw(self, key, counter, rng):
        pop, cum, total = self._cum(key, counter)
        i = bisect.bisect_left(cum, rng.random() * total)
        return pop[min(i, len(pop) - 1)]

    def gen_doc(self, rng, length):
        if not self.starts:
            return ""
        w1, w2 = rng.choice(self.starts)
        out = [w1, w2]
        for _ in range(length - 2):
            if (w1, w2) in self.tri:
                nxt = self._draw(("t", w1, w2), self.tri[(w1, w2)], rng)
            elif w2 in self.bi:
                nxt = self._draw(("b", w2), self.bi[w2], rng)
            else:
                nxt = self._draw(("u",), self.uni, rng)
            out.append(nxt); w1, w2 = w2, nxt
        return " ".join(out)

    def sample_corpus(self, rng, n, length):
        return [self.gen_doc(rng, length) for _ in range(n)]


def measure(corpus, det):
    vocab, ttr, dtr = diversity(corpus)
    scores = [det.score_text(d)["synthetic_likelihood"] for d in corpus]
    flagged = sum(1 for s in scores if s >= SYNTHETIC_THRESHOLD)
    return {"vocab": vocab, "ttr": ttr, "trigram_diversity": dtr,
            "mean_synth_score": round(sum(scores) / len(scores), 1),
            "est_contamination": round(flagged / len(corpus), 3)}


def run_config(name, pool, size, det, rng):
    gen0 = pool[:]                       # generation 0 = full real seed pool (same for all sizes)
    traj = [dict(generation=0, n=len(gen0), **measure(gen0, det))]
    corpus = gen0
    for g in range(1, GENERATIONS + 1):
        corpus = FastTrigram().fit(corpus).sample_corpus(rng, size, DOC_LEN)
        traj.append(dict(generation=g, n=size, **measure(corpus, det)))
    g0, gN = traj[0], traj[-1]
    return {
        "dataset": name, "size": size, "generations": GENERATIONS,
        "trajectory": traj,
        "vocab_drop_pct": round(100 * (1 - gN["vocab"] / g0["vocab"]), 1) if g0["vocab"] else 0,
        "trigram_div_start": g0["trigram_diversity"], "trigram_div_end": gN["trigram_diversity"],
        "detector_start": g0["mean_synth_score"], "detector_end": gN["mean_synth_score"],
        "contamination_end": gN["est_contamination"],
    }


def main():
    fm.fetch_all()
    det = SyntheticTextProbe()
    raid = fm.load("human")
    hc3, _ = fetch_corpora.load_samples()
    datasets = [("RAID-human", raid), ("HC3-human", hc3)]

    results = []
    print("scaling collapse over sizes %s on %s" % (SIZES, [d[0] for d in datasets]), flush=True)
    for name, pool in datasets:
        for size in SIZES:
            rng = random.Random(SEED)
            p = pool[:]
            rng.shuffle(p)
            r = run_config(name, p, size, det, rng)
            results.append(r)
            print("  %-10s size=%6d  vocab -%4.1f%%  tri-div %.3f->%.3f  detector %.1f->%.1f  contam_end %.0f%%"
                  % (name, size, r["vocab_drop_pct"], r["trigram_div_start"], r["trigram_div_end"],
                     r["detector_start"], r["detector_end"], 100 * r["contamination_end"]), flush=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump({"config": {"generations": GENERATIONS, "doc_len": DOC_LEN, "seed": SEED},
                           "results": results}, f, indent=2)

    print("\nDONE ->", RESULTS_PATH)


if __name__ == "__main__":
    main()
