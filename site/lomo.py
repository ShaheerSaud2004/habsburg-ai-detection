"""
lomo.py -- Leave-One-Model-Out (LOMO) evaluation: the fair head-to-head.

For each of the 8 RAID generators M:
  * TRAIN a logistic-regression classifier on the OTHER 7 generators (+ human),
    using features = [6 detector signals + perplexity].
  * TEST on the held-out generator M (which the classifier has never seen).

This is the honest test of generalization to UNSEEN models -- nobody gets an
in-distribution advantage. We compare three scorers on the SAME held-out test set:

  1. LOMO classifier   (trained on 7 other generators)
  2. SyntheticTextProbe  (zero-shot heuristic, no training)
  3. Perplexity        (bigram LM fit on the train fold, sign-calibrated on train)

Everything is standard library: the logistic regression and feature standardization
are hand-rolled. Writes lomo_results.json.

    python3 lomo.py        # uses the RAID corpus cached by fetch_multimodel.py
"""

import argparse
import hashlib
import json
import math
import os
import random

from detector import SyntheticTextProbe, combine_signals
from baselines import PerplexityBaseline
from evaluate import roc_auc, best_f1_metrics
import fetch_multimodel as fm

SEED = 1234
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "lomo_results.json")

N_HUMAN_TRAIN = 2000
N_HUMAN_TEST = 400      # kept constant across folds so only the generator changes

FEATURE_NAMES = ["repetition", "filler_density", "lexical_diversity",
                 "ngram_diversity", "burstiness", "duplicate_similarity", "perplexity"]


def _h(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def load_gpt2_cache():
    path = os.path.join(HERE, "corpora", "gpt2_ppl.tsv")
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) == 2:
                    try:
                        cache[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    return cache


# ---------------- hand-rolled logistic regression (stdlib) ----------------

class LogisticRegression:
    def __init__(self, lr=0.5, epochs=300, l2=1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.mean = None
        self.std = None

    def _standardize_fit(self, X):
        n = len(X)
        d = len(X[0])
        self.mean = [sum(row[j] for row in X) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / n
            s = math.sqrt(var)
            self.std.append(s if s > 1e-9 else 1.0)  # guard zero-variance (e.g. duplicate_similarity)

    def _standardize(self, X):
        return [[(row[j] - self.mean[j]) / self.std[j] for j in range(len(row))] for row in X]

    @staticmethod
    def _sigmoid(z):
        if z < -35:
            return 0.0
        if z > 35:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        self._standardize_fit(X)
        Z = self._standardize(X)
        n = len(Z)
        d = len(Z[0])
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                p = self._sigmoid(sum(self.w[j] * Z[i][j] for j in range(d)) + self.b)
                err = p - y[i]
                for j in range(d):
                    gw[j] += err * Z[i][j]
                gb += err
            for j in range(d):
                self.w[j] = self.w[j] - self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * (gb / n)
        return self

    def predict_proba(self, X):
        Z = self._standardize(X)
        return [self._sigmoid(sum(self.w[j] * z[j] for j in range(len(z))) + self.b) for z in Z]


# ---------------- features ----------------

def signal_vector(sig):
    return [sig["repetition"], sig["filler_density"], sig["lexical_diversity"],
            sig["ngram_diversity"], sig["burstiness"], sig["duplicate_similarity"]]


def main(ppl_source="bigram"):
    fm.fetch_all()
    detector = SyntheticTextProbe()

    human = fm.load("human")
    rng = random.Random(SEED)
    rng.shuffle(human)
    human_train = human[:N_HUMAN_TRAIN]
    human_test = human[N_HUMAN_TRAIN:N_HUMAN_TRAIN + N_HUMAN_TEST]

    machines = {m: fm.load(m) for m in fm.MODELS}

    # Pre-compute the 6 detector signals once (they do not depend on the fold;
    # duplicate_similarity is 0 here because there is no reference corpus).
    print("pre-computing detector signals ...")
    sig_human_train = [detector.raw_signals(t) for t in human_train]
    sig_human_test = [detector.raw_signals(t) for t in human_test]
    sig_machine = {m: [detector.raw_signals(t) for t in machines[m]] for m in fm.MODELS}

    gpt2_cache = None
    if ppl_source == "gpt2":
        gpt2_cache = load_gpt2_cache()
        all_texts = list(human_train) + list(human_test) + [t for m in fm.MODELS for t in machines[m]]
        missing = sum(1 for t in all_texts if _h(t) not in gpt2_cache)
        if not gpt2_cache or missing:
            raise SystemExit(
                "GPT-2 perplexity cache missing/incomplete (%d/%d texts missing).\n"
                "Build it first with:  .venv-ml/bin/python gpt2_perplexity.py"
                % (missing, len(all_texts)))
        print("perplexity feature: GPT-2 (%d cached values)" % len(gpt2_cache))
    else:
        print("perplexity feature: bigram LM (refit per fold)")

    folds = []
    coef_accum = [0.0] * len(FEATURE_NAMES)

    for held in fm.MODELS:
        train_models = [m for m in fm.MODELS if m != held]

        if ppl_source == "gpt2":
            def ppl(t):
                return gpt2_cache[_h(t)]
        else:
            # Perplexity LM fit on the TRAIN fold only (human_train + 7 generators).
            lm_texts = list(human_train) + [t for m in train_models for t in machines[m]]
            lm = PerplexityBaseline().fit(lm_texts)

            def ppl(t, _lm=lm):
                return _lm.cross_entropy(t)

        # Build training matrix.
        Xtr, ytr = [], []
        for t, sig in zip(human_train, sig_human_train):
            Xtr.append(signal_vector(sig) + [ppl(t)]); ytr.append(0)
        for m in train_models:
            for t, sig in zip(machines[m], sig_machine[m]):
                Xtr.append(signal_vector(sig) + [ppl(t)]); ytr.append(1)

        # Build held-out test matrix (human_test + the unseen generator).
        Xte, yte, raw_sig_te = [], [], []
        for t, sig in zip(human_test, sig_human_test):
            Xte.append(signal_vector(sig) + [ppl(t)]); yte.append(0); raw_sig_te.append(sig)
        for t, sig in zip(machines[held], sig_machine[held]):
            Xte.append(signal_vector(sig) + [ppl(t)]); yte.append(1); raw_sig_te.append(sig)

        # 1) LOMO classifier
        clf = LogisticRegression().fit(Xtr, ytr)
        p_clf = clf.predict_proba(Xte)
        auc_clf = roc_auc(p_clf, yte)
        f1_clf = best_f1_metrics(p_clf, yte)["f1"]
        for j in range(len(coef_accum)):
            coef_accum[j] += abs(clf.w[j])

        # 2) Zero-shot SyntheticTextProbe on the same test set
        p_hb = [combine_signals(s) for s in raw_sig_te]
        auc_hb = roc_auc(p_hb, yte)

        # 3) Perplexity alone (sign calibrated on train fold)
        ptr = [row[-1] for row in Xtr]
        flip = roc_auc(ptr, ytr) < 0.5
        p_px = [(-row[-1] if flip else row[-1]) for row in Xte]
        auc_px = roc_auc(p_px, yte)

        fold = {"held_out": held, "n_test": len(yte),
                "lomo_classifier_auc": round(auc_clf, 3), "lomo_classifier_f1": f1_clf,
                "habsburg_zeroshot_auc": round(auc_hb, 3), "perplexity_auc": round(auc_px, 3)}
        folds.append(fold)
        print("  held-out %-13s LOMO=%.3f  zero-shot=%.3f  perplexity=%.3f"
              % (held, auc_clf, auc_hb, auc_px))

    def mean(key):
        return round(sum(f[key] for f in folds) / len(folds), 3)

    n_folds = len(folds)
    coef_mean = [round(coef_accum[j] / n_folds, 3) for j in range(len(coef_accum))]
    importance = sorted(zip(FEATURE_NAMES, coef_mean), key=lambda x: -x[1])

    summary = {
        "corpus": "RAID, leave-one-model-out over 8 generators",
        "perplexity_source": ppl_source,
        "features": FEATURE_NAMES,
        "folds": folds,
        "mean_auc": {
            "lomo_classifier": mean("lomo_classifier_auc"),
            "habsburg_zeroshot": mean("habsburg_zeroshot_auc"),
            "perplexity": mean("perplexity_auc"),
        },
        "mean_abs_coef": dict(importance),
    }
    results_path = os.path.join(
        HERE, "lomo_gpt2_results.json" if ppl_source == "gpt2" else "lomo_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("LEAVE-ONE-MODEL-OUT (train on 7 generators, test on the unseen 8th)")
    print("perplexity feature = %s" % ppl_source)
    print("=" * 70)
    print("  %-14s %6s %10s %11s %11s" % ("held-out", "n", "LOMO clf", "zero-shot", "perplexity"))
    print("  " + "-" * 56)
    for f in folds:
        print("  %-14s %6d %10.3f %11.3f %11.3f"
              % (f["held_out"], f["n_test"], f["lomo_classifier_auc"],
                 f["habsburg_zeroshot_auc"], f["perplexity_auc"]))
    print("  " + "-" * 56)
    print("  %-14s %6s %10.3f %11.3f %11.3f"
          % ("MEAN", "", summary["mean_auc"]["lomo_classifier"],
             summary["mean_auc"]["habsburg_zeroshot"], summary["mean_auc"]["perplexity"]))
    print("\n  Classifier feature importance (mean |standardized coef| across folds):")
    for name, c in importance:
        print("    %-22s %.3f" % (name, c))
    print("\nWrote", results_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Leave-one-model-out evaluation")
    ap.add_argument("--ppl", choices=["bigram", "gpt2"], default="bigram",
                    help="perplexity feature source (gpt2 needs gpt2_perplexity.py cache)")
    main(ap.parse_args().ppl)
