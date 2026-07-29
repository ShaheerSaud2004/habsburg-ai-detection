"""
lomo_multiseed.py -- statistical significance for the leave-one-model-out result.

Repeats the bigram-perplexity LOMO experiment across N seeds (each seed re-draws
the human train/test partition), and reports, for the classifier vs perplexity-alone
vs zero-shot heuristic:
  * mean +/- standard deviation of the mean held-out AUC across seeds,
  * a 95% confidence interval (t-based),
  * a PAIRED, EXACT sign-flip permutation test answering "is the classifier
    really better than perplexity-alone?" (2^N sign assignments, no scipy).

Writes lomo_multiseed_results.json. Standard library only.

    python3 lomo_multiseed.py
"""

import itertools
import json
import math
import os
import random
import statistics

from detector import SyntheticTextProbe, combine_signals
from baselines import PerplexityBaseline
from evaluate import roc_auc
from lomo import LogisticRegression, signal_vector
import fetch_multimodel as fm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "lomo_multiseed_results.json")

SEEDS = [1234, 7, 13, 21, 42, 99, 101, 202, 303, 404]   # 1234 reproduces the headline
N_HUMAN_TRAIN = 2000
N_HUMAN_TEST = 400

# two-sided t critical values (alpha=0.05) by degrees of freedom
T95 = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def run_one(seed, human, machines, sig_human, sig_machine):
    idx = list(range(len(human)))
    random.Random(seed).shuffle(idx)
    tr_idx = idx[:N_HUMAN_TRAIN]
    te_idx = idx[N_HUMAN_TRAIN:N_HUMAN_TRAIN + N_HUMAN_TEST]
    h_train = [human[i] for i in tr_idx]
    s_train = [sig_human[i] for i in tr_idx]
    h_test = [human[i] for i in te_idx]
    s_test = [sig_human[i] for i in te_idx]

    fold = {}
    for held in fm.MODELS:
        train_models = [m for m in fm.MODELS if m != held]
        lm_texts = list(h_train) + [t for m in train_models for t in machines[m]]
        lm = PerplexityBaseline().fit(lm_texts)

        Xtr, ytr = [], []
        for t, s in zip(h_train, s_train):
            Xtr.append(signal_vector(s) + [lm.cross_entropy(t)]); ytr.append(0)
        for m in train_models:
            for t, s in zip(machines[m], sig_machine[m]):
                Xtr.append(signal_vector(s) + [lm.cross_entropy(t)]); ytr.append(1)

        Xte, yte, rs = [], [], []
        for t, s in zip(h_test, s_test):
            Xte.append(signal_vector(s) + [lm.cross_entropy(t)]); yte.append(0); rs.append(s)
        for t, s in zip(machines[held], sig_machine[held]):
            Xte.append(signal_vector(s) + [lm.cross_entropy(t)]); yte.append(1); rs.append(s)

        clf = LogisticRegression().fit(Xtr, ytr)
        auc_clf = roc_auc(clf.predict_proba(Xte), yte)
        auc_hb = roc_auc([combine_signals(s) for s in rs], yte)
        ptr = [r[-1] for r in Xtr]
        flip = roc_auc(ptr, ytr) < 0.5
        auc_px = roc_auc([(-r[-1] if flip else r[-1]) for r in Xte], yte)
        fold[held] = {"clf": auc_clf, "px": auc_px, "hb": auc_hb}

    return {
        "seed": seed,
        "per_fold": fold,
        "mean_clf": statistics.mean(v["clf"] for v in fold.values()),
        "mean_px": statistics.mean(v["px"] for v in fold.values()),
        "mean_hb": statistics.mean(v["hb"] for v in fold.values()),
    }


def msd(xs):
    return {"mean": round(statistics.mean(xs), 4),
            "std": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0}


def ci95(xs):
    n = len(xs)
    if n < 2:
        return [round(xs[0], 4), round(xs[0], 4)]
    t = T95.get(n - 1, 1.96)
    h = t * statistics.stdev(xs) / math.sqrt(n)
    m = statistics.mean(xs)
    return [round(m - h, 4), round(m + h, 4)]


def perm_test(diffs):
    """Exact paired sign-flip permutation test (two-sided)."""
    n = len(diffs)
    obs = abs(statistics.mean(diffs))
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = abs(sum(s * d for s, d in zip(signs, diffs)) / n)
        if m >= obs - 1e-12:
            hits += 1
    return hits / (2 ** n)


def write(runs, final=None):
    out = {"seeds_done": [r["seed"] for r in runs], "runs": runs}
    if final:
        out.update(final)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)


def main():
    fm.fetch_all()
    det = SyntheticTextProbe()
    human = fm.load("human")
    machines = {m: fm.load(m) for m in fm.MODELS}

    print("pre-computing detector signals (once) ...", flush=True)
    sig_human = [det.raw_signals(t) for t in human]
    sig_machine = {m: [det.raw_signals(t) for t in machines[m]] for m in fm.MODELS}

    runs = []
    for s in SEEDS:
        r = run_one(s, human, machines, sig_human, sig_machine)
        runs.append(r)
        print("seed %-5d clf=%.4f  px=%.4f  hb=%.4f   (clf-px=%+.4f)"
              % (s, r["mean_clf"], r["mean_px"], r["mean_hb"], r["mean_clf"] - r["mean_px"]),
              flush=True)
        write(runs)

    clf = [r["mean_clf"] for r in runs]
    px = [r["mean_px"] for r in runs]
    hb = [r["mean_hb"] for r in runs]
    diffs = [c - p for c, p in zip(clf, px)]

    final = {
        "n_seeds": len(SEEDS),
        "classifier": {**msd(clf), "ci95": ci95(clf)},
        "perplexity_alone": {**msd(px), "ci95": ci95(px)},
        "zero_shot": {**msd(hb), "ci95": ci95(hb)},
        "classifier_vs_perplexity": {
            "mean_diff": round(statistics.mean(diffs), 4),
            "std_diff": round(statistics.stdev(diffs), 4),
            "seeds_classifier_wins": sum(1 for d in diffs if d > 0),
            "n_seeds": len(diffs),
            "paired_permutation_p": round(perm_test(diffs), 4),
        },
    }
    write(runs, final)

    print("\n" + "=" * 64)
    print("MULTI-SEED LOMO (%d seeds) -- mean +/- std of mean held-out AUC" % len(SEEDS))
    print("=" * 64)
    for name, d in (("classifier", final["classifier"]),
                    ("perplexity-alone", final["perplexity_alone"]),
                    ("zero-shot heuristic", final["zero_shot"])):
        print("  %-20s %.4f +/- %.4f   95%% CI [%.4f, %.4f]"
              % (name, d["mean"], d["std"], d["ci95"][0], d["ci95"][1]))
    cv = final["classifier_vs_perplexity"]
    print("\n  classifier vs perplexity-alone:")
    print("    mean diff = %+.4f +/- %.4f | classifier wins %d/%d seeds | paired perm p = %.4f"
          % (cv["mean_diff"], cv["std_diff"], cv["seeds_classifier_wins"], cv["n_seeds"], cv["paired_permutation_p"]))
    print("\nWrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
