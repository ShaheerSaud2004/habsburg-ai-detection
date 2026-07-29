"""
evaluate.py -- HONEST evaluation on REAL human-vs-LLM text (HC3).

Pipeline:
  1. load real human + real ChatGPT answers (fetch_corpora)
  2. stratified train/test split (seeded)
  3. score the test set with:
        - SyntheticTextProbe (no fitting needed)
        - PerplexityBaseline (fit on train; sign calibrated on train)
        - MeanWordLengthBaseline (sanity floor)
  4. report ROC-AUC (threshold-free) + accuracy/precision/recall/F1 at the
     best-F1 threshold, for each method, on the SAME test set
  5. write evaluation_results.json

AUC is computed with a stdlib rank-based (Mann-Whitney) formula -- no sklearn.

Positive class = synthetic / LLM (label 1). Negative = human (label 0).
"""

import json
import os
import random

from detector import SyntheticTextProbe, combine_signals, WEIGHTS
from baselines import PerplexityBaseline, MeanWordLengthBaseline
import fetch_corpora

SEED = 1234
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "evaluation_results.json")


def roc_auc(scores, labels):
    """Rank-based AUC with average ranks for ties. labels in {0,1}."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [idx for idx in range(n) if labels[idx] == 1]
    n_pos = len(pos)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = sum(ranks[i] for i in pos)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(scores, labels, max_points=60):
    """Return downsampled [fpr, tpr] points by sweeping the threshold high->low."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [[0.0, 0.0], [1.0, 1.0]]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    pts = [[0.0, 0.0]]
    for i in order:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        pts.append([round(fp / n_neg, 4), round(tp / n_pos, 4)])
    if len(pts) > max_points:
        step = len(pts) / max_points
        pts = [pts[int(k * step)] for k in range(max_points)] + [pts[-1]]
    return pts


def best_f1_metrics(scores, labels):
    """Sweep thresholds, return metrics at the threshold that maximizes F1."""
    cands = sorted(set(scores))
    best = None
    for thr in cands:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / len(labels) if labels else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": round(thr, 4), "accuracy": round(acc, 3),
                    "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    return best or {"threshold": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def stratified_split(human, llm, frac_train=0.7, seed=SEED):
    rng = random.Random(seed)
    h = human[:]; l = llm[:]
    rng.shuffle(h); rng.shuffle(l)
    hc = int(len(h) * frac_train); lc = int(len(l) * frac_train)
    train = [(t, 0) for t in h[:hc]] + [(t, 1) for t in l[:lc]]
    test = [(t, 0) for t in h[hc:]] + [(t, 1) for t in l[lc:]]
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def evaluate_method(name, score_fn, test, train=None, calibrate_sign=False):
    labels = [y for _, y in test]
    if calibrate_sign and train is not None:
        tr_scores = [score_fn(t) for t, _ in train]
        tr_labels = [y for _, y in train]
        auc_tr = roc_auc(tr_scores, tr_labels)
        flip = auc_tr < 0.5
    else:
        flip = False
    scores = [(-score_fn(t) if flip else score_fn(t)) for t, _ in test]
    auc = roc_auc(scores, labels)
    metrics = best_f1_metrics(scores, labels)
    roc = roc_curve(scores, labels)
    return {"name": name, "auc": round(auc, 3), "sign_flipped": flip, "roc": roc, **metrics}


def main():
    print("Loading real corpus (HC3) ...")
    human, llm = fetch_corpora.load_samples()
    print("  human=%d  llm=%d" % (len(human), len(llm)))

    train, test = stratified_split(human, llm)
    print("  train=%d  test=%d" % (len(train), len(test)))

    detector = SyntheticTextProbe()
    base = PerplexityBaseline().fit([t for t, _ in train])
    mwl = MeanWordLengthBaseline()

    results = {
        "corpus": "HC3 (real human vs real ChatGPT)",
        "n_human": len(human), "n_llm": len(llm),
        "n_train": len(train), "n_test": len(test),
        "methods": [
            evaluate_method("SyntheticTextProbe",
                            lambda t: detector.score_text(t)["synthetic_likelihood"], test),
            evaluate_method("PerplexityBaseline (bigram LM)",
                            base.score, test, train=train, calibrate_sign=True),
            evaluate_method("MeanWordLength (sanity floor)",
                            mwl.score, test, train=train, calibrate_sign=True),
        ],
    }

    # ---- per-signal ablation for SyntheticTextProbe on this real test set ----
    test_sigs = [detector.raw_signals(t) for t, _ in test]
    test_labels = [y for _, y in test]
    full_auc = roc_auc([combine_signals(s) for s in test_sigs], test_labels)
    abl = {"full_auc": round(full_auc, 3), "signals": []}
    for key in WEIGHTS:
        loo = roc_auc([combine_signals(s, drop=[key]) for s in test_sigs], test_labels)
        others = [k for k in WEIGHTS if k != key]
        solo = roc_auc([combine_signals(s, drop=others) for s in test_sigs], test_labels)
        abl["signals"].append({
            "signal": key,
            "leave_one_out_auc": round(loo, 3),
            "delta_auc": round(full_auc - loo, 3),  # how much AUC drops without it
            "solo_auc": round(solo, 3),             # AUC using only this signal
        })
    abl["signals"].sort(key=lambda x: -x["delta_auc"])
    results["ablation"] = abl

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 68)
    print("REAL-DATA EVALUATION (HC3) -- positive class = LLM/synthetic")
    print("=" * 68)
    print("  %-34s %6s %6s %6s %6s %6s" % ("method", "AUC", "acc", "prec", "rec", "F1"))
    print("  " + "-" * 64)
    for m in results["methods"]:
        print("  %-34s %6.3f %6.3f %6.3f %6.3f %6.3f"
              % (m["name"], m["auc"], m["accuracy"], m["precision"], m["recall"], m["f1"]))
    print("\n[Ablation] per-signal importance for SyntheticTextProbe (HC3 test)")
    print("  full AUC = %.3f   (dAUC = AUC lost when the signal is removed)" % abl["full_auc"])
    print("  %-22s %9s %8s %8s" % ("signal", "LOO-AUC", "dAUC", "solo"))
    print("  " + "-" * 50)
    for s in abl["signals"]:
        print("  %-22s %9.3f %8.3f %8.3f"
              % (s["signal"], s["leave_one_out_auc"], s["delta_auc"], s["solo_auc"]))

    print("\nWrote", RESULTS_PATH)
    print("\nReminder: AUC 0.5 = chance, 1.0 = perfect. Report these numbers honestly;")
    print("they are the real measure of whether the heuristics transfer to real text.")


if __name__ == "__main__":
    main()
