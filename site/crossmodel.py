"""
crossmodel.py -- cross-model generalization on RAID.

The detector is UNTRAINED, so its per-generator AUC directly measures how well a
single fixed detector generalizes across many LLMs. For each generator we build
a balanced human-vs-that-model test set and report AUC + F1 for SyntheticTextProbe
and (as a per-model reference) a PerplexityBaseline re-fit in-distribution.

Writes crossmodel_results.json.

    python3 crossmodel.py        # fetches the RAID corpus on first run
"""

import json
import os
import random

from detector import SyntheticTextProbe
from baselines import PerplexityBaseline
from evaluate import roc_auc, best_f1_metrics, stratified_split
import fetch_multimodel as fm

SEED = 1234
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "crossmodel_results.json")


def eval_pair(human, machine, seed=SEED):
    """Balanced human-vs-machine: subsample human to len(machine), split, score."""
    rng = random.Random(seed)
    h = human[:]
    rng.shuffle(h)
    h = h[:len(machine)]

    train, test = stratified_split(h, machine, frac_train=0.6, seed=seed)
    labels = [y for _, y in test]

    detector = SyntheticTextProbe()
    det_scores = [detector.score_text(t)["synthetic_likelihood"] for t, _ in test]

    base = PerplexityBaseline().fit([t for t, _ in train])
    tr_scores = [base.score(t) for t, _ in train]
    tr_labels = [y for _, y in train]
    flip = roc_auc(tr_scores, tr_labels) < 0.5
    base_scores = [(-base.score(t) if flip else base.score(t)) for t, _ in test]

    return {
        "n_test": len(test),
        "habsburg_auc": round(roc_auc(det_scores, labels), 3),
        "habsburg_f1": best_f1_metrics(det_scores, labels)["f1"],
        "perplexity_auc": round(roc_auc(base_scores, labels), 3),
    }


def main():
    fm.fetch_all()
    human = fm.load("human")
    print("\nhuman pool: %d" % len(human))

    rows = []
    all_machine = []
    for m in fm.MODELS:
        machine = fm.load(m)
        all_machine.extend(machine)
        r = eval_pair(human, machine)
        r["model"] = m
        r["n_machine"] = len(machine)
        rows.append(r)
        print("  %-14s n=%4d  Habsburg AUC=%.3f  Perplexity AUC=%.3f"
              % (m, r["n_machine"], r["habsburg_auc"], r["perplexity_auc"]))

    pooled = eval_pair(human, all_machine)
    pooled["model"] = "POOLED(all)"
    pooled["n_machine"] = len(all_machine)
    rows.append(pooled)

    aucs = [r["habsburg_auc"] for r in rows if r["model"] != "POOLED(all)"]
    summary = {
        "corpus": "RAID (real human vs 8 generators, attack=none)",
        "human_pool": len(human),
        "models": rows,
        "habsburg_auc_mean": round(sum(aucs) / len(aucs), 3),
        "habsburg_auc_min": min(aucs),
        "habsburg_auc_max": max(aucs),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print("CROSS-MODEL GENERALIZATION (RAID) -- one fixed, untrained detector")
    print("=" * 64)
    print("  %-14s %6s %12s %12s" % ("generator", "n", "Habsburg AUC", "Perplx AUC"))
    print("  " + "-" * 48)
    for r in rows:
        print("  %-14s %6d %12.3f %12.3f"
              % (r["model"], r["n_test"], r["habsburg_auc"], r["perplexity_auc"]))
    print("  " + "-" * 48)
    print("  Habsburg AUC across generators: mean=%.3f  min=%.3f  max=%.3f"
          % (summary["habsburg_auc_mean"], summary["habsburg_auc_min"], summary["habsburg_auc_max"]))
    print("\nWrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
