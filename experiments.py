"""
experiments.py -- four controlled experiments on the Habsburg detector.

Runs fully offline (no API key, no pip installs), writes experiment_results.json,
and prints a summary table.

  Exp 1  accuracy vs contamination (0% .. 100%)
  Exp 2  signal degradation / separation across generations 1..4
  Exp 3  self-contamination: catching recycled model outputs
  Exp 4  false-positive rate on pure human data
"""

import json
import os
import random

from detector import SyntheticTextProbe
import data_generator as dg

SEED = 1234
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "experiment_results.json")


def _metrics(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l == "synthetic" and p == "synthetic")
    tn = sum(1 for l, p in zip(labels, preds) if l == "human" and p == "human")
    fp = sum(1 for l, p in zip(labels, preds) if l == "human" and p == "synthetic")
    fn = sum(1 for l, p in zip(labels, preds) if l == "synthetic" and p == "human")
    total = len(labels)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def _predict(detector, texts):
    return ["synthetic" if detector.score_text(t)["is_synthetic"] else "human" for t in texts]


def exp1_accuracy_vs_contamination(rng):
    detector = SyntheticTextProbe()
    out = {}
    for i in range(11):
        c = i / 10.0
        data = dg.make_dataset(rng, 40, c, generation=2)
        texts = [t for t, _ in data]
        labels = [lab for _, lab in data]
        preds = _predict(detector, texts)
        acc, prec, rec, f1 = _metrics(labels, preds)
        det = detector.detect_dataset(texts)
        out["%.2f" % c] = {
            "accuracy": round(acc, 3), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3),
            "estimated": det["estimated_contamination"], "true": round(c, 3),
        }
    return out


def exp2_generation_degradation(rng):
    detector = SyntheticTextProbe()
    datasets = dg.simulate_generations(rng, 4, n_per_gen=60)
    out = {}
    for g, data in enumerate(datasets, start=1):
        texts = [t for t, _ in data]
        labels = [lab for _, lab in data]
        acc, _, _, _ = _metrics(labels, _predict(detector, texts))
        syn = [detector.score_text(t)["synthetic_likelihood"] for t, lab in data if lab == "synthetic"]
        hum = [detector.score_text(t)["synthetic_likelihood"] for t, lab in data if lab == "human"]
        sep = (sum(syn) / len(syn) if syn else 0.0) - (sum(hum) / len(hum) if hum else 0.0)
        out["generation_%d" % g] = {"accuracy": round(acc, 3), "mean_separation": round(sep, 2)}
    return out


def exp3_self_contamination(rng):
    gen1 = [dg.generate_synthetic_text(rng, generation=1) for _ in range(20)]
    detector = SyntheticTextProbe(reference_synthetic=gen1)
    injected = rng.sample(gen1, 10)  # recycle exact gen-1 outputs back into the pipeline
    caught = sum(
        1 for t in injected
        if detector.score_text(t)["signals"]["duplicate_similarity"] >= 0.5
    )
    return {
        "injected": len(injected),
        "caught": caught,
        "catch_rate": round(caught / len(injected), 3) if injected else 0.0,
    }


def exp4_false_positive(rng):
    detector = SyntheticTextProbe()
    human = [dg.generate_human_text(rng) for _ in range(100)]
    flagged = sum(1 for p in _predict(detector, human) if p == "synthetic")
    return {"rate": round(flagged / len(human), 3), "flagged": flagged, "total": len(human)}


def main():
    rng = random.Random(SEED)
    results = {
        "contamination_levels": exp1_accuracy_vs_contamination(rng),
        "generation_degradation": exp2_generation_degradation(rng),
        "self_contamination": exp3_self_contamination(rng),
        "false_positive_rate": exp4_false_positive(rng),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("HABSBURG AI DETECTION -- EXPERIMENT RESULTS")
    print("=" * 60)

    print("\n[Exp 1] Detection accuracy vs contamination")
    for k in sorted(results["contamination_levels"], key=float):
        r = results["contamination_levels"][k]
        print("  true=%3d%%   acc=%.2f   est=%.2f   f1=%.2f"
              % (int(float(k) * 100), r["accuracy"], r["estimated"], r["f1"]))

    print("\n[Exp 2] Score separation across generations")
    for g in sorted(results["generation_degradation"]):
        r = results["generation_degradation"][g]
        print("  %-13s acc=%.2f   separation=%.1f pts" % (g, r["accuracy"], r["mean_separation"]))

    print("\n[Exp 3] Self-contamination (recycled-output detection)")
    r = results["self_contamination"]
    print("  injected=%d   caught=%d   catch_rate=%.2f" % (r["injected"], r["caught"], r["catch_rate"]))

    print("\n[Exp 4] False-positive rate on pure human data")
    r = results["false_positive_rate"]
    print("  flagged=%d / %d   rate=%.2f" % (r["flagged"], r["total"], r["rate"]))

    print("\nWrote", RESULTS_PATH)
    print("Next:  python3 visualizations.py   (needs matplotlib)")


if __name__ == "__main__":
    main()
