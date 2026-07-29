"""
main.py -- offline live demo. Runs five labeled scenarios through the
Habsburg detector and prints the verdicts. Standard library only.

    python3 main.py
"""

import random

from detector import SyntheticTextProbe
import data_generator as dg

SEED = 7


def _show(title, detector, texts):
    res = detector.detect_dataset(texts)
    print("=" * 66)
    print(title)
    print("-" * 66)
    print("  texts: %d  |  flagged synthetic: %d  |  estimated contamination: %d%%"
          % (res["total"], res["predicted_synthetic"],
             round(100 * res["estimated_contamination"])))
    for t in texts[:2]:
        r = detector.score_text(t)
        snippet = (t[:64] + "...") if len(t) > 64 else t
        print("    [%5.1f] %s" % (r["synthetic_likelihood"], snippet))
    print()


def main():
    rng = random.Random(SEED)

    human = [dg.generate_human_text(rng) for _ in range(20)]
    _show("Scenario 1 -- clean human corpus (expect ~0% flagged)",
          SyntheticTextProbe(), human)

    light = dg.make_dataset(rng, 20, 0.15, generation=2)
    _show("Scenario 2 -- lightly contaminated (~15% synthetic)",
          SyntheticTextProbe(), [t for t, _ in light])

    heavy = dg.make_dataset(rng, 20, 0.60, generation=2)
    _show("Scenario 3 -- heavily contaminated (~60% synthetic)",
          SyntheticTextProbe(), [t for t, _ in heavy])

    gen1 = [dg.generate_synthetic_text(rng, generation=1) for _ in range(15)]
    detector = SyntheticTextProbe(reference_synthetic=gen1)
    recycled = rng.sample(gen1, 8) + [dg.generate_human_text(rng) for _ in range(12)]
    rng.shuffle(recycled)
    _show("Scenario 4 -- self-recycled outputs mixed into fresh data",
          detector, recycled)

    collapsed = [dg.generate_synthetic_text(rng, generation=4) for _ in range(20)]
    _show("Scenario 5 -- pure collapsed gen-4 synthetic (expect ~100% flagged)",
          SyntheticTextProbe(), collapsed)

    print("Done. For metrics, run:  python3 experiments.py")


if __name__ == "__main__":
    main()
