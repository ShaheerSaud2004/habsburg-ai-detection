"""
check_system.py -- health check before running anything.

    python3 check_system.py

Reports Python version, which optional deps are present, whether every core
module imports, and whether one detector call works. Never hard-crashes on a
missing optional dependency.
"""

import sys


def main():
    print("Habsburg AI Detection -- system check")
    print("-" * 44)
    ok = True

    print("Python:", sys.version.split()[0])
    if sys.version_info < (3, 8):
        print("  WARN: Python 3.8+ recommended")

    for mod, why in (("matplotlib", "figures"), ("anthropic", "optional live mode")):
        try:
            __import__(mod)
            print("  optional dep present : %s" % mod)
        except ImportError:
            print("  optional dep MISSING : %s  (ok -- only needed for %s)" % (mod, why))

    for mod in ("detector", "data_generator", "experiments", "main", "visualizations"):
        try:
            __import__(mod)
            print("  core module imports  : %s" % mod)
        except Exception as e:  # noqa: BLE001 - report anything that breaks import
            ok = False
            print("  ERROR importing %s -> %s" % (mod, e))

    try:
        from detector import SyntheticTextProbe
        r = SyntheticTextProbe().score_text(
            "This is a short test sentence, written by a human, probably."
        )
        print("  detector.score_text  : OK (synthetic_likelihood = %s)" % r["synthetic_likelihood"])
    except Exception as e:  # noqa: BLE001
        ok = False
        print("  ERROR running detector -> %s" % e)

    print("-" * 44)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
