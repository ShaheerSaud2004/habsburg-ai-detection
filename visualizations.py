"""
visualizations.py -- render four publication figures from experiment_results.json.

Needs matplotlib (the ONLY part of the project that does). Degrades gracefully:
if matplotlib or the results file is missing, it prints what to do and exits 0.

    pip install matplotlib
    python3 experiments.py        # produces experiment_results.json
    python3 visualizations.py     # produces figure1..figure4 PNGs
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "experiment_results.json")


def concept_figure(plt, here):
    """The central-claim figure: two problems, and which metric answers which."""
    import matplotlib.patches as mpatches
    fig, (axd, axt) = plt.subplots(2, 1, figsize=(10, 6.2),
                                   gridspec_kw={"height_ratios": [1.0, 1.05]})

    axd.set_xlim(0, 30); axd.set_ylim(0, 10); axd.axis("off")
    axd.text(15, 9.5, "Two different problems", ha="center", fontsize=14, fontweight="bold")

    def box(x, y, text, fc, ec):
        axd.add_patch(mpatches.FancyBboxPatch((x - 3.3, y - 1.1), 6.6, 2.2,
                      boxstyle="round,pad=0.1,rounding_size=0.3", fc=fc, ec=ec, lw=1.5))
        axd.text(x, y, text, ha="center", va="center", fontsize=10.5)

    def arrow(x0, x1, y):
        axd.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="-|>", lw=1.7, color="#555"))

    box(4.2, 7.0, "AI-Text\nDetection", "#dceaff", "#5b8def")
    arrow(7.7, 10.4, 7.0); box(14, 7.0, "operates on\nONE document", "#eef4ff", "#5b8def")
    arrow(17.4, 20.3, 7.0); box(24, 7.0, "“is THIS text\nsynthetic?”", "#eef4ff", "#5b8def")

    box(4.2, 2.6, "Collapse\nMonitoring", "#ffe8cc", "#f4a236")
    arrow(7.7, 10.4, 2.6); box(14, 2.6, "operates on the\nDISTRIBUTION", "#fff3e3", "#f4a236")
    arrow(17.4, 20.3, 2.6); box(24, 2.6, "“is the CORPUS\nlosing diversity?”", "#fff3e3", "#f4a236")

    axt.axis("off")
    cols = ["Signal / metric", "Flag one AI document", "Detect corpus collapse"]
    rows = [
        ["Perplexity (under a fixed LM)", "Good", "Good"],
        ["Per-document detector score", "Good", "Poor"],
        ["Vocabulary size", "Poor", "Excellent"],
        ["Distinct n-gram diversity", "Poor", "Excellent"],
    ]
    cmap = {"Good": "#cdeccd", "Excellent": "#7cc47c", "Poor": "#f6c4c4"}
    cell_colours = [["#f1f3f6", cmap[r[1]], cmap[r[2]]] for r in rows]
    tbl = axt.table(cellText=rows, colLabels=cols, cellColours=cell_colours,
                    colColours=["#e7eaf0"] * 3, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfd5df")
        if r == 0:
            cell.set_text_props(fontweight="bold")
    axt.set_title("Which signal answers which question", fontsize=12, fontweight="bold", pad=4)

    fig.tight_layout()
    fig.savefig(os.path.join(here, "figure9_detection_vs_monitoring.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figure9_detection_vs_monitoring.png (concept)")


def scaling_figure(plt, here):
    """Collapse at scale: distribution metric falls at every size; detector stays flat."""
    path = os.path.join(here, "collapse_scaling_results.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)["results"]
    raid = sorted([r for r in data if r["dataset"] == "RAID-human"], key=lambda r: r["size"])
    if not raid:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for r in raid:
        g = [t["generation"] for t in r["trajectory"]]
        a1.plot(g, [t["trigram_diversity"] for t in r["trajectory"]], marker="o",
                label="%s docs" % format(r["size"], ","))
        a2.plot(g, [t["mean_synth_score"] for t in r["trajectory"]], marker="o",
                label="%s docs" % format(r["size"], ","))
    a1.set_xlabel("generation"); a1.set_ylabel("trigram diversity"); a1.set_ylim(0, 1.0)
    a1.set_title("Distribution metric collapses at every size"); a1.legend(fontsize=8)
    a2.axhline(50, ls="--", color="gray", lw=1)
    a2.text(0.05, 51, "detector threshold", fontsize=8, color="gray")
    a2.set_xlabel("generation"); a2.set_ylabel("per-document detector score"); a2.set_ylim(0, 60)
    a2.set_title("Per-document detector stays flat, sub-threshold"); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(here, "figure10_collapse_scaling.png"), dpi=120)
    plt.close(fig)
    print("Wrote figure10_collapse_scaling.png (scaling)")


def main():
    if not os.path.exists(RESULTS_PATH):
        print("experiment_results.json not found. Run:  python3 experiments.py  first.")
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with:  pip install matplotlib")
        print("(Only the figures need it; the rest of the system runs without it.)")
        return 0

    concept_figure(plt, HERE)
    scaling_figure(plt, HERE)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    levels = sorted(results["contamination_levels"], key=float)
    xs = [float(k) * 100 for k in levels]
    acc = [results["contamination_levels"][k]["accuracy"] for k in levels]
    est = [results["contamination_levels"][k]["estimated"] * 100 for k in levels]

    # Figure 1 -- accuracy vs contamination
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, acc, marker="o")
    ax.set_xlabel("true contamination (%)")
    ax.set_ylabel("detection accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection accuracy vs contamination")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figure1_accuracy_vs_contamination.png"), dpi=120)
    plt.close(fig)

    # Figure 2 -- score separation across generations
    gens = sorted(results["generation_degradation"])
    gx = [int(g.split("_")[1]) for g in gens]
    sep = [results["generation_degradation"][g]["mean_separation"] for g in gens]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gx, sep, marker="s", color="darkred")
    ax.set_xlabel("generation")
    ax.set_ylabel("synthetic - human mean score (pts)")
    ax.set_title("Signal separation across generations")
    ax.set_xticks(gx)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figure2_generation_degradation.png"), dpi=120)
    plt.close(fig)

    # Figure 3 -- estimated vs true contamination (risk framework)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0, 100], [0, 100], "--", color="gray", label="perfect estimate")
    ax.plot(xs, est, marker="o", color="teal", label="estimated")
    ax.set_xlabel("true contamination (%)")
    ax.set_ylabel("estimated contamination (%)")
    ax.set_title("Estimated vs true contamination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figure3_risk_framework.png"), dpi=120)
    plt.close(fig)

    # Figure 4 -- false positive rate (annotated so a 0% bar still reads clearly)
    fpr = results["false_positive_rate"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["measured"], [fpr["rate"]], color="orange", width=0.5,
           edgecolor="black", linewidth=1.2)
    ax.axhline(0.05, ls="--", color="gray", lw=1)
    ax.text(0.0, 0.053, "5% reference", color="gray", ha="center", fontsize=10)
    ax.text(0.0, 0.006, "%.1f%%  (%d/%d flagged)"
            % (100 * fpr["rate"], fpr["flagged"], fpr["total"]),
            ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.1)
    ax.set_ylabel("false-positive rate")
    ax.set_title("False positives on pure human stand-ins")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figure4_false_positive_rate.png"), dpi=120)
    plt.close(fig)

    # Figure 5 -- ROC on real data (only if evaluate.py has been run)
    eval_path = os.path.join(HERE, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
        for m in ev.get("methods", []):
            roc = m.get("roc")
            if roc:
                fpr = [p[0] for p in roc]
                tpr = [p[1] for p in roc]
                ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (m["name"], m["auc"]))
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("ROC on real human-vs-LLM text (%s)" % ev.get("corpus", "HC3"))
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figure5_roc_real_data.png"), dpi=120)
        plt.close(fig)
        print("Wrote figure5_roc_real_data.png (real-data ROC)")

    # Figure 6 -- cross-model generalization (only if crossmodel.py has been run)
    cm_path = os.path.join(HERE, "crossmodel_results.json")
    if os.path.exists(cm_path):
        with open(cm_path) as f:
            cm = json.load(f)
        models = [r for r in cm.get("models", []) if r["model"] != "POOLED(all)"]
        names = [r["model"] for r in models]
        hb = [r["habsburg_auc"] for r in models]
        px = [r["perplexity_auc"] for r in models]
        x = range(len(names))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar([i - 0.2 for i in x], hb, width=0.4, label="SyntheticTextProbe (zero-shot)")
        ax.bar([i + 0.2 for i in x], px, width=0.4, label="Perplexity (trained)")
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1.0)
        ax.set_title("Cross-model generalization (RAID): human vs each generator")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figure6_crossmodel.png"), dpi=120)
        plt.close(fig)
        print("Wrote figure6_crossmodel.png (cross-model AUC)")

    # Figure 7 -- leave-one-model-out (only if lomo.py has been run)
    lomo_path = os.path.join(HERE, "lomo_results.json")
    if os.path.exists(lomo_path):
        with open(lomo_path) as f:
            lm = json.load(f)
        folds = lm.get("folds", [])
        names = [r["held_out"] for r in folds]
        clf = [r["lomo_classifier_auc"] for r in folds]
        zs = [r["habsburg_zeroshot_auc"] for r in folds]
        px = [r["perplexity_auc"] for r in folds]
        x = range(len(names))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar([i - 0.27 for i in x], clf, width=0.27, label="LOMO classifier (signals+perplexity)")
        ax.bar([i for i in x], px, width=0.27, label="Perplexity alone")
        ax.bar([i + 0.27 for i in x], zs, width=0.27, label="Zero-shot heuristic")
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUC on the held-out (unseen) generator")
        ax.set_ylim(0, 1.0)
        ax.set_title("Leave-one-model-out: generalization to an unseen LLM (RAID)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figure7_lomo.png"), dpi=120)
        plt.close(fig)
        print("Wrote figure7_lomo.png (leave-one-model-out)")

    # Figure 8 -- neural collapse (only if collapse_gpt2.py has been run)
    ncp = os.path.join(HERE, "collapse_gpt2_results.json")
    if os.path.exists(ncp):
        with open(ncp) as f:
            nc = json.load(f)
        gens = nc.get("generations", [])
        gx = [r["generation"] for r in gens]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        a1.plot(gx, [r["trigram_diversity"] for r in gens], marker="o", label="trigram diversity")
        a1.plot(gx, [r["mean_synth_score"] / 100 for r in gens], marker="s", label="detector score /100")
        a1.plot(gx, [r["est_contamination"] for r in gens], marker="^", label="est. contamination")
        a1.set_xlabel("generation"); a1.set_ylabel("value (0-1)"); a1.set_ylim(0, 1.05)
        a1.set_xticks(gx); a1.set_title("Neural collapse: diversity vs detector"); a1.legend(fontsize=8)
        a2.plot(gx, [r["ref_perplexity"] for r in gens], marker="o", color="darkred")
        a2.set_xlabel("generation"); a2.set_ylabel("perplexity under fixed distilGPT-2")
        a2.set_xticks(gx); a2.set_title("Generated-text perplexity collapses 69 -> 7")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figure8_neural_collapse.png"), dpi=120)
        plt.close(fig)
        print("Wrote figure8_neural_collapse.png (neural collapse)")

    print("Wrote figures to", HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
