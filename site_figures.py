"""
site_figures.py -- regenerate the five figures used on the website in the site's
dark palette, so charts stop punching white holes in the dark theme.

Writes site/figures/dark_*.png. Run with the ML venv (matplotlib):
    .venv-ml/bin/python site_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "site", "figures")

# site palette
BG = "#0e1016"
PANEL = "#13151d"
INK = "#e7ebf3"
MUTED = "#8b93a7"
GRID = "#262b38"
GOLD = "#f3b24a"
BLUE = "#6aa8ff"
GREEN = "#46d39a"
RED = "#ff6b78"
VIOLET = "#9b8cff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.labelcolor": INK,
})


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- dark_figure5: ROC on HC3 ----
ev = load("evaluation_results.json")
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="chance")
colors = {"SyntheticTextProbe": GOLD, "PerplexityBaseline (bigram LM)": BLUE, "MeanWordLength (sanity floor)": VIOLET}
for m in ev["methods"]:
    roc = m.get("roc")
    if roc:
        c = colors.get(m["name"], INK)
        ax.plot([p[0] for p in roc], [p[1] for p in roc], color=c, lw=2.4,
                label="%s (AUC %.3f)" % (m["name"].split(" (")[0], m["auc"]))
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("ROC on HC3 — real human vs ChatGPT")
ax.legend(loc="lower right", fontsize=9)
save(fig, "dark_figure5_roc.png")

# ---- dark_figure6: cross-model bars ----
cm = load("crossmodel_results.json")
rows = [r for r in cm["models"] if r["model"] != "POOLED(all)"]
rows.sort(key=lambda r: -r["habsburg_auc"])
names = [r["model"] for r in rows]
hb = [r["habsburg_auc"] for r in rows]
px = [r["perplexity_auc"] for r in rows]
x = range(len(names))
fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax.bar([i - 0.21 for i in x], hb, width=0.42, color=GOLD, label="SyntheticTextProbe (zero-shot)")
ax.bar([i + 0.21 for i in x], px, width=0.42, color=BLUE, label="Perplexity (in-distribution)")
ax.axhline(0.5, ls="--", color=MUTED, lw=1)
ax.text(len(names) - 0.4, 0.515, "chance", color=MUTED, fontsize=9, ha="right")
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUC"); ax.set_ylim(0, 1.02)
ax.set_title("Cross-model detection (RAID): one fixed detector vs 8 generators")
ax.legend(fontsize=9, loc="lower right")
save(fig, "dark_figure6_crossmodel.png")

# ---- dark_figure7: LOMO grouped bars ----
lm = load("lomo_results.json")
folds = lm["folds"]
folds.sort(key=lambda f: -f["lomo_classifier_auc"])
names = [f["held_out"] for f in folds]
clf = [f["lomo_classifier_auc"] for f in folds]
ppl = [f["perplexity_auc"] for f in folds]
zs = [f["habsburg_zeroshot_auc"] for f in folds]
x = range(len(names))
fig, ax = plt.subplots(figsize=(9.6, 4.8))
ax.bar([i - 0.27 for i in x], clf, width=0.27, color=GOLD, label="LOMO classifier")
ax.bar([i for i in x], ppl, width=0.27, color=BLUE, label="Perplexity alone")
ax.bar([i + 0.27 for i in x], zs, width=0.27, color=VIOLET, label="Zero-shot probe")
ax.axhline(0.5, ls="--", color=MUTED, lw=1)
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("AUC on the unseen generator"); ax.set_ylim(0, 1.02)
ax.set_title("Leave-one-model-out: generalization to an unseen LLM")
ax.legend(fontsize=9, loc="lower right")
save(fig, "dark_figure7_lomo.png")

# ---- dark_figure8: neural collapse curves ----
nc = load("collapse_gpt2_results.json")
gens = nc["generations"]
g = [r["generation"] for r in gens]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
a1.plot(g, [r["trigram_diversity"] for r in gens], marker="o", color=BLUE, lw=2.2, label="trigram diversity")
a1.plot(g, [r["mean_synth_score"] / 100 for r in gens], marker="s", color=GOLD, lw=2.2, label="detector score / 100")
a1.axhline(0.5, ls="--", color=RED, lw=1)
a1.text(0.05, 0.52, "detector threshold", color=RED, fontsize=8.5)
a1.set_xlabel("generation"); a1.set_ylim(0, 1.02); a1.set_xticks(g)
a1.set_title("Diversity falls; the detector stays sub-threshold")
a1.legend(fontsize=9)
a2.plot(g, [r["ref_perplexity"] for r in gens], marker="o", color=RED, lw=2.4)
a2.set_xlabel("generation"); a2.set_ylabel("perplexity under fixed reference")
a2.set_xticks(g)
a2.set_title("Generated-text perplexity crashes 69 → 7")
save(fig, "dark_figure8_neural.png")

# ---- dark_figure10: collapse at scale ----
sc = load("collapse_scaling_results.json")["results"]
raid = sorted([r for r in sc if r["dataset"] == "RAID-human"], key=lambda r: r["size"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
palette = [GOLD, BLUE, GREEN]
for r, c in zip(raid, palette):
    gg = [t["generation"] for t in r["trajectory"]]
    a1.plot(gg, [t["trigram_diversity"] for t in r["trajectory"]], marker="o", color=c, lw=2.2,
            label="{:,} docs".format(r["size"]))
    a2.plot(gg, [t["mean_synth_score"] for t in r["trajectory"]], marker="o", color=c, lw=2.2,
            label="{:,} docs".format(r["size"]))
a1.set_xlabel("generation"); a1.set_ylabel("trigram diversity"); a1.set_ylim(0, 1.02)
a1.set_title("Distribution metric collapses at every size"); a1.legend(fontsize=9)
a2.axhline(50, ls="--", color=RED, lw=1)
a2.text(0.05, 52, "detector threshold", color=RED, fontsize=8.5)
a2.set_xlabel("generation"); a2.set_ylabel("per-document detector score"); a2.set_ylim(0, 62)
a2.set_title("Per-document detector never fires"); a2.legend(fontsize=9)
save(fig, "dark_figure10_scaling.png")

print("done")


# ---- dark_figure_multiseed: paired per-seed comparison (significance) ----
ms = load("lomo_multiseed_results.json")
runs = ms["runs"]
seeds = [str(r["seed"]) for r in runs]
clf = [r["mean_clf"] for r in runs]
px = [r["mean_px"] for r in runs]
x = range(len(runs))
fig, ax = plt.subplots(figsize=(9.6, 4.6))
for i in x:
    ax.plot([i, i], [px[i], clf[i]], color=GRID, lw=2, zorder=1)
ax.scatter(list(x), px, s=70, color=BLUE, zorder=3, label="perplexity alone")
ax.scatter(list(x), clf, s=70, color=GOLD, zorder=3, label="classifier (signals + perplexity)")
for i in x:
    ax.annotate("", xy=(i, clf[i] - 0.0006), xytext=(i, px[i] + 0.0006),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6), zorder=2)
ax.set_xticks(list(x)); ax.set_xticklabels(["seed\n" + s for s in seeds], fontsize=8.5)
ax.set_ylabel("mean held-out AUC")
ax.set_title("The classifier beats perplexity on 10/10 seeds  (paired permutation p = 0.002)")
ax.legend(fontsize=9, loc="lower right")
save(fig, "dark_figure_multiseed.png")

# ---- dark_figure_ablation: which signals carry the probe ----
ev2 = load("evaluation_results.json")
sigs = ev2["ablation"]["signals"]
sigs = sorted(sigs, key=lambda s: s["delta_auc"])
names = [s["signal"].replace("_", " ") for s in sigs]
delta = [s["delta_auc"] for s in sigs]
solo = [s["solo_auc"] for s in sigs]
y = range(len(sigs))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
a1.barh(list(y), delta, color=GOLD, height=0.62)
a1.set_yticks(list(y)); a1.set_yticklabels(names, fontsize=10)
a1.set_xlabel("AUC lost when the signal is removed")
a1.set_title("Removing it hurts…")
a2.barh(list(y), solo, color=BLUE, height=0.62)
a2.axvline(0.5, ls="--", color=RED, lw=1.2)
a2.text(0.505, len(sigs) - 0.7, "chance", color=RED, fontsize=9)
a2.set_xlabel("AUC using only this signal")
a2.set_xlim(0.4, 0.9)
a2.set_title("…and how far it gets alone")
fig.suptitle("Inside the probe: which of the six signals carry the detection (HC3)", fontweight="bold", y=1.02, color=INK)
save(fig, "dark_figure_ablation.png")

# ---- dark_figure_universal: collapse across three architectures ----
tg = load("collapse_results.json")["generations"]
dg = load("collapse_gpt2_results.json")["generations"]
qw = load("collapse_modern_results.json")["generations"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for gens, color, lab in ((tg, VIOLET, "trigram LM"), (dg, GOLD, "distilGPT-2"), (qw, GREEN, "Qwen2.5-0.5B")):
    g = [r["generation"] for r in gens]
    a1.plot(g, [r["trigram_diversity"] for r in gens], marker="o", lw=2.2, color=color, label=lab)
a1.set_xlabel("generation"); a1.set_ylabel("trigram diversity"); a1.set_ylim(0, 1.02)
a1.set_title("Diversity falls for every architecture"); a1.legend(fontsize=9)
for gens, color, lab in ((dg, GOLD, "distilGPT-2 (69 → 7)"), (qw, GREEN, "Qwen2.5-0.5B (37 → 2.9)")):
    g = [r["generation"] for r in gens]
    p0 = gens[0]["ref_perplexity"]
    a2.plot(g, [100.0 * r["ref_perplexity"] / p0 for r in gens], marker="o", lw=2.2, color=color, label=lab)
a2.set_xlabel("generation"); a2.set_ylabel("reference perplexity (% of generation 0)")
a2.set_ylim(0, 105)
a2.set_title("Perplexity crashes for 2019 and 2024 models alike"); a2.legend(fontsize=9)
save(fig, "dark_figure_universal.png")
