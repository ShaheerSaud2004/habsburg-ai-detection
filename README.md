# Habsburg AI Detection

A small, **stdlib-only** research prototype for studying *Habsburg AI* /
**model collapse**: what happens when generative models are trained, generation
after generation, on the synthetic output of earlier models.

The system scores text 0–100 for how "synthetic / collapsed" it looks, estimates
the synthetic-contamination ratio of a dataset, and runs four controlled
experiments that demonstrate the idea end to end.

> **This was rebuilt locally on your Mac.** The original copy lived in a claude.ai
> web sandbox (`/home/claude/...`), which is ephemeral and gets wiped when the
> session ends — so it could not be found on disk. This is a faithful local rebuild.

---

## 📄 The paper

The full write-up is **[PAPER.md](PAPER.md)** / **[PAPER.pdf](PAPER.pdf)** — an 8-page,
workshop-format paper ("Per-Document AI-Text Detection Is Not a Reliable
Model-Collapse Monitor") covering every result below. Every number was
fact-checked against the result JSONs by an independent audit pass.

## 🛡️ TailGuard — the MVP built from this research

The paper's conclusion, productized: a **corpus-health monitor + CI contamination
tripwire** ([tailguard.py](tailguard.py), stdlib-only).

```bash
python3 tailguard.py scan data/ -o baseline.json          # snapshot health
python3 tailguard.py compare baseline.json data/          # CI tripwire (exit 0/1/2)
python3 tailguard.py compare baseline.json data/ --html report.html
```

Demo: contaminating a corpus with 60% collapsed/AI text moved the per-document
detector by almost nothing (16.2 → 15.5) but tripped TailGuard's distribution
metrics (trigram diversity −11.1% → WARNING). GitHub Action example:
[examples/tailguard-ci.yml](examples/tailguard-ci.yml).

## Files

| File | What it does |
|------|--------------|
| `detector.py` | Core engine: `SyntheticTextProbe` — six signals → 0–100 synthetic score |
| `data_generator.py` | Offline human/synthetic text + contaminated datasets + generation simulation |
| `experiments.py` | Four *controlled* experiments → writes `experiment_results.json` |
| `fetch_corpora.py` | Downloads + caches a **real** human-vs-LLM corpus (HC3) |
| `fetch_multimodel.py` | Pulls an 8-generator corpus from RAID (HF datasets-server) |
| `baselines.py` | Perplexity + mean-word-length baselines to compare against |
| `evaluate.py` | **Real-data** held-out evaluation + ROC/AUC + ablation → `evaluation_results.json` |
| `crossmodel.py` | **Cross-model** generalization across 8 LLMs → `crossmodel_results.json` |
| `lomo.py` | **Leave-one-model-out** trained classifier (signals+perplexity) → `lomo_results.json` |
| `gpt2_perplexity.py` | (venv) real GPT-2 perplexity for every text → `corpora/gpt2_ppl.tsv` |
| `collapse_experiment.py` | **Recursive model collapse** of a trigram LM, instrumented → `collapse_results.json` |
| `collapse_gpt2.py` | (venv) **Neural** recursive collapse: fine-tune distilGPT-2 on its own output → `collapse_gpt2_results.json` |
| `visualizations.py` | Five figures from the results JSON (needs `matplotlib`) |
| `main.py` | Offline demo: five labeled scenarios |
| `check_system.py` | Health check before you run anything |
| `requirements.txt` | Optional extras only (core needs nothing) |

## The six detection signals

For each text the detector measures (each normalized to 0–1):

1. **repetition** — fraction of repeated trigrams (looping phrasing → synthetic)
2. **filler_density** — canned filler phrases per ~50 tokens ("it is important to note…")
3. **lexical_diversity** — type-token ratio (low → synthetic)
4. **ngram_diversity** — distinct-trigram ratio (low → synthetic)
5. **burstiness** — sentence-length variation (uniform → synthetic)
6. **duplicate_similarity** — max Jaccard overlap with a corpus of *known* model
   outputs (catches **recycling**: a model's own output fed back into training)

These combine via documented weights into `synthetic_likelihood` (0–100); a score
≥ `SYNTHETIC_THRESHOLD` (50) is predicted synthetic.

## How to run

```bash
cd /Users/shaheersaud/habsburg-ai-detection

python3 check_system.py     # 1. confirm everything imports
python3 main.py             # 2. see the five scenarios
python3 experiments.py      # 3. run the 4 experiments -> experiment_results.json

# optional figures:
pip install -r requirements.txt
python3 visualizations.py   # -> figure1..figure4 PNGs
```

No API key and no pip install are required for steps 1–3.

## Real-data evaluation (the part that actually counts)

```bash
python3 evaluate.py        # downloads HC3 once (~70 MB), then evaluates
```

`evaluate.py` pulls the **HC3 corpus** (real human answers vs real ChatGPT
answers to the same questions), holds out a test set, and compares the detector
against baselines using **ROC-AUC** (threshold-free) on genuine text:

| Method | AUC | F1 |
|---|---|---|
| **SyntheticTextProbe** | **0.912** | 0.850 |
| Perplexity baseline (bigram LM) | 0.832 | 0.772 |
| Mean-word-length (sanity floor) | 0.646 | 0.679 |

So on HC3 (ChatGPT) the six heuristic signals **do** transfer and beat the
baseline — 0.91 is honest, not the ~1.0 the synthetic stand-ins give by
construction. **But HC3 is one model. The cross-model picture below is different
— and more important.**

### Which signals carry the weight (ablation, HC3)

`evaluate.py` also runs a leave-one-signal-out ablation:

| Signal | ΔAUC if removed | Solo AUC |
|---|---|---|
| burstiness | 0.037 | 0.735 |
| lexical_diversity | 0.036 | 0.848 |
| repetition | 0.012 | 0.848 |
| ngram_diversity | 0.004 | 0.850 |
| filler_density | 0.003 | 0.613 |
| duplicate_similarity | 0.000 | 0.500 |

**burstiness** and **lexical_diversity** do the heavy lifting. **duplicate_similarity
is dead weight here** (solo AUC 0.500 = chance) — it only fires when you supply a
reference corpus of known model outputs (the *recycling* scenario), which HC3
doesn't have. Honest: it is not a general AI-text signal.

### Cross-model generalization (RAID, 8 generators)

`crossmodel.py` evaluates one **fixed, untrained** detector against human text for
each of eight generators, balanced and held-out:

| Generator | Habsburg AUC (zero-shot) | Perplexity AUC (trained) |
|---|---|---|
| ChatGPT | 0.829 | 0.968 |
| Llama-chat | 0.794 | 0.934 |
| Mistral-chat | 0.764 | 0.945 |
| GPT-3 | 0.759 | 0.953 |
| GPT-4 | 0.703 | 0.897 |
| MPT-chat | 0.654 | 0.847 |
| Cohere | 0.610 | 0.845 |
| GPT-2 | 0.566 | 0.819 |
| **Pooled** | **0.723** | **0.908** |

Honest reading (this is the real finding):

- The zero-shot heuristics **transfer unevenly**: strongest on RLHF-chat models
  (ChatGPT 0.83, Llama-chat 0.79) whose "polished assistant" style is exactly what
  the filler/uniformity signals target; near **chance on GPT-2** base completions
  (0.57). Mean across generators ≈ **0.71**.
- **The perplexity baseline wins here, across the board** (pooled 0.91 vs 0.72).
  Caveat — it is *not* apples-to-apples: the baseline is **trained in-distribution**
  (it sees a train split of each model and RAID's structured domains), while the
  detector is fully **zero-shot**. RAID's long-form domains (abstracts/books/news)
  also favor an n-gram LM.
- Net: the heuristics are a useful *training-free* signal, but a trained model is
  more robust across generators. The right system **learns a classifier over the
  six signals + perplexity** and tests it **leave-one-model-out** so neither side
  has a distribution advantage.

This **flips** the HC3 takeaway — and that's the point. Multi-model evaluation is
where you find out what actually holds. Quote the RAID numbers, not just HC3.

### The fair head-to-head: leave-one-model-out (the strongest result)

The cross-model table is unfair to the detector (zero-shot vs an
in-distribution-trained baseline). `lomo.py` fixes that: a logistic regression
over **[6 signals + perplexity]** is trained on **7 generators** and tested on the
**held-out 8th it has never seen**. No model gets an in-distribution advantage.

| Held-out (unseen) | LOMO classifier | Zero-shot heuristic | Perplexity alone |
|---|---|---|---|
| Mistral-chat | 0.981 | 0.784 | 0.965 |
| ChatGPT | 0.976 | 0.848 | 0.968 |
| GPT-3 | 0.962 | 0.765 | 0.967 |
| MPT-chat | 0.950 | 0.679 | 0.885 |
| Llama-chat | 0.942 | 0.814 | 0.936 |
| GPT-4 | 0.907 | 0.734 | 0.868 |
| Cohere | 0.837 | 0.636 | 0.862 |
| GPT-2 | 0.834 | 0.594 | 0.840 |
| **Mean** | **0.924** | **0.732** | **0.911** |

Honest reading (this is the defensible contribution):

- A simple 7-feature model trained on *other* generators **generalizes to an
  unseen generator at mean AUC 0.92**, beating both the zero-shot heuristic (0.73)
  and perplexity alone (0.91). On the hardest case (GPT-2) it lifts a near-chance
  heuristic (0.59) up to 0.83.
- **Feature importance** (mean |standardized coef| across folds):
  `perplexity 2.73 ≫ lexical_diversity 1.24 > repetition 0.59 > burstiness 0.29 >
  filler 0.14 ≈ ngram 0.13`, `duplicate_similarity 0.00`. Perplexity is the
  workhorse; the cheap statistics add a **small but consistent** lift;
  duplicate_similarity is inert without a reference corpus (it only matters for
  the recycling scenario).
- So don't oversell the heuristics — perplexity does most of the work — but the
  combination is interpretable, training-free to compute, and generalizes across
  unseen LLMs. That is a real, reviewer-proof claim.

#### Real GPT-2 perplexity vs the bigram LM — and a corrected prediction

`gpt2_perplexity.py` (run with `.venv-ml`) caches a **real GPT-2** perplexity per
text; `python3 lomo.py --ppl gpt2` re-runs LOMO with it. Only that one script
touches torch — the cache is plain text and `lomo.py` stays stdlib.

| Perplexity feature | LOMO classifier (mean AUC) | Perplexity alone (mean AUC) |
|---|---|---|
| Bigram LM (refit per fold, in-distribution) | **0.924** | **0.911** |
| Real GPT-2 (fixed, zero-shot) | 0.895 | 0.864 |

I predicted a real GPT-2 perplexity would *raise* the floor (esp. on GPT-2/Cohere).
**It didn't** — the small bigram LM won on average. Per-model, GPT-2 perplexity
helped only on GPT-2 (0.857 vs 0.840) and Llama-chat, and *hurt* on GPT-4 (0.756 vs
0.868), MPT-chat, and Cohere.

Why, honestly: the bigram LM is **refit in-distribution** on RAID's exact domain mix
each fold, so it captures corpus-specific statistics; GPT-2 perplexity is a **fixed,
zero-shot** signal measuring "GPT-2-likeness," which is weak for newer/stronger
generators. So it is *not* apples-to-apples — and reaching 0.86–0.90 with **zero**
corpus fitting is arguably the more *deployable* signal. Note also that when the
perplexity feature is weaker (GPT-2), the classifier leans harder on the cheap text
statistics (lexical_diversity coef 1.24 → 1.46, repetition 0.59 → 0.90) — the
interpretable, sensible behavior.

Takeaway: don't assume a bigger model is a better feature. A cheap in-distribution
bigram LM is a surprisingly strong perplexity signal; GPT-2 perplexity is
competitive zero-shot but does not dominate.

### Does the detector catch model collapse? (`collapse_experiment.py`)

The experiment that ties back to *model collapse* itself: a small **trigram LM** is
trained on a human seed corpus, samples a new corpus, and the next generation is
trained on that — Shumailov-style recursion.

| Gen | Vocabulary | Trigram diversity | Detector score | Est. contamination |
|---|---|---|---|---|
| 0 (human seed) | 6,707 | 0.943 | 16.7 | 0% |
| 2 | 2,501 | 0.591 | 10.8 | 0% |
| 4 | 1,704 | 0.370 | 11.7 | 1% |
| 6 | 1,287 | 0.270 | 11.9 | 1% |

Two honest findings:

1. **Collapse is real and severe** — vocabulary **−81%**, trigram-diversity
   0.94 → 0.27 in six generations (a trigram LM can never re-introduce a word it
   stopped sampling, so diversity only falls).
2. **The per-document detector does NOT flag it** — its score stays flat (~11–17),
   contamination ~0%. The detector keys on *ChatGPT-style* per-document cues
   (filler, uniform structure); recursive n-gram collapse instead destroys
   **distribution-level** diversity while each short sampled document still looks
   locally varied.

The lesson: **detecting "this text is AI" (per-document) and detecting "this corpus
is collapsing" (distribution-level) are different problems.** Collapse lives in
corpus diversity (vocab, distinct-trigram ratio) — that's what a collapse monitor
should track, and this script is the harness for it.

### Neural collapse: distilGPT-2 fine-tuned on its own output (`collapse_gpt2.py`)

The neural counterpart (run with `.venv-ml`, uses the MPS GPU). Each generation, a
fresh distilGPT-2 is fine-tuned on the previous generation's text and samples a new
corpus. We add the generated text's **perplexity under a fixed reference distilGPT-2**
as a fluency gauge.

| Gen | Vocabulary | Trigram diversity | Ref perplexity | Detector score | Est. contamination |
|---|---|---|---|---|---|
| 0 (human seed) | 6,707 | 0.943 | 69.2 | 16.7 | 0% |
| 1 | 1,708 | 0.926 | 25.7 | 16.3 | 1% |
| 3 | 1,196 | 0.768 | 9.5 | 21.7 | 5% |
| 5 | 1,012 | 0.639 | 7.4 | 27.1 | 9% |

Findings:

- **Collapse is unambiguous** — vocabulary **−85%**, trigram-diversity 0.94 → 0.64,
  and the generated text's perplexity under a *fixed* reference model crashes
  **69 → 7** (monotonic): the model degenerates into low-entropy, repetitive text.
  That perplexity curve is the cleanest collapse signal in the whole project.
- **This time the per-document detector DOES respond** — score 16.7 → 27.1,
  contamination 0% → 9% — because neural degeneration is *locally* repetitive
  (unlike the trigram model, whose per-doc text stayed varied).
- **But only partially** — the detector score stays below the 50 threshold, so ~91%
  of collapsed docs still aren't flagged. The reliable, monotonic collapse signals
  remain distribution-level (diversity, reference perplexity), not the per-document
  AI-text score.

**Unified conclusion across both collapse experiments:** distribution-level metrics
catch collapse cleanly; per-document AI-text detection catches it weakly (neural) or
not at all (trigram). That is exactly why a *collapse monitor* should track corpus
diversity / reference perplexity rather than just run an AI-text classifier per doc.

## Expected outputs

- `experiment_results.json` — controlled accuracy vs contamination, per-generation
  separation, self-contamination catch rate, false-positive rate.
- `evaluation_results.json` — real-data AUC / precision / recall / F1 + ROC points
  + the per-signal ablation.
- `crossmodel_results.json` — per-generator AUC for the detector and the baseline.
- `lomo_results.json` / `lomo_gpt2_results.json` — leave-one-model-out AUC per held-out
  generator + feature importance (bigram and real-GPT-2 perplexity variants).
- `collapse_results.json` / `collapse_gpt2_results.json` — per-generation vocabulary,
  diversity, detector readings (trigram and neural distilGPT-2 collapse).
- `figure1..figure8 .png` — publication-style charts (after `visualizations.py`),
  including the real-data ROC curve, cross-model AUC bars, the LOMO comparison, and
  the neural-collapse curves.

## ⚠️ Limitations (read this before quoting any numbers)

There are now **two** evaluation tracks; don't confuse them.

- **`experiments.py` (controlled):** the "human"/"synthetic" texts are
  **synthetic stand-ins** built to differ on the six signals. The ~100% accuracy
  there is *expected by construction* and is **not** evidence about real text.
  Use it to understand the mechanics, not to quote performance.
- **`evaluate.py` (real):** uses the HC3 corpus (real human vs real ChatGPT).
  This is the honest measure — AUC ≈ 0.91, beating a perplexity baseline. Quote
  *these* numbers, not the controlled ones.

Still true / still out of scope:

- HC3 is one corpus and a relatively *easy* one (a single LLM, mostly Q&A). A
  serious result needs multiple LLMs, multiple domains, and an iterative-collapse
  experiment — see `NEXT_STEPS.md`.
- The detector still **cannot** audit released frontier models for their actual
  training-data contamination; you can't observe their training data.

Treat this repo as a **measurement framework + reproducible harness** with a
real first datapoint, not as a finished product.
