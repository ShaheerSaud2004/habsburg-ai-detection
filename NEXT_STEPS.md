# Next steps — research & publication roadmap (honest version)

This prototype is a reproducible harness. To turn it into something publishable,
the work is mostly about **real data and honest evaluation**, not more code.

## What is already known (cite as prior work — do NOT claim to have discovered it)

- **Shumailov et al., "The Curse of Recursion / AI models collapse when trained on
  recursively generated data"** (2023; *Nature*, 2024). The canonical model-collapse
  result. Your work builds on this; it does not discover it.
- Follow-on work on data filtering, "accumulate vs replace" training data, and
  synthetic-data mixing ratios (2024–2025).
- Provenance / licensing efforts (e.g. RSL, publisher data deals) as a *mitigation*
  on the supply side.

If you submit anything that implies you found model collapse, it will be desk-rejected.
Your contribution is **detection + risk-quantification as a measurable pipeline**.

## What is genuinely novel here (the defensible wedge)

1. A lightweight, interpretable **contamination-estimation** method (six signals →
   dataset-level synthetic ratio), framed as something a training pipeline can run
   at ingestion time.
2. A **recycling detector** (`duplicate_similarity` vs a corpus of known model
   outputs) aimed at the specific failure where a lab re-ingests its *own* output.
3. A **reproducible benchmark harness** with controllable contamination and
   "generation" knobs.

## Progress so far

`evaluate.py` (HC3, real human vs real ChatGPT, held out):

| Method | AUC | F1 |
|---|---|---|
| SyntheticTextProbe | **0.912** | 0.850 |
| Perplexity baseline (bigram LM) | 0.832 | 0.772 |
| Mean-word-length floor | 0.646 | 0.679 |

`crossmodel.py` (RAID, one fixed detector vs 8 generators) — **the result that
reframes the project**:

| Generator | Habsburg (zero-shot) | Perplexity (trained) |
|---|---|---|
| ChatGPT | 0.829 | 0.968 |
| Llama-chat | 0.794 | 0.934 |
| GPT-4 | 0.703 | 0.897 |
| Cohere | 0.610 | 0.845 |
| GPT-2 | 0.566 | 0.819 |
| **Pooled (8 models)** | **0.723** | **0.908** |

**Honest headline:** the heuristic signals transfer *unevenly* (0.57–0.83, best on
RLHF-chat models) and a trained perplexity baseline is more robust across
generators. The HC3 "we beat the baseline" story does **not** survive multi-model
evaluation. That is a real, honest finding — and a better paper than "our detector
wins," because it characterizes *when* cheap heuristics work and when they don't.

The ablation (HC3) shows **burstiness** and **lexical_diversity** carry most of the
weight; **duplicate_similarity** is inert without a reference corpus.

## What REMAINS before submitting

1. ✅ Real human + real LLM corpus (HC3) — done.
2. ✅ Held-out eval + ROC/AUC + baselines — done.
3. ✅ More LLMs (8 generators, RAID) + cross-model table — done.
4. ✅ Per-signal ablation — done.
5. ✅ **Fair head-to-head — leave-one-model-out** (`lomo.py`). A logistic
   regression over [6 signals + perplexity], trained on 7 generators and tested on
   the unseen 8th, reaches **mean AUC 0.924**, beating zero-shot heuristics (0.732)
   and perplexity alone (0.911). Feature importance: perplexity ≫ lexical_diversity
   > repetition > burstiness > filler ≈ ngram; duplicate_similarity ≈ 0. **Done.**
6. ◐ **Stronger perplexity** — `gpt2_perplexity.py` adds a **real GPT-2** perplexity
   feature (`python3 lomo.py --ppl gpt2`). Surprising honest result: it did NOT beat
   the in-distribution bigram LM (LOMO mean 0.895 vs 0.924), because GPT-2 perplexity
   is fixed/zero-shot and weak on newer models (GPT-4/MPT/Cohere). Worth trying next:
   an instruction-tuned or larger LM's perplexity, and a fairer setup that fine-tunes
   the reference LM on the train fold. Still TODO: more domains/code/fiction + an open
   neural detector baseline. **Partly done.**
7. ✅ **A real "generations" experiment** — both `collapse_experiment.py` (trigram)
   and `collapse_gpt2.py` (neural distilGPT-2 fine-tuned on its own output, MPS)
   reproduce collapse. Trigram: detector stays flat. Neural: detector rises
   16.7→27.1 (still sub-threshold, 9% flagged) and generated-text perplexity under a
   fixed model crashes 69→7 — the cleanest collapse signal. Conclusion:
   distribution-level metrics catch collapse; per-document detection catches it
   weakly/not at all. **Done.**
8. ⬜ **Stat rigor for the writeup** — repeat each fold over several seeds and report
   mean ± std / confidence intervals, not single-run point estimates.
9. ⬜ **Distribution-level collapse monitor** — formalize the corpus-diversity signal
   (vocab / distinct-n-gram ratio / embedding spread) as the actual "collapse
   detector," distinct from per-document AI-text detection.

## Reframed contribution (what the paper actually argues)

Not "we built a great AI detector." Instead, a claim the experiments now support:

> *An interpretable, lightweight feature set — six text statistics plus a small
> perplexity signal — combines into a classifier that generalizes to **unseen**
> generators (mean leave-one-model-out AUC 0.92 across 8 LLMs), beating either the
> raw heuristics (0.73) or perplexity alone (0.91). Perplexity is the dominant
> feature; the cheap statistics add a consistent lift. The same statistics, used
> zero-shot, degrade predictably across generators (collapsing toward chance on
> base models like GPT-2). We quantify each feature's contribution and the
> recycling-only role of duplicate-similarity.*

That is defensible, reviewer-proof, and honest about where the signal comes from —
in a way the original hype-version was not. The remaining items (6–8) are what turn
this from a strong workshop paper into a main-track one.

## Where to submit (realistic ordering)

1. **arXiv preprint first** — immediate, citable, no gatekeeping. Do this once the
   real-data evaluation exists.
2. **Workshops** — NeurIPS / ICML / ICLR workshops on data-centric ML, synthetic
   data, or trustworthy ML. Faster, friendlier review; the right home for a first paper.
3. **Main track** — only after workshop feedback and a stronger eval. Data-centric ML
   venues (DMLR, the NeurIPS Datasets & Benchmarks track) fit better than a general
   main track for this kind of contribution.

## Honest scope check

- ✅ In scope: contamination estimation, recycling detection, a reproducible benchmark,
  a real small-scale iterative-collapse experiment.
- ❌ Out of scope (do not claim): auditing GPT-4 / Claude / Gemini for their actual
  training-data contamination. You cannot observe their training data; any such claim
  is unfalsifiable and reviewers will reject it.

## Suggested order of operations

1. Swap the synthetic stand-ins for real corpora (biggest credibility jump).
2. Add baselines + held-out eval + ROC/AUC.
3. Run the small-model iterative-collapse experiment.
4. Write it up, post to arXiv, submit to a workshop.
