# Quickstart (5 minutes)

Everything here runs on **Python 3.8+ with no pip install** (core is stdlib-only).

```bash
cd /Users/shaheersaud/habsburg-ai-detection

# 1. health check
python3 check_system.py

# 2. see it work (5 scenarios, offline)
python3 main.py

# 3. run the controlled experiments -> writes experiment_results.json
python3 experiments.py

# 4. real data (HC3): held-out ROC/AUC vs baselines + per-signal ablation.
#    Downloads ~70 MB once.
python3 evaluate.py

# 5. cross-model: one fixed detector vs 8 generators from RAID (GPT-2/3/4,
#    ChatGPT, Llama, Mistral, Cohere, MPT). Fetches via the HF datasets-server.
python3 crossmodel.py

# 6. the fair head-to-head: leave-one-model-out classifier (signals+perplexity),
#    trained on 7 generators, tested on the unseen 8th. ~40s.
python3 lomo.py

# 7. recursive model collapse (trigram LM retrained on its own output). Stdlib, fast.
python3 collapse_experiment.py
```

### Optional: real GPT-2 features (needs the torch venv)

```bash
python3.11 -m venv .venv-ml
.venv-ml/bin/python -m pip install torch transformers
.venv-ml/bin/python gpt2_perplexity.py      # caches real GPT-2 perplexity
python3 lomo.py --ppl gpt2                   # LOMO with the GPT-2 perplexity feature

.venv-ml/bin/python collapse_gpt2.py        # NEURAL recursive collapse (distilGPT-2, ~15-20 min)
```

`crossmodel.py` is the result that matters most: the detector's zero-shot AUC
ranges 0.57–0.83 across generators (mean ~0.72) and a trained perplexity baseline
beats it on RAID. `evaluate.py` (HC3, AUC ≈ 0.91) is a single-model snapshot;
`experiments.py` is a controlled demo whose ~100% accuracy is expected by
construction. See README for the honest interpretation.

### Optional: the figures

Only `visualizations.py` needs a third-party package.

```bash
pip install -r requirements.txt     # installs matplotlib (+ anthropic)
python3 visualizations.py           # writes figure1..figure4 PNGs
```

### Optional: live generation

`ANTHROPIC_API_KEY` is **only** needed for the optional live-generation path
(`data_generator.generate_via_api`). Nothing else uses it.

```bash
export ANTHROPIC_API_KEY="sk-..."   # optional, live mode only
```

### What you should see

- `main.py` → scenario 1 (clean human) flags ~0%; scenario 5 (collapsed gen-4)
  flags ~100%; the mixed scenarios land in between.
- `experiments.py` → a results table + `experiment_results.json`.

> Heads-up: the accuracy is high *by construction* — the human/synthetic texts
> are stand-ins built to differ on the signals. See the Limitations section in
> `README.md` and the validation plan in `NEXT_STEPS.md` before citing numbers.
