"""
collapse_gpt2.py -- NEURAL recursive model collapse with distilGPT-2.

The neural counterpart of collapse_experiment.py. Each generation, a FRESH
distilGPT-2 is fine-tuned on the previous generation's text, then samples a new
corpus; the next generation trains on THAT corpus -- Shumailov-style recursion
with a real transformer.

  gen 0 : human seed corpus (RAID human)
  gen g : fine-tune distilGPT-2 on corpus_{g-1}, then generate corpus_g

Each generation is measured with the SAME instruments as the trigram experiment
(vocabulary, trigram diversity, Habsburg detector score/contamination) plus the
generated text's perplexity under a FIXED reference distilGPT-2 (fluency).

Run with the ML venv (torch + transformers), uses Apple MPS if available:

    .venv-ml/bin/python collapse_gpt2.py                 # full run (slow; background it)
    .venv-ml/bin/python collapse_gpt2.py --generations 1 --docs 8 --epochs 1 \
        --max-new-tokens 24                              # quick smoke test

Writes collapse_gpt2_results.json incrementally (one row per generation).
"""

import argparse
import json
import math
import os
import random

# Reused stdlib instruments (importable by any python; they need no heavy deps).
from detector import SyntheticTextProbe
from collapse_experiment import diversity, toks
import fetch_multimodel as fm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "collapse_gpt2_results.json")
SEED = 1234
MODEL_NAME = "distilgpt2"


def measure(corpus, detector, ref_perplexity_fn):
    vocab, ttr, dtr = diversity(corpus)
    scores = [detector.score_text(d)["synthetic_likelihood"] for d in corpus]
    est = detector.detect_dataset(corpus)["estimated_contamination"]
    mean_len = sum(len(toks(d)) for d in corpus) / max(1, len(corpus))
    return {
        "vocab": vocab, "ttr": ttr, "trigram_diversity": dtr,
        "mean_synth_score": round(sum(scores) / len(scores), 1),
        "est_contamination": est,
        "mean_words": round(mean_len, 1),
        "ref_perplexity": round(ref_perplexity_fn(corpus[:60]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--docs", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| model:", MODEL_NAME, "| gens:", args.generations,
          "docs:", args.docs, "epochs:", args.epochs, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    # Fixed reference model (never trained) for fluency/perplexity of generated text.
    ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

    @torch.no_grad()
    def ref_perplexity(texts):
        vals = []
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            loss = ref(ids, labels=ids).loss.item()
            vals.append(math.exp(min(20.0, loss)))
        return sum(vals) / len(vals) if vals else float("nan")

    def train(texts):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        enc = [tok(t, truncation=True, max_length=128)["input_ids"] for t in texts]
        enc = [e for e in enc if len(e) >= 2]
        for _ in range(args.epochs):
            random.shuffle(enc)
            for i in range(0, len(enc), args.batch_size):
                batch = enc[i:i + args.batch_size]
                maxlen = max(len(x) for x in batch)
                input_ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
                attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
                for j, x in enumerate(batch):
                    input_ids[j, :len(x)] = torch.tensor(x)
                    attn[j, :len(x)] = 1
                labels = input_ids.clone()
                labels[attn == 0] = -100
                out = model(input_ids=input_ids.to(device),
                            attention_mask=attn.to(device), labels=labels.to(device))
                out.loss.backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        return model

    @torch.no_grad()
    def generate(model, n):
        outs = []
        start = torch.tensor([[tok.eos_token_id]], device=device)
        bs = 20
        while len(outs) < n:
            k = min(bs, n - len(outs))
            inp = start.repeat(k, 1)
            gen = model.generate(inp, do_sample=True, max_new_tokens=args.max_new_tokens,
                                 top_p=0.95, temperature=1.0, pad_token_id=tok.eos_token_id)
            for row in gen:
                txt = tok.decode(row, skip_special_tokens=True).strip()
                if txt:
                    outs.append(txt)
        return outs[:n]

    detector = SyntheticTextProbe()

    human = fm.load("human")
    random.shuffle(human)
    corpus = human[:args.docs]

    rows = []
    m0 = measure(corpus, detector, ref_perplexity)
    m0.update({"generation": 0, "note": "human seed"})
    rows.append(m0)
    print("gen 0 (human): vocab=%d tri-div=%.3f detector=%.1f ref_ppl=%.1f"
          % (m0["vocab"], m0["trigram_diversity"], m0["mean_synth_score"], m0["ref_perplexity"]), flush=True)
    json.dump({"model": MODEL_NAME, "generations": rows}, open(RESULTS_PATH, "w"), indent=2)

    for g in range(1, args.generations + 1):
        model = train(corpus)
        corpus = generate(model, args.docs)
        m = measure(corpus, detector, ref_perplexity)
        m.update({"generation": g, "note": "generated"})
        rows.append(m)
        print("gen %d: vocab=%d tri-div=%.3f ttr=%.3f detector=%.1f ref_ppl=%.1f est_contam=%.0f%%"
              % (g, m["vocab"], m["trigram_diversity"], m["ttr"], m["mean_synth_score"],
                 m["ref_perplexity"], 100 * m["est_contamination"]), flush=True)
        json.dump({"model": MODEL_NAME, "generations": rows}, open(RESULTS_PATH, "w"), indent=2)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    g0, gN = rows[0], rows[-1]
    summary = {
        "model": MODEL_NAME, "generations": rows,
        "vocab_drop_pct": round(100 * (1 - gN["vocab"] / g0["vocab"]), 1) if g0["vocab"] else 0,
        "trigram_diversity_drop": round(g0["trigram_diversity"] - gN["trigram_diversity"], 3),
        "ref_perplexity_change": round(gN["ref_perplexity"] - g0["ref_perplexity"], 1),
    }
    json.dump(summary, open(RESULTS_PATH, "w"), indent=2)
    print("DONE. vocab -%.0f%%, trigram-diversity -%.3f, ref-perplexity change %+.1f -> %s"
          % (summary["vocab_drop_pct"], summary["trigram_diversity_drop"],
             summary["ref_perplexity_change"], RESULTS_PATH), flush=True)


if __name__ == "__main__":
    main()
