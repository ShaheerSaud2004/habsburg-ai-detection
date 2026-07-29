"""
collapse_modern.py -- recursive collapse with a MODERN small LLM (default Qwen2.5-0.5B).

Generalizes collapse_gpt2.py to any HuggingFace causal LM, to test whether the
detection-vs-monitoring finding holds for a contemporary (2024) architecture and
not only distilGPT-2 (2019). Reduced config by default so it runs on commodity
MPS hardware. Writes collapse_modern_results.json incrementally.

    .venv-ml/bin/python collapse_modern.py --model Qwen/Qwen2.5-0.5B \
        --generations 3 --docs 50 --epochs 2 --max-new-tokens 64
    # smoke test:
    .venv-ml/bin/python collapse_modern.py --generations 1 --docs 6 --epochs 1 --max-new-tokens 24
"""

import argparse
import json
import math
import os
import random

from detector import SyntheticTextProbe, SYNTHETIC_THRESHOLD
from collapse_experiment import toks, diversity
import fetch_multimodel as fm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "collapse_modern_results.json")
SEED = 1234


def measure(corpus, det, ref_ppl):
    vocab, ttr, dtr = diversity(corpus)
    scores = [det.score_text(d)["synthetic_likelihood"] for d in corpus]
    flagged = sum(1 for s in scores if s >= SYNTHETIC_THRESHOLD)
    return {"vocab": vocab, "ttr": ttr, "trigram_diversity": dtr,
            "mean_synth_score": round(sum(scores) / len(scores), 1),
            "est_contamination": round(flagged / len(corpus), 3),
            "ref_perplexity": round(ref_ppl(corpus[:40]), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--docs", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--lora", action="store_true", help="LoRA fine-tuning (low memory)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    random.seed(SEED); torch.manual_seed(SEED)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| model:", args.model, "| gens:", args.generations,
          "docs:", args.docs, "epochs:", args.epochs, flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ref = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    @torch.no_grad()
    def ref_perplexity(texts):
        vals = []
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            vals.append(math.exp(min(20.0, ref(ids, labels=ids).loss.item())))
        return sum(vals) / len(vals) if vals else float("nan")

    def train(texts):
        model = AutoModelForCausalLM.from_pretrained(args.model)
        if args.lora:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model = model.to(device)
        model.train()
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
        enc = [tok(t, truncation=True, max_length=args.max_len)["input_ids"] for t in texts]
        enc = [e for e in enc if len(e) >= 2]
        for _ in range(args.epochs):
            random.shuffle(enc)
            for i in range(0, len(enc), args.batch_size):
                batch = enc[i:i + args.batch_size]
                m = max(len(x) for x in batch)
                ii = torch.full((len(batch), m), tok.pad_token_id, dtype=torch.long)
                at = torch.zeros((len(batch), m), dtype=torch.long)
                for j, x in enumerate(batch):
                    ii[j, :len(x)] = torch.tensor(x); at[j, :len(x)] = 1
                lab = ii.clone(); lab[at == 0] = -100
                out = model(input_ids=ii.to(device), attention_mask=at.to(device), labels=lab.to(device))
                out.loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        return model

    @torch.no_grad()
    def generate(model, n):
        outs = []
        start = torch.tensor([[tok.eos_token_id or tok.bos_token_id or 0]], device=device)
        bs = 10
        while len(outs) < n:
            k = min(bs, n - len(outs))
            gen = model.generate(start.repeat(k, 1), do_sample=True, max_new_tokens=args.max_new_tokens,
                                 top_p=0.95, temperature=1.0, pad_token_id=tok.pad_token_id)
            for row in gen:
                txt = tok.decode(row, skip_special_tokens=True).strip()
                if txt:
                    outs.append(txt)
        return outs[:n]

    det = SyntheticTextProbe()
    human = fm.load("human"); random.shuffle(human)
    corpus = human[:args.docs]

    rows = [dict(generation=0, note="human seed", **measure(corpus, det, ref_perplexity))]
    print("gen 0 (human): vocab=%d tri-div=%.3f detector=%.1f ref_ppl=%.1f"
          % (rows[0]["vocab"], rows[0]["trigram_diversity"], rows[0]["mean_synth_score"], rows[0]["ref_perplexity"]), flush=True)
    json.dump({"model": args.model, "generations": rows}, open(RESULTS_PATH, "w"), indent=2)

    for g in range(1, args.generations + 1):
        model = train(corpus)
        corpus = generate(model, args.docs)
        m = measure(corpus, det, ref_perplexity)
        m.update(generation=g, note="generated")
        rows.append(m)
        print("gen %d: vocab=%d tri-div=%.3f ttr=%.3f detector=%.1f ref_ppl=%.1f est_contam=%.0f%%"
              % (g, m["vocab"], m["trigram_diversity"], m["ttr"], m["mean_synth_score"],
                 m["ref_perplexity"], 100 * m["est_contamination"]), flush=True)
        json.dump({"model": args.model, "generations": rows}, open(RESULTS_PATH, "w"), indent=2)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    g0, gN = rows[0], rows[-1]
    summary = {"model": args.model, "generations": rows,
               "vocab_drop_pct": round(100 * (1 - gN["vocab"] / g0["vocab"]), 1) if g0["vocab"] else 0,
               "trigram_diversity_drop": round(g0["trigram_diversity"] - gN["trigram_diversity"], 3),
               "ref_perplexity_change": round(gN["ref_perplexity"] - g0["ref_perplexity"], 1),
               "detector_start": g0["mean_synth_score"], "detector_end": gN["mean_synth_score"]}
    json.dump(summary, open(RESULTS_PATH, "w"), indent=2)
    print("DONE ->", RESULTS_PATH, flush=True)


if __name__ == "__main__":
    main()
