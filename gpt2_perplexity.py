"""
gpt2_perplexity.py -- compute REAL GPT-2 perplexity for every corpus text.

This is the ONLY script that needs heavy deps. Run it with the ML venv, not the
stdlib python:

    .venv-ml/bin/python gpt2_perplexity.py

It caches results to corpora/gpt2_ppl.tsv as  md5(text)<TAB>perplexity  and is
resumable (skips already-cached texts). lomo.py (stdlib) then reads this cache to
use GPT-2 perplexity as a feature -- so the core project stays zero-install.

Uses Apple MPS (GPU) if available, else CUDA, else CPU.
"""

import hashlib
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = os.path.join(HERE, "corpora")
CACHE = os.path.join(CORPORA, "gpt2_ppl.tsv")
MAX_TOKENS = 256
MODEL_NAME = "gpt2"

CORPUS_FILES = [
    "hc3_human.txt", "hc3_llm.txt", "raid_human.txt",
    "raid_gpt2.txt", "raid_gpt3.txt", "raid_gpt4.txt", "raid_chatgpt.txt",
    "raid_llama-chat.txt", "raid_mistral-chat.txt", "raid_cohere.txt", "raid_mpt-chat.txt",
]


def _h(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def load_texts():
    seen, texts = set(), []
    for fn in CORPUS_FILES:
        p = os.path.join(CORPORA, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    h = _h(ln)
                    if h not in seen:
                        seen.add(h)
                        texts.append(ln)
    return texts


def load_cache():
    done = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) == 2:
                    done[parts[0]] = parts[1]
    return done


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| model:", MODEL_NAME)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

    texts = load_texts()
    done = load_cache()
    todo = [t for t in texts if _h(t) not in done]
    print("texts: %d | cached: %d | to compute: %d" % (len(texts), len(done), len(todo)))

    n, t0 = 0, time.time()
    with open(CACHE, "a", encoding="utf-8") as out, torch.no_grad():
        for t in todo:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids.to(device)
            if ids.shape[1] < 2:
                ppl = float("nan")
            else:
                loss = model(ids, labels=ids).loss.item()
                ppl = math.exp(min(20.0, loss))
            out.write("%s\t%.4f\n" % (_h(t), ppl))
            out.flush()
            n += 1
            if n % 200 == 0:
                rate = n / max(1e-6, time.time() - t0)
                print("  %d/%d  (%.1f/s)" % (n, len(todo), rate), flush=True)

    print("DONE. computed %d new; cache now has %d entries -> %s" % (n, len(done) + n, CACHE))


if __name__ == "__main__":
    main()
