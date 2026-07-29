"""
fetch_multimodel.py -- build a MULTI-LLM corpus from RAID (liamdugan/raid).

Uses the HuggingFace datasets-server REST API, which returns JSON rows over
plain HTTP -- so no `datasets`/`pandas`/parquet needed, stdlib urllib only.

We pull, per generator model, real machine-generated text with attack='none'
(no adversarial perturbation), plus a pool of real human text. Offsets are
spread across each model's rows so we get domain diversity (RAID spans
abstracts, books, news, poetry, recipes, reddit, reviews, wiki). Each sample is
cleaned and truncated to ~250 words to control for length.

Generators: GPT-2, GPT-3, GPT-4, ChatGPT, Llama-chat, Mistral-chat, Cohere, MPT-chat.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA_DIR = os.path.join(HERE, "corpora")

BASE = "https://datasets-server.huggingface.co"
DS = "liamdugan/raid"
CONFIG, SPLIT = "raid", "train"

MODELS = ["gpt2", "gpt3", "gpt4", "chatgpt", "llama-chat", "mistral-chat", "cohere", "mpt-chat"]
PER_MODEL = 400
N_HUMAN = 3000
MIN_WORDS = 60
MAX_WORDS = 250


def _path(name):
    return os.path.join(CORPORA_DIR, "raid_%s.txt" % name)


def _clean(t):
    return re.sub(r"\s+", " ", t or "").strip()


def _trunc(t):
    return " ".join(_clean(t).split()[:MAX_WORDS])


def _filter(where, offset, length):
    params = {"dataset": DS, "config": CONFIG, "split": SPLIT,
              "where": where, "offset": str(offset), "length": str(length)}
    url = BASE + "/filter?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "habsburg/1.0"})
    for _ in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (500, 503):   # index loading / transient
                time.sleep(12)
                continue
            raise
    raise RuntimeError("filter failed after retries: " + where)


def _collect(where, target):
    total = _filter(where, 0, 1).get("num_rows_total", 0)
    if total <= 0:
        return []
    if total <= target:
        offsets = list(range(0, total, 100))
    else:
        n_chunks = min((target // 100) + 2, 40)
        step = max(100, (total - 100) // n_chunks)
        offsets = [min(total - 100, i * step) for i in range(n_chunks)]
    out, seen = [], set()
    for off in offsets:
        if len(out) >= target:
            break
        for row in _filter(where, off, 100).get("rows", []):
            g = row["row"].get("generation") or ""
            if len(_clean(g).split()) >= MIN_WORDS:
                key = g[:80]
                if key not in seen:
                    seen.add(key)
                    out.append(_trunc(g))
            if len(out) >= target:
                break
        time.sleep(0.3)
    return out[:target]


def fetch_all():
    os.makedirs(CORPORA_DIR, exist_ok=True)
    if not os.path.exists(_path("human")):
        print("fetching human ...")
        h = _collect("\"model\"='human' AND \"attack\"='none'", N_HUMAN)
        with open(_path("human"), "w", encoding="utf-8") as f:
            f.write("\n".join(h))
        print("  human: %d" % len(h))
    else:
        print("cached: human")
    for m in MODELS:
        if os.path.exists(_path(m)):
            print("cached: %s" % m)
            continue
        print("fetching %s ..." % m)
        s = _collect("\"model\"='%s' AND \"attack\"='none'" % m, PER_MODEL)
        with open(_path(m), "w", encoding="utf-8") as f:
            f.write("\n".join(s))
        print("  %s: %d" % (m, len(s)))


def load(name):
    with open(_path(name), "r", encoding="utf-8") as f:
        return [ln for ln in f.read().split("\n") if ln.strip()]


if __name__ == "__main__":
    fetch_all()
    print("\nLoaded counts:")
    print("  %-14s %d" % ("human", len(load("human"))))
    for m in MODELS:
        print("  %-14s %d" % (m, len(load(m))))
