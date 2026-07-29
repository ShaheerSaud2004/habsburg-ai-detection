"""
fetch_corpora.py -- download and cache a REAL human-vs-LLM text corpus.

Source: HC3 (Human ChatGPT Comparison Corpus, Hello-SimpleAI/HC3). Each entry
is a question with one or more real human answers and one or more real ChatGPT
answers, so the two classes answer the SAME questions -- topic is controlled.

We download once, cache to corpora/, and write two plain-text files
(one sample per line). To reduce length as a trivial giveaway, every sample is
truncated to its first ~250 words.

Standard library only (urllib).
"""

import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA_DIR = os.path.join(HERE, "corpora")
RAW_PATH = os.path.join(CORPORA_DIR, "HC3_all.jsonl")
HUMAN_PATH = os.path.join(CORPORA_DIR, "hc3_human.txt")
LLM_PATH = os.path.join(CORPORA_DIR, "hc3_llm.txt")

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"

SEED = 1234
PER_CLASS = 2500          # balanced samples per class
MIN_WORDS = 60            # require some substance before truncation
MAX_WORDS = 250           # truncate to control for length


def _words(text):
    return text.split()


def _clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text):
    return " ".join(_words(text)[:MAX_WORDS])


def download_raw():
    import urllib.request
    os.makedirs(CORPORA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH) and os.path.getsize(RAW_PATH) > 1_000_000:
        print("Using cached HC3:", RAW_PATH)
        return
    print("Downloading HC3 (~70 MB) ...")
    req = urllib.request.Request(HC3_URL, headers={"User-Agent": "habsburg-detection/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(RAW_PATH, "wb") as f:
        chunk = 1 << 20
        done = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            print("  %.1f MB" % (done / 1e6), end="\r")
    print("\nSaved", RAW_PATH)


def build_text_files():
    """Parse the raw JSONL into balanced, length-controlled human/llm text files."""
    human, llm = [], []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for a in obj.get("human_answers", []) or []:
                a = _clean(a)
                if len(_words(a)) >= MIN_WORDS:
                    human.append(_truncate(a))
            for a in obj.get("chatgpt_answers", []) or []:
                a = _clean(a)
                if len(_words(a)) >= MIN_WORDS:
                    llm.append(_truncate(a))

    rng = random.Random(SEED)
    rng.shuffle(human)
    rng.shuffle(llm)
    n = min(PER_CLASS, len(human), len(llm))
    human, llm = human[:n], llm[:n]

    with open(HUMAN_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(human))
    with open(LLM_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(llm))

    print("Wrote %d human samples -> %s" % (len(human), HUMAN_PATH))
    print("Wrote %d llm   samples -> %s" % (len(llm), LLM_PATH))
    return len(human), len(llm)


def load_samples():
    """Return (human_list, llm_list); downloads + builds on first call."""
    if not (os.path.exists(HUMAN_PATH) and os.path.exists(LLM_PATH)):
        download_raw()
        build_text_files()
    with open(HUMAN_PATH, "r", encoding="utf-8") as f:
        human = [ln for ln in f.read().split("\n") if ln.strip()]
    with open(LLM_PATH, "r", encoding="utf-8") as f:
        llm = [ln for ln in f.read().split("\n") if ln.strip()]
    return human, llm


if __name__ == "__main__":
    download_raw()
    build_text_files()
    h, l = load_samples()
    print("Loaded:", len(h), "human /", len(l), "llm")
    print("\nExample HUMAN:\n ", h[0][:240])
    print("\nExample LLM:\n ", l[0][:240])
