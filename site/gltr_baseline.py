"""
gltr_baseline.py -- a GLTR-style likelihood detector under GPT-2, and a zero-shot
comparison of established statistical detectors against SyntheticTextProbe.

GLTR [Gehrmann2019] detects machine text from two per-token statistics under a
reference LM: (1) the mean log-probability of the actual tokens, and (2) the
fraction of tokens that fall in the model's top-k most-likely set (the "green"
fraction). Machine text tends to have higher log-probability and a larger
top-k fraction. We compute both under GPT-2 (124M) and report each as a zero-shot
detector, alongside SyntheticTextProbe, on HC3 and pooled RAID.

(DetectGPT's perturbation-curvature test and the commercial GPTZero are *not* run:
the former needs many mask-fill perturbations and white-box access to the scoring
model, the latter a paid API. Our GPT-2 log-probability detector is the same
likelihood family and serves as the runnable representative.)

Run with the ML venv (torch + transformers). Writes corpora/gltr.tsv (cache) and
detector_comparison.json.

    .venv-ml/bin/python gltr_baseline.py
"""

import hashlib
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = os.path.join(HERE, "corpora")
CACHE = os.path.join(CORPORA, "gltr.tsv")
RESULTS = os.path.join(HERE, "detector_comparison.json")
MAX_TOKENS = 192
TOPK = 10
MODEL_NAME = "gpt2"

CORPUS_FILES = [
    "hc3_human.txt", "hc3_llm.txt", "raid_human.txt",
    "raid_gpt2.txt", "raid_gpt3.txt", "raid_gpt4.txt", "raid_chatgpt.txt",
    "raid_llama-chat.txt", "raid_mistral-chat.txt", "raid_cohere.txt", "raid_mpt-chat.txt",
]


def _h(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def load_lines(fn):
    p = os.path.join(CORPORA, fn)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [ln for ln in f.read().split("\n") if ln.strip()]


def all_texts():
    seen, out = set(), []
    for fn in CORPUS_FILES:
        for ln in load_lines(fn):
            h = _h(ln)
            if h not in seen:
                seen.add(h); out.append(ln)
    return out


def load_cache():
    d = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) == 3:
                    try:
                        d[p[0]] = (float(p[1]), float(p[2]))
                    except ValueError:
                        pass
    return d


def compute_features():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| computing GLTR features under", MODEL_NAME, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

    texts = all_texts()
    done = load_cache()
    todo = [t for t in texts if _h(t) not in done]
    print("texts: %d | cached: %d | to compute: %d" % (len(texts), len(done), len(todo)), flush=True)

    n, t0 = 0, time.time()
    with open(CACHE, "a", encoding="utf-8") as out, torch.no_grad():
        for t in todo:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids.to(device)
            if ids.shape[1] < 2:
                mean_logp, frac_top = float("nan"), float("nan")
            else:
                logits = model(ids).logits[:, :-1, :]          # predict tokens 1..T-1
                target = ids[:, 1:]                              # [1, T-1]
                logp = torch.log_softmax(logits, dim=-1)
                tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                mean_logp = tok_logp.mean().item()
                tgt_logit = logits.gather(-1, target.unsqueeze(-1))     # [1,T-1,1]
                rank = (logits >= tgt_logit).sum(dim=-1)               # 1-based rank
                frac_top = (rank <= TOPK).float().mean().item()
            out.write("%s\t%.5f\t%.5f\n" % (_h(t), mean_logp, frac_top)); out.flush()
            n += 1
            if n % 200 == 0:
                print("  %d/%d (%.1f/s)" % (n, len(todo), n / max(1e-6, time.time() - t0)), flush=True)
    print("DONE features. computed %d new." % n, flush=True)


def roc_auc(scores, labels):
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [k for k in range(n) if labels[k] == 1]
    npos, nneg = len(pos), n - len(pos)
    if not npos or not nneg:
        return 0.5
    return (sum(ranks[k] for k in pos) - npos * (npos + 1) / 2.0) / (npos * nneg)


def directed_auc(scores, labels):
    a = roc_auc(scores, labels)
    return a if a >= 0.5 else 1.0 - a


def evaluate():
    from detector import SyntheticTextProbe, combine_signals
    cache = load_cache()
    det = SyntheticTextProbe()

    def feats(texts):
        lp, ft, hb = [], [], []
        for t in texts:
            v = cache.get(_h(t))
            if v and not math.isnan(v[0]):
                lp.append(v[0]); ft.append(v[1])
            else:
                lp.append(None); ft.append(None)
            hb.append(combine_signals(det.raw_signals(t)))
        return lp, ft, hb

    def auc_set(human, machine):
        h_lp, h_ft, h_hb = feats(human)
        m_lp, m_ft, m_hb = feats(machine)
        labels = [0] * len(human) + [1] * len(machine)
        def col(hcol, mcol):
            xs = hcol + mcol
            pair = [(x, y) for x, y in zip(xs, labels) if x is not None]
            return directed_auc([p[0] for p in pair], [p[1] for p in pair])
        return {
            "n_human": len(human), "n_machine": len(machine),
            "SyntheticTextProbe_zeroshot": round(col(h_hb, m_hb), 3),
            "GPT2_logprob_GLTR": round(col(h_lp, m_lp), 3),
            "GPT2_top%d_rank_GLTR" % TOPK: round(col(h_ft, m_ft), 3),
        }

    hc3 = auc_set(load_lines("hc3_human.txt"), load_lines("hc3_llm.txt"))
    raid_machine = []
    for m in ["raid_gpt2", "raid_gpt3", "raid_gpt4", "raid_chatgpt",
              "raid_llama-chat", "raid_mistral-chat", "raid_cohere", "raid_mpt-chat"]:
        raid_machine += load_lines(m + ".txt")
    raid = auc_set(load_lines("raid_human.txt"), raid_machine)

    out = {
        "note": "Zero-shot AUC (no training). GLTR statistics computed under GPT-2 (124M). "
                "DetectGPT-curvature and GPTZero not run (need white-box perturbations / paid API).",
        "HC3_human_vs_chatgpt": hc3,
        "RAID_human_vs_all8generators_pooled": raid,
        "reference_trained_classifier_RAID_LOMO": 0.915,
    }
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 64)
    print("ZERO-SHOT DETECTOR COMPARISON (AUC)")
    print("=" * 64)
    for name, d in (("HC3 (human vs ChatGPT)", hc3),
                    ("RAID (human vs 8 generators, pooled)", raid)):
        print("\n %s   [n=%d/%d]" % (name, d["n_human"], d["n_machine"]))
        for k in ("SyntheticTextProbe_zeroshot", "GPT2_logprob_GLTR", "GPT2_top%d_rank_GLTR" % TOPK):
            print("   %-28s %.3f" % (k, d[k]))
    print("\n (reference) trained LOMO classifier on RAID: 0.915")
    print("Wrote", RESULTS)


if __name__ == "__main__":
    compute_features()
    evaluate()
