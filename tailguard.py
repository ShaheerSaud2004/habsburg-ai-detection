#!/usr/bin/env python3
"""
TailGuard -- corpus-health monitoring and a contamination tripwire for LLM
training data. ("The tails go first.")

Born from the paper "Contamination Detection Is Not Collapse Monitoring":
per-document AI-text detection misses model collapse, but distribution-level
metrics (vocabulary, n-gram diversity, duplication, score distributions) catch
it cleanly. TailGuard computes those metrics over a corpus snapshot, compares
snapshots, and fails CI when a corpus is degrading.

Zero dependencies -- Python standard library only.

USAGE
  python3 tailguard.py scan <path> [-o baseline.json] [--html report.html]
  python3 tailguard.py compare <baseline.json> <path> [--html report.html]
  python3 tailguard.py --help

  <path> may be: a .txt file (one document per line), a .jsonl file (uses the
  first of these fields: text, content, body, output, response), or a directory
  of .txt/.md files (one document per file).

CI: `compare` exits 0 (healthy), 1 (warnings), 2 (critical) -- wire it into any
pipeline as a tripwire before a training run.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter

from detector import SyntheticTextProbe, SYNTHETIC_THRESHOLD

VERSION = "0.1.0"
_WORD = re.compile(r"[a-zA-Z']+")
_SENT = re.compile(r"[.!?]+")

# drift thresholds: (metric, warn_drop_pct, critical_drop_pct, direction)
# direction -1 means "a DROP is bad", +1 means "a RISE is bad"
THRESHOLDS = [
    ("trigram_diversity", 5.0, 15.0, -1),
    ("bigram_diversity", 5.0, 15.0, -1),
    ("vocab_per_10k_tokens", 8.0, 20.0, -1),
    ("type_token_ratio", 8.0, 20.0, -1),
    ("mean_burstiness", 10.0, 25.0, -1),
    ("near_duplicate_rate", 50.0, 150.0, +1),
    ("mean_synthetic_score", 25.0, 60.0, +1),
    ("flagged_synthetic_pct", 50.0, 150.0, +1),
]


# ---------------- loading ----------------

def load_corpus(path):
    docs = []
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.endswith((".txt", ".md")):
                with open(os.path.join(path, fn), encoding="utf-8", errors="replace") as f:
                    t = f.read().strip()
                if t:
                    docs.append(t)
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                for k in ("text", "content", "body", "output", "response"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        docs.append(v.strip())
                        break
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            docs = [ln.strip() for ln in f if ln.strip()]
    return docs


# ---------------- metrics ----------------

def _toks(t):
    return _WORD.findall(t.lower())


def _burstiness(t):
    lens = [len(_toks(s)) for s in _SENT.split(t) if s.strip()]
    lens = [l for l in lens if l > 0]
    if len(lens) < 2:
        return None
    m = statistics.mean(lens)
    return (statistics.pstdev(lens) / m) if m else None


def _shingles(toks, k=5):
    if len(toks) < k:
        return {hashlib.md5(" ".join(toks).encode()).hexdigest()[:12]} if toks else set()
    return {hashlib.md5(" ".join(toks[i:i + k]).encode()).hexdigest()[:12]
            for i in range(len(toks) - k + 1)}


def near_duplicate_rate(docs, sample=400, seed=1234):
    """Fraction of sampled pairs with shingle-Jaccard > 0.5."""
    rng = random.Random(seed)
    pool = docs if len(docs) <= sample else rng.sample(docs, sample)
    sh = [_shingles(_toks(d)) for d in pool]
    pairs = 0
    dups = 0
    n = len(sh)
    max_pairs = 4000
    for _ in range(min(max_pairs, n * (n - 1) // 2)):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        pairs += 1
        a, b = sh[i], sh[j]
        if a and b:
            inter = len(a & b)
            if inter and inter / len(a | b) > 0.5:
                dups += 1
    return (dups / pairs) if pairs else 0.0


def scan_corpus(docs, probe=None, label="corpus"):
    probe = probe or SyntheticTextProbe()
    all_tokens = []
    bursts = []
    scores = []
    for d in docs:
        tk = _toks(d)
        all_tokens.extend(tk)
        b = _burstiness(d)
        if b is not None:
            bursts.append(b)
        scores.append(probe.score_text(d)["synthetic_likelihood"])

    n_tok = len(all_tokens)
    vocab = len(set(all_tokens))
    bi = list(zip(all_tokens, all_tokens[1:]))
    tri = list(zip(all_tokens, all_tokens[1:], all_tokens[2:]))
    grams5 = Counter(tuple(all_tokens[i:i + 5]) for i in range(max(0, n_tok - 4)))
    boiler = [" ".join(g) for g, c in grams5.most_common(8) if c >= max(3, n_tok // 20000)]

    flagged = sum(1 for s in scores if s >= SYNTHETIC_THRESHOLD)
    hist = [0] * 10
    for s in scores:
        hist[min(9, int(s // 10))] += 1

    return {
        "tailguard_version": VERSION,
        "label": label,
        "n_docs": len(docs),
        "n_tokens": n_tok,
        "mean_doc_tokens": round(n_tok / len(docs), 1) if docs else 0,
        "vocab": vocab,
        "vocab_per_10k_tokens": round(10000 * vocab / n_tok, 1) if n_tok else 0,
        "type_token_ratio": round(vocab / n_tok, 4) if n_tok else 0,
        "bigram_diversity": round(len(set(bi)) / len(bi), 4) if bi else 0,
        "trigram_diversity": round(len(set(tri)) / len(tri), 4) if tri else 0,
        "mean_burstiness": round(statistics.mean(bursts), 3) if bursts else 0,
        "near_duplicate_rate": round(near_duplicate_rate(docs), 4),
        "mean_synthetic_score": round(statistics.mean(scores), 1) if scores else 0,
        "flagged_synthetic_pct": round(100 * flagged / len(docs), 1) if docs else 0,
        "score_histogram": hist,
        "top_repeated_5grams": boiler,
    }


# ---------------- compare ----------------

def compare(baseline, current):
    alerts = []
    for metric, warn, crit, direction in THRESHOLDS:
        b, c = baseline.get(metric), current.get(metric)
        if b in (None, 0) or c is None:
            continue
        change_pct = 100.0 * (c - b) / abs(b)
        bad = change_pct * direction  # positive = bad
        level = "ok"
        if bad >= crit:
            level = "critical"
        elif bad >= warn:
            level = "warning"
        alerts.append({
            "metric": metric, "baseline": b, "current": c,
            "change_pct": round(change_pct, 1), "level": level,
        })
    worst = "ok"
    if any(a["level"] == "critical" for a in alerts):
        worst = "critical"
    elif any(a["level"] == "warning" for a in alerts):
        worst = "warning"
    return {"status": worst, "alerts": alerts}


# ---------------- report ----------------

def _bar(pct, color):
    pct = max(0.0, min(100.0, pct))
    return ('<div style="background:#101319;border-radius:6px;height:10px;overflow:hidden">'
            '<div style="width:%.1f%%;height:100%%;background:%s;border-radius:6px"></div></div>' % (pct, color))


def html_report(current, baseline=None, cmp_result=None, out="tailguard_report.html"):
    gold, green, red, muted = "#f3b24a", "#46d39a", "#ff6b78", "#8b93a7"
    rows = []
    show = [("Documents", "n_docs", ""), ("Tokens", "n_tokens", ""),
            ("Vocabulary", "vocab", ""), ("Vocab / 10k tokens", "vocab_per_10k_tokens", ""),
            ("Type-token ratio", "type_token_ratio", ""), ("Bigram diversity", "bigram_diversity", ""),
            ("Trigram diversity", "trigram_diversity", ""), ("Mean burstiness", "mean_burstiness", ""),
            ("Near-duplicate rate", "near_duplicate_rate", ""), ("Mean synthetic score", "mean_synthetic_score", "/100"),
            ("Flagged synthetic", "flagged_synthetic_pct", "%")]
    amap = {a["metric"]: a for a in (cmp_result or {}).get("alerts", [])}
    for name, key, suffix in show:
        cur = current.get(key)
        b = baseline.get(key) if baseline else None
        a = amap.get(key)
        delta = ""
        if a:
            col = {"ok": green, "warning": gold, "critical": red}[a["level"]]
            delta = '<td style="color:%s;font-weight:600">%+.1f%%&nbsp;%s</td>' % (
                col, a["change_pct"], "&#9888;" if a["level"] != "ok" else "&#10003;")
        elif baseline is not None:
            delta = '<td style="color:%s">—</td>' % muted
        rows.append("<tr><td>%s</td>%s<td><b>%s%s</b></td>%s</tr>" % (
            name, ("<td>%s</td>" % (b if b is not None else "—")) if baseline is not None else "",
            cur, suffix, delta))

    hist = current.get("score_histogram", [0] * 10)
    mx = max(hist) or 1
    hbars = "".join('<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                    '<span style="width:64px;color:%s;font-size:12px">%d–%d</span>%s'
                    '<span style="color:%s;font-size:12px">%d</span></div>'
                    % (muted, i * 10, i * 10 + 9, _bar(100 * v / mx, gold if i < 5 else red), muted, v)
                    for i, v in enumerate(hist))

    status = (cmp_result or {}).get("status", "scan")
    badge = {"ok": (green, "HEALTHY"), "warning": (gold, "WARNINGS"),
             "critical": (red, "CRITICAL"), "scan": (gold, "SNAPSHOT")}[status]
    boiler = current.get("top_repeated_5grams") or []
    boiler_html = ("".join("<li><code>%s</code></li>" % b for b in boiler)) or "<li style='color:%s'>none detected</li>" % muted

    head_cols = "<th>Metric</th>" + ("<th>Baseline</th>" if baseline is not None else "") + "<th>Current</th>" + ("<th>Δ</th>" if baseline is not None else "")
    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>TailGuard report — %s</title>
<style>body{background:#0a0b0f;color:#f3f5f9;font-family:-apple-system,Inter,sans-serif;max-width:860px;margin:40px auto;padding:0 20px;line-height:1.6}
h1{font-size:1.6rem}code{background:#181b25;padding:2px 7px;border-radius:5px;font-size:.85em}
table{width:100%%;border-collapse:collapse;margin:18px 0;font-size:.95rem}
td,th{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left}
th{color:#8b93a7;font-weight:600}.badge{display:inline-block;padding:5px 16px;border-radius:999px;font-weight:700;background:%s22;color:%s}
.sec{margin-top:30px}.foot{color:#5b6373;font-size:.82rem;margin-top:40px}</style></head><body>
<h1>&#128737;&#65039; TailGuard corpus report</h1>
<p><span class="badge">%s</span>&nbsp;&nbsp;<span style="color:#8b93a7">%s · TailGuard v%s</span></p>
<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>
<div class="sec"><h3>Synthetic-score distribution</h3>%s</div>
<div class="sec"><h3>Most-repeated 5-grams (boilerplate check)</h3><ul>%s</ul></div>
<div class="foot">Distribution-level corpus health, per "Contamination Detection Is Not Collapse Monitoring" —
per-document detection misses collapse; these metrics catch it. habsburg-ai.vercel.app</div>
</body></html>""" % (current.get("label", ""), badge[0], badge[0], badge[1],
                     current.get("label", ""), VERSION, head_cols, "".join(rows), hbars, boiler_html)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(prog="tailguard", description="Corpus-health monitoring for LLM training data.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a corpus -> health snapshot JSON")
    s.add_argument("path"); s.add_argument("-o", "--out", default="tailguard_baseline.json")
    s.add_argument("--html"); s.add_argument("--label", default=None)
    c = sub.add_parser("compare", help="compare a corpus against a baseline snapshot (CI tripwire)")
    c.add_argument("baseline"); c.add_argument("path")
    c.add_argument("--html"); c.add_argument("--label", default=None)
    args = ap.parse_args()

    docs = load_corpus(args.path)
    if not docs:
        print("tailguard: no documents found at %s" % args.path); sys.exit(2)
    label = args.label or os.path.basename(args.path.rstrip("/"))
    snap = scan_corpus(docs, label=label)

    if args.cmd == "scan":
        with open(args.out, "w") as f:
            json.dump(snap, f, indent=2)
        print("TailGuard snapshot of %r  (%d docs, %d tokens)" % (label, snap["n_docs"], snap["n_tokens"]))
        for k in ("vocab", "trigram_diversity", "mean_burstiness", "near_duplicate_rate",
                  "mean_synthetic_score", "flagged_synthetic_pct"):
            print("  %-24s %s" % (k, snap[k]))
        print("wrote %s" % args.out)
        if args.html:
            print("wrote %s" % html_report(snap, out=args.html))
        sys.exit(0)

    with open(args.baseline) as f:
        base = json.load(f)
    result = compare(base, snap)
    print("TailGuard compare: %r vs baseline %r  ->  %s" % (label, base.get("label"), result["status"].upper()))
    for a in result["alerts"]:
        mark = {"ok": " ", "warning": "!", "critical": "X"}[a["level"]]
        print("  [%s] %-24s %s -> %s  (%+.1f%%)" % (mark, a["metric"], a["baseline"], a["current"], a["change_pct"]))
    if args.html:
        print("wrote %s" % html_report(snap, base, result, out=args.html))
    sys.exit({"ok": 0, "warning": 1, "critical": 2}[result["status"]])


if __name__ == "__main__":
    main()
