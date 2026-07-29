// /api/ask — "Ask this research" backend.
// Streams answers from an OpenAI-compatible provider (Groq primary, AI-Loop fallback).
// API keys live ONLY in Vercel env vars; the frontend never sees them.

export const config = { maxDuration: 60 };

const FACTS = `You are the research assistant for the paper "Per-Document AI-Text Detection Is Not a Reliable Model-Collapse Monitor" by Shaheer Saud (independent researcher). Site: habsburg-ai.vercel.app. Answer questions about this research only, grounded strictly in these facts:

CENTRAL CLAIM: AI-text detection (judging ONE document) and model-collapse monitoring (judging a corpus DISTRIBUTION) are different objectives. A detector can correctly flag synthetic documents while completely missing distributional degeneration.

KEY RESULTS (all reproducible, seed 1234):
- A 6-signal detector (SyntheticTextProbe: repetition, filler density, lexical diversity, n-gram diversity, burstiness, duplicate similarity) + one perplexity feature, in a leave-one-model-out logistic classifier, reaches mean held-out AUC 0.915 +/- 0.006 across 8 unseen LLMs on RAID (10 seeds; 95% CI [0.911,0.920]; beats perplexity alone 0.903, paired permutation p=0.002, wins 10/10 seeds).
- Zero-shot, the heuristic transfers unevenly: 0.829 on ChatGPT but 0.566 (near chance) on GPT-2 base completions.
- A stronger GLTR-style likelihood detector under GPT-2 beats the heuristics (0.995 vs 0.915 on HC3) but is ALSO per-document, so equally blind to collapse.
- Frozen GPT-2 perplexity did NOT beat a tiny in-distribution bigram LM as a feature (0.895 vs 0.924) - an information-mismatch caveat applies.
- Collapse experiments: recursively retraining a trigram LM, distilGPT-2, and Qwen2.5-0.5B on their own output. Vocabulary fell up to 87%, distilGPT-2 reference perplexity crashed 69->7 (Qwen: 37->2.9), yet the per-document detector stayed flat or sub-threshold (~91% of collapsed distilGPT-2 documents unflagged). Holds from 150 to 15,000 documents and on two datasets.
- Datasets: HC3 (human vs ChatGPT) and RAID (8 generators, attack=none). Prior work credited: Shumailov et al. discovered model collapse; this paper does NOT claim that discovery.
- TailGuard: the open-source corpus-health CLI built from the finding - snapshots vocabulary/n-gram diversity/duplication/score distributions and fails CI when a training corpus degrades.
- Honest limitations: English only, hand-fixed weights, collapse shown on small models under replacement-style retraining (accumulating data can mitigate collapse, per Gerstgrasser et al.), quantitative magnitudes are model-specific.

RULES: Be concise (under 150 words). Plain language. If asked something these facts don't cover, say so and point to the paper (PAPER.pdf on the site). Never invent numbers.`;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }
  // Provider chain: Groq (cloud, reliable) first; self-hosted AI-Loop as fallback.
  let base, key, model;
  if (process.env.GROQ_API_KEY) {
    base = "https://api.groq.com/openai/v1";
    key = process.env.GROQ_API_KEY;
    model = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
  } else if (process.env.AI_LOOP_BASE_URL && process.env.AI_LOOP_API_KEY) {
    base = process.env.AI_LOOP_BASE_URL;
    key = process.env.AI_LOOP_API_KEY;
    model = process.env.AI_LOOP_MODEL || "auto";
  } else {
    res.status(503).json({ error: "assistant_not_configured" });
    return;
  }
  let question = "";
  try {
    question = String((req.body && req.body.question) || "").trim().slice(0, 500);
  } catch (_) {}
  if (!question) {
    res.status(400).json({ error: "empty_question" });
    return;
  }

  const payload = JSON.stringify({
    model,
    stream: true,
    max_tokens: 400,
    temperature: 0.3,
    messages: [
      { role: "system", content: FACTS },
      { role: "user", content: question },
    ],
  });

  const call = () =>
    fetch(base.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + key },
      body: payload,
      signal: AbortSignal.timeout(55000),
    });

  let upstream;
  try {
    upstream = await call();
    // retry once (only before any stream bytes) on transient gateway states
    if ([429, 502, 503, 504].includes(upstream.status)) {
      await new Promise((r) => setTimeout(r, 1200));
      upstream = await call();
    }
  } catch (e) {
    res.status(504).json({ error: "assistant_offline" });
    return;
  }
  if (upstream.status === 429) {
    res.status(429).json({ error: "assistant_busy" });
    return;
  }
  if (!upstream.ok || !upstream.body) {
    res.status(502).json({ error: "assistant_error", status: upstream.status });
    return;
  }

  // Pipe SSE -> plain text chunks
  res.writeHead(200, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
  });
  const reader = upstream.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const ln of lines) {
        const t = ln.trim();
        if (!t.startsWith("data:")) continue;
        const data = t.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const delta = JSON.parse(data).choices?.[0]?.delta?.content;
          if (delta) res.write(delta);
        } catch (_) {}
      }
    }
  } catch (_) {
    /* client left or upstream died mid-stream; deliver what we have */
  }
  res.end();
}
