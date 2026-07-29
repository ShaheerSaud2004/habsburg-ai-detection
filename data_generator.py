"""
data_generator.py -- offline generation of human-like vs synthetic-like text,
contaminated datasets, and multi-generation collapse simulation.

The offline generators are deterministic given a passed-in random.Random(seed),
so every experiment is reproducible. They use ONLY the standard library.

IMPORTANT (honesty): these are SYNTHETIC STAND-INS designed to exercise the
detector's statistical signals (diversity, burstiness, filler, repetition).
"Human" text here is varied/bursty word-salad, not real prose. Validating the
detector on REAL corpora (Wikipedia/books vs real LLM output) is the key piece
of future work -- see NEXT_STEPS.md.

An OPTIONAL live mode (generate_via_api) uses the anthropic SDK if it is
installed and ANTHROPIC_API_KEY is set. It is never required for offline use.
"""

import os

# --- broad, varied vocabulary -> HIGH lexical diversity for "human" text ---
_HUMAN_NOUNS = [
    "harbor", "kestrel", "ledger", "monsoon", "violin", "granite", "orchard",
    "lantern", "satchel", "meadow", "compass", "thicket", "ferry", "almanac",
    "cobbler", "marsh", "trellis", "quarry", "saffron", "buoy", "cipher",
    "willow", "anvil", "plateau", "drizzle", "sextant", "bramble", "kiln",
    "marrow", "fjord", "lichen", "pewter", "tundra", "vellum", "zephyr",
    "barnacle", "cinder", "dovetail", "estuary", "flint",
]
_HUMAN_VERBS = [
    "wandered", "tangled", "drifted", "scattered", "whittled", "hummed",
    "fractured", "simmered", "lurched", "blossomed", "creaked", "splintered",
    "meandered", "flickered", "ricocheted", "unspooled", "smoldered",
    "capsized", "germinated", "echoed", "wilted", "shimmered",
]
_HUMAN_ADJ = [
    "brittle", "luminous", "crooked", "feral", "sodden", "ramshackle",
    "iridescent", "gaunt", "saline", "verdant", "threadbare", "molten",
    "wizened", "jagged", "muffled", "sprawling", "tarnished", "cavernous",
]
_HUMAN_CONNECTORS = [
    "and yet", "though", "while", "until", "because", "after", "before",
    "whenever", "wherever", "as if", "even when", "long before",
]
_HUMAN_FUNCTION = ["the", "a", "her", "his", "their", "some", "no", "that", "this"]

# --- narrow vocabulary + heavy filler -> LOW diversity, HIGH filler ("synthetic") ---
_SYN_FILLER = [
    "It is important to note that",
    "In today's world,",
    "When it comes to this topic,",
    "Furthermore, it is worth noting that",
    "In conclusion,",
    "Overall,",
    "As we can see,",
    "Moreover,",
    "It plays a crucial role in how",
    "This is a testament to how",
]
_SYN_NOUNS = ["solution", "system", "process", "approach", "framework", "result"]
_SYN_VERBS = ["provides", "enables", "delivers", "ensures", "supports"]
_SYN_ADJ = ["powerful", "innovative", "robust", "seamless", "comprehensive"]


def generate_human_text(rng, topic=None):
    """Varied sentence lengths, rich/rare vocabulary, almost no filler."""
    n_sent = rng.randint(4, 9)
    sents = []
    for _ in range(n_sent):
        roll_len = rng.random()
        if roll_len < 0.35:
            n = rng.randint(3, 7)      # short
        elif roll_len < 0.75:
            n = rng.randint(8, 18)     # medium
        else:
            n = rng.randint(19, 30)    # long
        words = []
        for _ in range(n):
            r = rng.random()
            if r < 0.30:
                words.append(rng.choice(_HUMAN_NOUNS))
            elif r < 0.50:
                words.append(rng.choice(_HUMAN_VERBS))
            elif r < 0.68:
                words.append(rng.choice(_HUMAN_ADJ))
            elif r < 0.80:
                words.append(rng.choice(_HUMAN_CONNECTORS))
            else:
                words.append(rng.choice(_HUMAN_FUNCTION))
        s = " ".join(words)
        if rng.random() < 0.08 and len(s) > 5:   # occasional typo
            j = rng.randrange(len(s))
            s = s[:j] + s[j] + s[j:]
        sents.append(s[0].upper() + s[1:])
    return ". ".join(sents) + "."


def generate_synthetic_text(rng, topic=None, generation=1):
    """
    Templated, filler-heavy, uniform-length, narrow-vocabulary text.
    Higher `generation` => MORE collapsed (narrower vocab, more repetition).
    """
    vocab_cap = max(2, 5 - generation)          # gen1->4, gen2->3, gen3->2, gen4->2
    nouns = _SYN_NOUNS[:vocab_cap]
    verbs = _SYN_VERBS[:max(2, vocab_cap)]
    adjs = _SYN_ADJ[:max(2, vocab_cap)]
    repeat_prob = min(0.85, 0.35 + 0.15 * generation)

    n_sent = rng.randint(4, 7)
    sents, prev_core = [], None
    for _ in range(n_sent):
        filler = rng.choice(_SYN_FILLER)
        if prev_core is not None and rng.random() < repeat_prob:
            core = prev_core                    # repeat -> repeated trigrams
        else:
            core = "the {a} {n} {v} the {a2} {n2}".format(
                a=rng.choice(adjs), n=rng.choice(nouns),
                v=rng.choice(verbs), a2=rng.choice(adjs), n2=rng.choice(nouns),
            )
        prev_core = core
        sents.append(filler + " " + core)
    return ". ".join(sents) + "."


def make_dataset(rng, n, contamination, generation=1):
    """Return a shuffled list of (text, label) with label in {human, synthetic}."""
    n_syn = int(round(n * contamination))
    items = []
    for i in range(n):
        if i < n_syn:
            items.append((generate_synthetic_text(rng, generation=generation), "synthetic"))
        else:
            items.append((generate_human_text(rng), "human"))
    rng.shuffle(items)
    return items


def simulate_generations(rng, num_generations, n_per_gen=60):
    """One dataset per generation, each more synthetic-dominated AND more collapsed."""
    datasets = []
    for g in range(1, num_generations + 1):
        contamination = min(1.0, 0.25 * g)
        datasets.append(make_dataset(rng, n_per_gen, contamination, generation=g))
    return datasets


def generate_via_api(prompt, model="claude-haiku-4-5"):
    """OPTIONAL live generation. Raises a friendly error if unavailable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Live generation needs ANTHROPIC_API_KEY in your environment. "
            "The offline generators need no API key and are the default."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run:  pip install anthropic  "
            "(only needed for the optional live-generation mode)."
        )
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
