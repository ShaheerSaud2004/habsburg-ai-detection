# Ready-to-paste submission metadata (updated short paper)

Use these exact fields when finishing Zenodo / OSF / OSF Preprints.

## Title
```
Per-Document AI-Text Detection Is Not a Reliable Model-Collapse Monitor
```

## Abstract
```
Model collapse—degradation under recursive training on generated data [Shumailov2023]—is often conflated with per-document AI-text detection. We argue these are distinct objectives: detectors that correctly flag synthetic documents can fail to register distributional degeneration. We evaluate a six-statistic heuristic (SyntheticTextProbe) and a leave-one-model-out logistic classifier over those statistics plus one perplexity feature on HC3 and RAID (eight generators). The classifier attains mean held-out AUC 0.915 ± 0.006 (10 seeds; 95% CI [0.911, 0.920]), above perplexity alone (0.903 ± 0.007; paired permutation p = 0.002) and a fixed zero-shot heuristic (0.732). Replacing an in-distribution bigram perplexity feature with frozen GPT-2 (124M) perplexity does not improve mean AUC under this protocol (0.895), a comparison we flag as information-mismatched. In recursive-collapse experiments (trigram LM, distilGPT-2, Qwen2.5-0.5B), vocabulary size, n-gram diversity, and reference perplexity change monotonically with collapse, while per-document detector scores remain flat or sub-threshold (e.g., ~91% of collapsed distilGPT-2 outputs unflagged). Stronger likelihood detectors (GLTR-style) improve document AUC but share the same unit-of-analysis limitation. Collapse monitoring should track corpus-level diversity and reference perplexity rather than aggregated document verdicts.
```

## Author
- Name: Shaheer Saud
- Email: shaheersaud2004@gmail.com
- Affiliation: Independent

## License / type
- Resource type: Preprint
- License: CC-BY 4.0
- Language: English
- Subjects: Computer and Information Sciences → Artificial Intelligence
- Tags: model collapse, AI-text detection, synthetic data, LLM training data, data quality, machine-generated text, reproducibility

## Files to upload
- PAPER.pdf (repo root; new short version)
- PAPER.md
- /tmp/habsburg-paper-code.zip

## Links
- OSF project (already public): https://osf.io/4m53j
- Zenodo new upload: https://zenodo.org/uploads/new
- OSF Preprints: https://osf.io/preprints/
- OpenReview signup: https://openreview.net/signup
- arXiv submit: https://arxiv.org/submit
