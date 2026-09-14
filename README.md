# ARGUS-Adaptive-Reasoning-Grounded-Under-Statistics-for-Phishing-Detection
We design ARUGUS that uses deterministic statistical and structural signals not only to score webpages, but to dy- namically decide what to verify, which evidence sources to trust, and how to constrain LLM reasoning so that decisions remain grounded under adaptive attack



## Install and run the CPU baseline (Issue #2)

Requires Python 3.10 or newer; CI uses Python 3.12. From a clean checkout:

```bash
git clone https://github.com/swei8272/ARGUS-Adaptive-Reasoning-Grounded-Under-Statistics-for-Phishing-Detection.git
cd ARGUS-Adaptive-Reasoning-Grounded-Under-Statistics-for-Phishing-Detection
python3 -m venv .venv
source .venv/bin/activate
# Linux: install the CPU wheel before installing the package.
# macOS: skip this line; the default PyTorch wheel supports CPU.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip install -e '.[test]'
python -m pytest -q
python -m argus.inference --help
python -c "import argus.train; import argus.inference; print('Entry points import successfully')"
```

No manual `PYTHONPATH` changes, GPU, dataset, checkpoint or paid API are needed
for these checks. The inference `--help` invocation only exercises the entry
point. Tests explicitly use CPU tensors and synthetic component inputs; they
are not measured phishing-detection results. WHOIS is disabled in tests to keep
them offline. Test dependencies are separate from runtime dependencies.

### Apple Silicon Mac deployment

Apple Silicon acceleration uses **PyTorch MPS/Metal, not CUDA**. Install the
standard macOS PyTorch wheel through the package setup above; skip the Linux
CPU-wheel command. Training and inference share `argus.utils.runtime` device
selection: **MPS first, then CUDA on NVIDIA systems, otherwise CPU**. On the
Mac this means MPS when available and CPU otherwise.

The MPS path explicitly uses float32 for model parameters, features, prior
scores, floating labels, constants and fusion buffers. CPU float64 inputs are
converted before MPS dispatch. Do not call `.double()` on MPS tensors. Host data
loading remains on CPU; pinned host memory is enabled only for CUDA loaders.

```bash
# Do not conceal unsupported Metal operators with CPU fallback.
unset PYTORCH_ENABLE_MPS_FALLBACK
python -m argus.utils.device_smoke
python -m pytest -q tests/test_devices.py tests/test_gradients.py -s
```

The smoke command prints MPS build/availability, the selected and actual device,
and real float32 tensor/model and ARGUS backward checks using synthetic inputs.
Unsupported operators raise normally; ARGUS does not enable CPU fallback. The
training seed helper seeds Python, NumPy, the CPU generator and only the selected
accelerator's generator. Seeding is not a promise of bitwise reproducibility
across backends. Device-selection mocks run on Linux CI; real MPS tests run only
when MPS is available. No Mac test skip is evidence of real Metal execution.

Optional capabilities:

```bash
python -m pip install -e '.[html]'   # BeautifulSoup HTML feature extraction
python -m pip install -e '.[whois]'  # live WHOIS lookups (network-dependent)
```

The default install includes `tld` for URL parsing. Without HTML/WHOIS extras,
the extractor prints its existing availability warnings and uses its existing
fallbacks. Installing these extras can change which existing feature paths run
and thus feature values; comparisons must use the same dependency environment.
Dependency ranges provide a supported installation baseline, not a fully locked
cross-platform numerical environment.

### Package layout and migration

The distribution is `argus-phishing`; all Python imports use lowercase `argus`.
This conventional namespace replaces the previously unresolved `ARGUS`, `core`,
and `training` imports. There are no duplicate root modules or compatibility
shims. Use `python -m argus.train` and `python -m argus.inference` instead of the
old root scripts. Public component classes are also exported from `argus`.

| Old root files | New package paths |
| --- | --- |
| `feature_extractor.py`, `prior_trigger.py`, `learnable_trigger.py`, `confidence_estimator.py`, `multimodal_fusion.py`, `conflict_detector.py`, `conflict_resolver.py`, `prompt_generator.py`, `constants.py` | Same filenames under `argus/core/` |
| `trainer.py`, `adversarial_aug.py`, `cost_tracker.py` | Same filenames under `argus/training/` |
| `config.py` | `argus/utils/config.py` |
| `train.py`, `inference.py` | Same filenames under `argus/` |
| `test_argus.py` (manual print script) | Replaced by assertion-based `tests/test_smoke.py` |

### Scientific limitations

Issue #2 packaging preserved algorithmic and numerical source behavior:
feature definitions, thresholds, losses, labels, dataset splits and metrics are
preserved. Imports and module identities change, so external scripts using old
paths must migrate; full-object pickles referencing old module paths are not
compatibility-tested. Existing state-dict keys are unchanged.

Visual/semantic evidence and the LLM integration still contain placeholders;
without a checkpoint neural components have untrained initialization. Simplified conflict rules remain in place; Issue #3 adds explicit dataset and
extraction failure handling as described below.
Passing CPU smoke tests does not establish an end-to-end scientifically complete
ARGUS system, validate training, or reproduce paper metrics. No experiment
numbers are generated here. The prompt example below is illustrative, not a
measured model output.

## Training/evaluation correctness (Issue #3)

Fusion scores are probabilities: training uses `BCELoss`, and evaluation applies
`> 0.5` directly. Fusion weights now receive detection-loss gradients. The
learnable trigger remains disconnected from detection loss and is trained only
through the existing auxiliary terms; a strict xfail diagnostic exposes this.

Dataset labels must be explicit (`label: 0` or `label: 1` in `info.txt`), or an
explicit `--default-label 0/1` policy must be supplied. Label counts are reported;
empty/single-class evaluation is rejected. Feature/prior failures are counted
and abort processing instead of entering metrics as zero observations.

Inference uncertainty now uses the confidence estimator's conservative minimum,
and only true conflicts are reported. `use_llm=True` still only generates a
prompt, and `image` is unused. Results expose `llm_status`, `visual_status`, and
`semantic_status` so these placeholders are explicit.

```bash
python -m pip install -e .
python -m pip install -e '.[test]'
python -m pytest -q
python -m pytest -q tests/test_gradients.py -s
python -m argus.inference --help
```

These correctness fixes intentionally change numerical training/evaluation
behavior; no improved detection performance is claimed. See
[the correctness contract](docs/correctness.md) for the BEFORE/AFTER table,
parameter-level gradient paths, failure policy, checkpoint compatibility and
remaining limitations.

## Below is the example of the complete Prompts mentioned in ARGUS:




Prompt Example 1: clear  logo，domain does not match the brand

You are a security analyst LLM that must decide whether a webpage is a phishing site.

Your job:
- Carefully check all the evidence provided (visual, URL/HTML features, semantic analysis, conflicts).
- Perform cross-modal reasoning.
- Output ONLY a JSON object that follows the schema at the end. Do not add any extra text.

========================
 BASIC CONTEXT
========================
——
URL: https://secure-paypa1-login.com/account/verify
Page intent (semantic): login / account verification

This site visually imitates a well-known brand: "PayPal".

========================
VISUAL SUMMARY (from screenshot analysis)
========================

- has_logo: true
- detected_brand: "PayPal"
- official_brand_domain: "paypal.com"
- visual_brand_confidence: 0.96
- visual_verification_status: "verified_by_template_match"
- logo_count: 1
- ui_complexity_score: 0.4        # simple single-page login

Interpretation:
The page looks very similar to the official PayPal login page 
and the logo is detected with high confidence.

========================
QUANTITATIVE SUMMARY (SpacePhish-style URL/HTML features)
========================

risk_score: 91 / 100              # aggregated URL + HTML risk score

Top risk indicators:
- feature: "has_ip_like_pattern", value: 1
- feature: "domain_brand_similarity", value: 0.93   # looks like the brand name but not exact
- feature: "has_login_keyword", value: 1
- feature: "has_verify_keyword", value: 1
- feature: "domain_age_days", value: 7
- feature: "suspicious_tld", value: 1               # .com, but used in a low-reputation context
- feature: "external_script_ratio", value: 0.85

Important URL features:
- domain: secure-paypa1-login.com
- path: /account/verify
- is_ip_domain: false
- num_subdomains: 1
- has_suspicious_keywords: true

========================
SEMANTIC SUMMARY (from URL + HTML text)
========================

intent_type: "account_verification"
asks_password: true
asks_2fa: true
asks_card: false
urgency_score: 0.82        # text like "Your account will be limited within 24 hours"
threat_score: 0.76         # mentions "account suspension"
semantic_brand_consistency_score: 0.65
content_language: "English"
lang_region_mismatch: false

Key semantic observations:
- The page strongly pushes the user to "verify your account now" and "avoid account restriction".
- Multiple references to "PayPal", but the domain in the browser address bar does not match the official PayPal domain.

========================
FORMALIZED CROSS-MODAL CONFLICTS
========================

conflicts:
- [CRITICAL] Visual brand-domain mismatch:
  - visual_brand: "PayPal"
  - expected_official_domain: "paypal.com"
  - actual_domain: "secure-paypa1-login.com"
  - domain_similarity: 0.93
  - description: The logo and design match PayPal, but the domain uses 'paypa1' with the digit '1' instead of 'l', which is a typical typosquatting pattern.

- [HIGH] Visual vs quantitative inconsistency:
  - visual_confidence: 0.96 (page looks like official PayPal)
  - risk_score: 91/100 (very high)
  - top_risk_features: ["typosquatting", "new_domain", "login+verify keywords"]
  - description: The page appears visually legitimate but URL/HTML risk features are highly suspicious.

Overall conflict_score (0–1): 0.88

========================
INSPECTION CHECKLIST (FOCUSED TASKS)
========================

Focus on the following when making your judgment:

1) Brand–domain consistency
   - Check whether "PayPal" and the official domain "paypal.com" are compatible with the actual domain "secure-paypa1-login.com".
   - Consider typosquatting patterns (digit '1' replacing letter 'l').

2) Credential harvesting risk
   - The page requests password and 2FA codes under an urgent "verify your account" scenario.
   - Evaluate whether this behaviour is typical for official PayPal login pages at this domain.

3) Cross-modal conflicts
   - Visual analysis says "highly likely official PayPal".
   - Quantitative features and semantic patterns strongly suggest phishing.
   - Resolve this contradiction based on how real services behave.

4) Zero-day consideration
   - Assume you do NOT have any internal blacklist or pre-known signature for this domain.
   - Base your reasoning only on the evidence above, without relying on any external threat feed.

========================
DECISION RULE
========================

You must output one of three verdicts:
- "phishing": clear malicious intent or very strong evidence of phishing.
- "legitimate": highly likely an authentic webpage of the claimed service.
- "suspicious": unclear, but there are non-trivial red flags; needs manual review.

Be strict: if strong critical conflicts exist (like brand-domain mismatch with high 
risk_score), "phishing" is usually appropriate unless there is a very strong reason otherwise.

========================
OUTPUT FORMAT (JSON ONLY)
========================

Return ONLY a JSON object with this schema:

{
  "verdict": "phishing" | "legitimate" | "suspicious",
  "confidence": 0.0-1.0,
  "risk_score": 0-100,
  "key_evidence": [
    "short bullet 1",
    "short bullet 2",
    "short bullet 3"
  ],
  "conflict_analysis": {
    "has_critical_conflict": true/false,
    "main_conflicts": ["type1", "type2"]
  },
  "reasoning_summary": "2-4 sentences explaining why you chose this verdict."
}

Remember: output ONLY this JSON, no extra commentary.
