# Issue #3: correctness contract and limitations

This change repairs training/evaluation mechanics after the Issue #2 packaging
baseline. It intentionally changes loss values, gradient flow, evaluation
predictions, input rejection, and inference uncertainty/conflict reporting.
It does not establish improved ARGUS performance.

| Before | After |
| --- | --- |
| Fusion probabilities fed to `BCEWithLogitsLoss` | Fusion probabilities fed to `BCELoss`; boundary loss/gradients tested for finiteness |
| Evaluation and relevance gating sigmoid the probability again | Both compare the probability directly with the unchanged `> 0.5` threshold |
| Fusion softmax weights converted to Python floats in computation | Tensor weights remain differentiable; float conversion is only for reports/inference conflict overrides |
| CPU float32 temporaries, including fallback and reshape | Temporary tensors follow the first floating input's device/dtype, or module parameters for all-scalar/empty input; MPS uses float32 on the selected device; reshape retains gradients |
| Trigger detection coupling unverified | Separate backward diagnostics expose no detection-loss connection; strict xfail documents it |
| Missing labels default to phishing; URL substrings can set labels | Explicit `label:` / `class:` fields or standalone label tokens; missing labels rejected unless `--default-label 0/1` is explicitly selected and counted |
| Single-class or empty evaluation reports misleading metrics | Label counts checked; evaluation raises before returning metrics or appending metric history |
| Feature/prior errors replaced by valid-looking zeros | Invalid samples carry stage/error/folder metadata; parent process counts worker failures and aborts |
| URL/HTML extraction block and WHOIS exceptions swallowed | Those exceptions propagate to countable dataset failures (or the inference caller) |
| Inference uncertainty falls back to 0.5 | Uses `ConfidenceEstimator.compute_overall_confidence()` and its existing conservative minimum rule |
| One true conflict causes all names to be reported | Only true conflicts are reported |
| LLM/image status ambiguous | Results explicitly label the prompt-only LLM stub and unused-image/visual and semantic placeholders |

## Probability and confidence contracts

`OmniModalFusion.forward()` returns clamped probabilities in `[0, 1]`.
`ARGUSTrainer.compute_loss_terms()` returns tensor-valued losses and those
probabilities so each loss can be differentiated independently. The existing
`compute_loss()` interface still returns total loss plus detached logging values.
Evaluation reuses the same forward result for loss and predictions. No sigmoid
is applied to a fused probability. Inference retains its existing 0.4/0.7 risk
bands and conflict adjustment. Neither threshold set was tuned.

The existing branch-specific risk heuristics are preserved: training uses
`sigmoid(mean(rule_features))` and 0.5 visual/semantic placeholders, while
inference uses the rule mean and 0.5/0.6 placeholders. Both feed probability
scores to fusion; this PR does not redesign or reconcile those heuristics.

Overall confidence has one inference source: the confidence estimator's minimum
of quantitative, visual, and semantic confidence; uncertainty is its complement.
The existing 0.8 defaults for missing confidence entries are retained, now with
broadcasting and matching dtype/device. This is a heuristic including placeholder
modality confidences, not calibrated uncertainty. It is separate from fusion's
normalized evidence weights. Inference results remain JSON-serializable.

## Gradient diagnostics

Run on CPU with fixed synthetic inputs (not measured phishing performance):

```bash
python -m pytest -q tests/test_gradients.py -s
```

Each diagnostic rebuilds the components with seed 7 and backpropagates exactly
one tensor loss. A finite nonzero gradient means the **parameter tensor's gradient
norm is nonzero**, not that every element or every possible input has a gradient.
The test prints every parameter's name and distinguishes nonzero, zero, and
absent (`grad is None`) gradients.

| Loss differentiated | Parameter tensors with finite nonzero gradients | Disconnected |
| --- | --- | --- |
| Detection loss alone | Confidence and fusion parameters below | All trigger parameters |
| Efficiency penalty alone | All trigger parameters below | Confidence and fusion |
| Relevance reward alone | All trigger parameters below | Confidence and fusion |
| Weighted auxiliary loss (efficiency minus relevance) | All trigger parameters below | Confidence and fusion |
| Total loss | All parameters below | None in this fixture |

Trigger parameters: `trigger.alpha_logit`, `trigger.residual_net.0.weight`,
`trigger.residual_net.0.bias`, `trigger.residual_net.3.weight`,
`trigger.residual_net.3.bias`, `trigger.residual_net.6.weight`,
`trigger.residual_net.6.bias`.

Confidence parameters: `confidence.trustworthiness_weight`,
`confidence.coverage_weight`, `confidence.consistency_weight`.

Fusion parameters: `fusion.weight_logits.quantitative`,
`fusion.weight_logits.visual`, `fusion.weight_logits.semantic`.

**LearnableTrigger is not detection-loss-connected after this PR.** Selected
scores/tasks still do not condition detection risk. The existing relevance term
uses a nondifferentiable correctness gate, not task-specific causal relevance.
No new coupling/objective was introduced. The strict xfail
`test_trigger_is_detection_loss_connected` will become an unexpected pass if that
relationship changes and must then be reviewed under a research-design issue.
Explicit inference `adjusted_weights` still override learned base weights; this
path is not expected to backpropagate to base-weight parameters.

## Dataset and failure policy

Each selected sample must contain an observed HTTP(S) URL and a label in
`info.txt`, for example:

```text
url: https://example.com/
label: 0
```

Recognized labels are `0`, `benign`, `legitimate`, `legit` and `1`, `phishing`,
`phish`, `malicious`. Matching applies to a label/class field or a standalone
line, never a URL substring. Conflicting/unrecognized explicit labels fail even
when a default was supplied. URL fabrication is removed. The optional
`--default-label 0` or `--default-label 1` applies only to missing labels;
`defaulted_label_count` records its use. Missing/invalid metadata fails preflight
with the sample name. It is not silently removed from the split.

Split order/shuffle, ratios, seed and label meanings (0 benign, 1 phishing) are
unchanged. Counts are printed when each dataset is constructed, before training
or evaluation. Empty splits fail. Single-class training warns; single-class
validation/test splits fail. The trainer additionally checks the labels actually
seen before returning binary metrics, including for custom loaders.

`collate_samples` carries structured failures from DataLoader workers to the
parent. No failed item contains feature/prior zero substitutes. Shape/nonfinite
feature/prior outputs are failures too. `trainer.data_report` includes attempted
sample counts, valid processed label counts, and separate feature/prior/file
failure counts; dataset preflight counts are included when available. Any failed
batch aborts before that batch's optimizer update or any evaluation metrics.
Counts cover attempted batches through the first failure, not unseen samples
or prefetched worker work. Earlier successful training batches may already have
updated parameters; this is fail-fast handling, not transactional rollback or
an automatic sample-skipping policy. Custom loaders must use `collate_samples`
for structured worker-failure aggregation.

Optional WHOIS/HTML dependencies remain optional; absent capabilities retain the
existing documented fallback features and availability warnings. Normal helper
fallbacks (e.g. a domain not parsing as an IP or TLD) and missing WHOIS date fields
are unchanged. This is not a redesign of feature missingness. Enabled WHOIS
lookup exceptions and URL/HTML extraction-block exceptions now fail explicitly.
Visual image loading/augmentation remains unused by detection, with its existing
warnings; no real visual/semantic pipeline or LLM API is added.

## Compatibility and validation limits

Parameter names/shapes, state-dict keys, optimizer parameter groups and checkpoint
schema are unchanged. A synthetic checkpoint round-trip and exact parameter
key/shape regression are tested. Old checkpoints are structurally loadable but
resuming training uses the corrected objective; old and new losses/metrics are
not numerically comparable. Full legacy datasets/checkpoints and experiments
were not run. No new accuracy/F1/TPR/FPR results are claimed.

The full suite includes float32/float64 CPU fusion cases, scalar/missing/reshape
and no-modality paths, gradient checks, multiprocess failure aggregation,
metadata/evaluation safeguards, inference stubs and checkpoint structure. CUDA
variants are collected and skipped when hardware is absent; a skip is not CUDA
execution evidence. CI runs the same full `python -m pytest -q` command.

## Apple Silicon deployment update

The latest Issue #3 comment adds MPS as a first-class target. Current main's
initial device selection (`961d5ac`) is integrated into the same Issue #3 PR.
`argus.utils.runtime.select_device()` is the shared implementation; the old
`config._select_device` name remains a compatibility alias. Priority is MPS,
CUDA, CPU. Training seeding uses the CPU generator plus `torch.mps.manual_seed`
or `torch.cuda.manual_seed_all` only for the selected backend, with Python and
NumPy seeds retained. No CUDA API is used to seed MPS.

The trainer moves its modules and all compute inputs to the selected device;
MPS casts to float32 before transfer, including direct loss callers supplying
CPU float64 arrays. Entry points explicitly use float32, inference constants
follow the feature tensor, and MPS fusion/overall-confidence normalization uses
the module's selected device. CPU floating precision remains supported. Pinned
DataLoader memory is enabled only for CUDA. State-dict keys/shapes remain the
same; choosing a different backend can produce normal floating-point/RNG
variation, not bitwise equality or a model-performance improvement.

`python -m argus.utils.device_smoke` reports real device availability and runs
Linear plus ARGUS total-loss backward operations; it refuses an enabled silent
MPS CPU fallback. It reports no accuracy/F1 or model risk output. Unsupported
operators are allowed to fail explicitly, never hidden with automatic fallback.
Real MPS tests cover fusion shapes/fallback, float32 transfer, optimizer step,
inference, and separate detection/auxiliary/total gradient diagnostics. They
confirm the same trigger disconnection on MPS; the objective is unchanged by
this deployment refinement. CUDA hardware execution remains unverified on the
Mac. Synthetic smoke success is not a complete production deployment audit.
