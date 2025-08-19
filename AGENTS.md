# Agent Implementation Summary (PPOLagSem Semantic Extension)

Internal reference capturing current semantic integration (CLIP shaping, spatial batching, risk head scaffold) and forward plan.

## 1. High-Level Snapshot
Baseline: PPO-Lagrangian (on‑policy, cost‑constrained).
Variant: `PPOLagSem` (activates when `semantic_cfgs.enable=True`).
Core Components:
* `SemanticManager`: CLIP load, prompt centroids, shaping, risk buffers, spatial batch embedding.
* `SemanticOnPolicyAdapter`: Frame capture, embedding (batched or single), shaping injection, semantic logging.
* (Scaffold) Risk head MLP for future auxiliary prediction & modulation.
Flow: env render(s) → CLIP embedding(s) → safe vs unsafe centroid cosine margin → annealed beta(t) → shaping reward addition.

## 2. Key Files
| File | Role |
|------|------|
| `omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag_semantic.py` | Registers semantic config & adapter swap |
| `omnisafe/common/semantics/semantic_manager.py` | Core semantics: CLIP, centroids, shaping, risk buffers, batching |
| `omnisafe/adapter/semantic_onpolicy_adapter.py` | Rollout with capture & per‑env shaping |
| `omnisafe/common/semantics/risk_head.py` | Auxiliary predictor (optional) |
| `omnisafe/algorithms/on_policy/base/policy_gradient.py` | Logging keys & future risk training hook |
| `omnisafe/configs/on-policy/PPOLagSem.yaml` | Extended semantic config (batching keys) |
| `tests/test_semantic_spatial_batching.py` | Step increment parity test (batch vs single) |
| `thesis.md` | Design narrative (v1.5 includes batching section) |

## 3. Runtime Behavior
Spatial Batching: Enabled if `batch_across_envs=True` & `vector_env_nums>1`; per‑env frames embedded together (capped by `batch_max`). One global semantic step per env step.
Fallback: Single frame path when batching disabled.
Shaping: Vector of per‑env shaping values added to reward; mean logged as `Semantics/Shaping`.
Risk Buffer: Stores embeddings & costs per capture when `risk_enable=True` (future training cycle).
Latency: Average per‑frame embedding latency recorded.
CLIP Safety: bfloat16→fp16→fp32 fallback; safetensors enforced; status logged.

## 4. Devices & Dtypes
| Component | Device Source | Dtype Fallback | Notes |
|-----------|---------------|----------------|-------|
| Policy nets | `train_cfgs.device` | fp32 | Baseline training |
| CLIP model | `semantic_cfgs.model_device` | bf16→fp16→fp32 | Frozen |
| Host embeddings | `semantic_cfgs.host_device` | Cast to fp32 if CPU | Used for shaping/risk |
| Risk head | Planned same as CLIP | Matches CLIP | Hook minimal now |

## 5. Logging Keys
Semantic: `Semantics/Shaping`, `Semantics/RawReward`, `Semantics/EmbedLatencyMs`, `Semantics/EmbedSuccessRate`, `Semantics/Debug/EmbedAttempts`, `Semantics/Debug/EmbedSuccess`, `Semantics/Debug/ClipReady`, `Semantics/Debug/ClipStatus`.
Risk (registered; pending population): `Risk/Loss`, `Risk/PredMean`, `Risk/TargetMean`, `Risk/Corr`.

## 6. Limitations
1. No temporal (cross‑step) embedding batching; spatial only.
2. Risk head training & metrics not yet active.
3. Shaping distribution (std/min/max) not logged (only mean).
4. Per‑env render may replicate one frame if true multi‑render unsupported.
5. Lacks tests for margin normalization, dtype fallback, anneal schedule, risk correlation.

## 7. Planned Enhancements (Priority)
1. Implement risk head training loop (mini‑batch updates; populate risk metrics).
2. Add batch diagnostics: shaping std/min/max and batch size metric.
3. Lagrange modulation via risk quantile gating (`modulation_scale`).
4. Temporal or async embedding micro‑batching.
5. Expanded unit tests (margin normalization, beta schedule, dtype fallback, risk correlation, shaping parity disabled batching).
6. Prompt adaptation strategies (dynamic weighting / refresh).

## 8. Troubleshooting
| Issue | Diagnostic | Resolution |
|-------|-----------|------------|
| YAML edits ignored | Check startup log path; inspect `config.json` | Avoid conflicting CLI overrides |
| CLIP not loaded | `ClipStatus` = `load_error:*` | Verify model id & safetensors; ensure cache/internet |
| High latency | Large `EmbedLatencyMs` | Increase `capture_interval` or move CLIP to GPU |
| No shaping effect | `Semantics/Shaping` ~0 early | Confirm `shaping_enable`, `beta_start>0`, capture interval active |
| Headless render failure | Render exceptions | Use synthetic frame fallback (planned utility) |

## 9. Semantic Config Flags
| Key | Purpose | Default (YAML) |
|-----|---------|----------------|
| enable | Master semantic switch | False |
| capture_interval | Steps between embedding captures | 4 |
| model_device | CLIP compute device | cpu |
| host_device | Embedding storage device | cpu |
| shaping_enable | Enable reward shaping | False |
| risk_enable | Enable risk head | False |
| beta_start | Initial shaping coefficient | 0.05 |
| beta_end_step_fraction | Anneal fraction of total steps | 0.4 |
| risk_horizon | Discount horizon for risk target | 64 |
| discount | Risk target discount | 0.99 |
| window_size | Risk buffer size | 2048 |
| safe_prompts / unsafe_prompts | Prompt sets | 3 each |
| batch_across_envs | Spatial batch toggle | True |
| batch_max | Max frames per embedding batch | 32 |
| oom_backoff | Recursive halve on CUDA OOM | True |

## 10. Performance Notes
* Balance `capture_interval` vs overhead; target <10% wall time embedding cost.
* Increase `vector_env_nums` until embedding latency or memory limits reached; adjust `batch_max` accordingly.
* If CLIP on CPU, consider interval ≥4 to mitigate latency.

## 11. Open Design Decisions
* Direct embedding infusion into policy (feature concat) deferred pending shaping baseline evaluation.
* Prompt refresh cadence (static vs adaptive) undecided.
* Risk head depth & regularization tuning postponed until baseline predictive utility observed.

## 12. Immediate TODOs
- [ ] Populate & train risk head; log risk metrics.
- [ ] Add shaping distribution & batch size logging.
- [ ] Unit tests: margin normalization & beta schedule correctness.
- [ ] Central synthetic frame fallback helper (avoid ad‑hoc patches).

## 13. Version Stamp
* Thesis Document: v1.5
* Agent Summary: v1.2 (spatial batching integrated; updated TODOs & limitations)

---
For continuity—update incrementally with each semantic feature change (do not delete).
