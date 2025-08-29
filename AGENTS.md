# Agent Implementation Summary (PPOLagSem Semantic Extension)

Internal reference capturing current semantic integration (CLIP shaping, spatial batching, risk head scaffold) and forward plan.

## 1. High-Level Snapshot
Baseline: PPO-Lagrangian (on‑policy, cost‑constrained).
Variant: `PPOLagSem` (activates when `semantic_cfgs.enable=True`).
Core Components:
* `SemanticManager`: CLIP load, prompt centroids, shaping, risk buffers, spatial batch embedding.
* `SemanticOnPolicyAdapter`: Frame capture, embedding (batched or single), shaping injection, semantic logging.
* Risk head MLP for auxiliary future cost prediction & (initial) Lagrange modulation.
Flow: env render(s) → CLIP embedding(s) → safe vs unsafe centroid cosine margin → annealed beta(t) → shaping reward addition.

Shaping Modes:
* Additive: r' = r + beta * margin (normalized/scaled & clipped).
* Potential-based (preferred for neutrality): r' = r + beta * (gamma * phi(s') - phi(s)), with phi(s)=processed margin. Toggle via `potential_enable`.

## 2. Key Files
| File | Role |
|------|------|`
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
Semantic core:
* `Semantics/Shaping` – shaping term applied (aggregated)
* `Semantics/RawReward` – environment reward pre-shaping
* `Semantics/ShapingRewardRatio` – mean shaping / mean raw reward (curriculum influence)
* `Semantics/ShapingStd` – per-capture shaping std (0 if scalar path)
* `Semantics/EmbedLatencyMs` – per-frame latency (avg for batch)
* `Semantics/EmbedSuccessRate`, `Semantics/Debug/EmbedAttempts`, `Semantics/Debug/EmbedSuccess` – embedding reliability
* `Semantics/RawMargin`, `Semantics/NormMargin` – last raw & normalized cosine margins
* `Semantics/Beta` – current shaping coefficient
* `Semantics/ClampFrac` – fraction of historical normalized margins clipped
* `Semantics/CaptureCount` – number of capture events
* `Semantics/CaptureIntervalEffective` – steps since last capture
* `Semantics/TemporalWindowFill` – fraction (0–1) of temporal embedding window currently filled (0 when disabled)

Risk (active): `Risk/Loss`, `Risk/PredMean`, `Risk/TargetMean`, `Risk/Corr`, `Risk/ModulationScale`, `Risk/ModulationActive`.

Modulation gating telemetry:
* `Risk/ModulationActive` – 1.0 once `episodes_completed >= modulation_min_episodes`, else 0.0; allows clear visualization of when modulation actually began.

Removed legacy: `Semantics/Debug/ClipReady`, `Semantics/Debug/ClipStatus`.

## 6. Limitations
1. Temporal pooling is simple mean over last k embeddings (if `temporal_window>1`); no learned weighting or attention yet.
2. Risk targets use backward discounted rollout with episode-aware masking (prevents cross-episode leakage) but no bootstrap tail.
3. Per-capture shaping std not logged (only epoch aggregation for shaping mean/min/max).
4. Some vector envs may yield a single composite frame (replication heuristic used).
5. Missing unit tests: margin normalization toggle, beta schedule edges, risk correlation synthetic.
6. Lagrange modulation currently uses a simple logistic quantile gap scaling with *episode-count gating* only (no quality gating or EMA smoothing yet).

## 7. Planned Enhancements (Priority)
1. Bootstrap tail for risk targets (optional value function mix) & target normalization.
2. Smoothing / ablation for Lagrange modulation (EMA, alternative metrics).
3. Shaping distribution (per-capture std) & batch size logging.
4. Unit tests: beta schedule, margin scaling toggle, risk correlation synthetic, batching parity.
5. Prompt adaptation / dynamic refresh (guided by offline probe scoring).
6. Temporal/async embedding (only if profiling bottleneck).
7. Offline probe temporal smoothing & automated prompt pruning.

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
| margin_norm_enable | Enable z-normalization of margin | True |
| margin_scale | Multiplicative scale on raw margin | 1.0 |
| potential_enable | Use potential-based shaping instead of additive | False |
| risk_lr | Risk head learning rate | 1e-3 |
| risk_episode_mask_enable | Episode-aware masking for risk targets | True |
| modulation_enable | Enable Lagrange lr modulation | False |
| modulation_min_episodes | Episode count gate before modulation activates | 10 |
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
- [ ] Episode boundary masking for backward risk targets.
- [ ] Lagrange modulation quantile experiment.
- [ ] Shaping per-capture std & batch size metric.
- [ ] Unit tests (beta schedule, margin scaling, risk correlation synthetic).
- [ ] Synthetic frame fallback helper.
- [ ] Offline probe temporal smoothing + prompt pruning scoring.

## 13. Version Stamp
* Thesis Document: v1.9 (episode-aware risk masking + lr modulation)
* Agent Summary: v1.7 (adds modulation gating: `modulation_min_episodes`, logging key `Risk/ModulationActive`)

### 13.1 Modulation Gating (New in v1.7)
Purpose: prevent early, noisy risk predictions from influencing Lagrange multiplier adaptation. A minimum number of fully completed episodes (`modulation_min_episodes`) must elapse before any scaling (<1) is applied. Until the gate opens, `Risk/ModulationScale` will log 1.0, ensuring baseline λ dynamics during the high-variance cold start phase.

Rationale:
* Early embedding/cost statistics are sparse and high variance; premature down/up-scaling of λ learning rate can amplify constraint oscillations.
* Episode count is a simple, environment-agnostic proxy for data coverage. Future iterations may incorporate a *quality gate* (e.g., minimum risk head correlation or minimum buffer size) plus EMA smoothing of the scale.

Planned Extensions:
* Add optional `modulation_quality_min_corr` threshold.
* Exponential moving average over raw modulation scale to reduce jitter.
* Per-environment gating once per-env risk buffers land.

---
For continuity—update incrementally with each semantic feature change (do not delete).
