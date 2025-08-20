# Project Progress & Design Compendium (v1.9)

Comprehensive, self-contained record of the semantic-guided safe RL effort (PPOLagSem) up to document version v1.9.

---
## 1. Objective
Enhance PPO-Lagrangian sample efficiency and cost constraint stability by injecting frozen vision-language semantic priors (CLIP) and a learned auxiliary risk predictor, while preserving baseline optimality (neutrality) and keeping all features *opt-in* and incrementally extensible.

Key success criteria:
- Baseline parity when semantics disabled (identical learning curves within stochastic variance).
- Early training acceleration (faster approach to target return) without higher constraint violation.
- Stable or reduced variance of episodic costs post-convergence (with modulation enabled).
- Overhead < ~10–15% wall clock at default settings.

---
## 2. High-Level Architecture
| Layer | Component | Responsibility |
|-------|-----------|----------------|
| Algorithm | `PPOLagSem` | Wrap baseline PPO-Lag; optionally swap rollout adapter; call modulation hook. |
| Adapter | `SemanticOnPolicyAdapter` | Capture frames (spatial batch), apply shaping, hand off to buffer. |
| Semantics Core | `SemanticManager` | CLIP load, prompt centroids, margin normalization, beta schedule, risk buffers, modulation scale. |
| Auxiliary | `SemanticRiskHead` | Predict truncated discounted future cost (episode-aware). |
| Modulation | (in algorithm update) | Scale λ optimizer LR via risk quantile gating. |
| Offline Probe | `offline_clip_semantic_probe.py` | Prompt set discrimination diagnostics, temporal smoothing experiments. |

---
## 3. Feature Chronicle
| Version | Feature | Rationale | Impact |
|---------|---------|-----------|--------|
| v1.0 | Semantic scaffolding (manager, adapter) | Isolate semantics from PPO core | Safe toggling, minimal diff |
| v1.1 | Logging & dtype/device separation | Observability + perf | Faster embedding, debugging clarity |
| v1.2 | Formal math + shaping normalization | Theoretical transparency | Easier ablation / neutrality claims |
| v1.5 | Spatial batching | Remove per-env signal dilution | Higher per-step semantic coverage |
| v1.6 | Telemetry expansion (margins, beta, clamp) | Diagnose shaping influence | Early detection of mis-scaling |
| v1.7 | Potential-based shaping mode | Neutral shaping option | Reduced long-term bias risk |
| v1.8 | Backward risk target rollout | Lower variance targets | Better correlation early |
| v1.9 | Episode-aware risk masking + λ modulation | Prevent leakage; adaptive constraint tuning | Lower λ oscillation risk |

---
## 4. Semantic Pipeline Data Flow
1. Frame batch (N envs) captured every `capture_interval` steps.
2. CLIP embeddings (bf16→fp16→fp32 fallback) → L2-normalized.
3. Margin = cos(e, safe_centroid) - cos(e, unsafe_centroid).
4. (Optional) Normalize (running z), scale (`margin_scale`), clamp to [-2,2].
5. Shaping term: additive or potential-based with cosine-annealed β.
6. Rewards adjusted before buffer store.
7. Embeddings + costs + done flags stored in ring buffers (length `window_size`).
8. PPO update completes; risk head trains (single batch) → predictions inform modulation scale at next λ update.

---
## 5. Parameter & Flag Compendium
| Parameter | Location | Type | Default | Purpose | Notes / Intuition |
|-----------|----------|------|---------|---------|------------------|
| enable | semantic_cfgs | bool | False | Master switch | Hard gate for parity. |
| capture_interval | semantic_cfgs | int | 4 | Steps between captures | Trade latency vs signal density. |
| frame_size | SemanticConfig | int | 224 | Resize for CLIP | Aligns with model pretrain. |
| model_name | SemanticConfig | str | clip-vit-base-patch16/32 | CLIP backbone | Frozen; deterministic. |
| model_device | SemanticConfig | str | cuda:0 | Device for CLIP | Keeps heavy ops off CPU. |
| host_device | SemanticConfig | str | cpu | Storage & shaping compute | Cheap ops / memory locality. |
| beta_start | SemanticConfig | float | 0.15 | Initial shaping weight | Early curriculum boost. |
| beta_end_step_fraction | SemanticConfig | float | 0.7 | Fraction of total steps to decay β→0 | Ensures asymptotic neutrality. |
| shaping_enable | semantic_cfgs | bool | False | Enable reward shaping | Off for parity baselines. |
| potential_enable | semantic_cfgs | bool | False | Use potential difference shaping | Bias-free variant. |
| margin_norm_enable | semantic_cfgs | bool | True | Z-normalize margin | Cross-run scale stability. |
| margin_scale | semantic_cfgs | float | 1.0 | Amplify raw margin pre-normalization | Compensate weak prompt separability. |
| risk_enable | semantic_cfgs | bool | False | Train risk head | Auxiliary representation shaping. |
| risk_lr | semantic_cfgs | float | 1e-3 | Risk head optimizer LR | Decoupled tuning. |
| risk_horizon | semantic_cfgs | int | 64 | Truncation length | Balance bias/variance. |
| discount | semantic_cfgs | float | 0.99 | Risk target discount | Often aligns with cost discount. |
| risk_episode_mask_enable | semantic_cfgs | bool | True | Reset accumulation at terminals | Prevent cross-episode leakage. |
| window_size | semantic_cfgs | int | 2048 | Buffer size for embeddings/costs | Memory/time trade. |
| norm_window | semantic_cfgs | int | 1000 | Running margin stats window | Avoid old distribution drift. |
| batch_across_envs | semantic_cfgs | bool | True | Spatially batch env frames | Per-env shaping & efficiency. |
| batch_max | semantic_cfgs | int | 32 | Cap on batch size | Guard VRAM & latency spikes. |
| oom_backoff | semantic_cfgs | bool | True | Halve batch recursively on OOM | Fail-soft reliability. |
| alpha_modulation | semantic_cfgs | float | 2.0 | λ LR scaling aggressiveness | Higher → stronger damping. |
| threshold_percentile | semantic_cfgs | int | 60 | Quantile for tail risk | Mid-tail sensitivity anchor. |
| slope | semantic_cfgs | float | 5.0 | Scale denominator for logistic | Controls sharpness. |
| modulation_enable | semantic_cfgs | bool | False | Activate λ modulation | Keep off for clean ablation. |

Derived / Internal:
- β schedule: cosine decay β_t = β_0 * 0.5 * (1 + cos(π * min(1, t/T_end))).
- Modulation scale: scale = 1 / (1 + α * σ((mean_pred - q_α)/scale_factor)).
- scale_factor ≈ perc/slope (guarded) → keeps argument magnitude moderate.

---
## 6. Design Rationale (Key Choices)
| Choice | Alternatives | Rationale |
|--------|-------------|-----------|
| Frozen CLIP | Fine-tuned CLIP / adapter | Reduce complexity; leverage general semantics; reproducible. |
| Cosine margin (safe-unsafe) | Separate sims (2-dim) | Single scalar simplifies shaping & normalization. |
| Running z-norm + clamp | Static scaling | Adapts to prompt set variance; prevents outlier spikes. |
| Cosine β decay | Linear / exponential | Smooth early emphasis; gentle tail. |
| Additive FIRST, potential optional | Only potential | Additive faster early shaping; potential ensures neutrality option. |
| Truncated discounted risk (no bootstrap) | Bootstrapped value tail | Simplicity, no critic dependency. |
| Backward construction + forward clamp | Only forward windows | O(L) vs O(LH); more stable variance. |
| Episode masking | Ignore terminals | Prevents target leakage across resets. |
| Single full-batch risk update | Mini-batch epochs | Fast iteration, acceptable for small buffer; add later if needed. |
| LR modulation (quantile gap) | Direct λ scaling via difference | Non-invasive (temp lr change) preserves optimizer state. |
| Logistic gating | Linear scaling | Bounded (0,1), robust to outliers. |

---
## 7. Modulation Scale Interpretation
scale ∈ (1/(1+α), 1]. With α=2 ⇒ (≈0.33, 1].
- Near 1.0: high tail risk relative to mean (keep adaptation strong).
- Mid ~0.5: neutral regime.
- Toward 0.33: widespread elevated predictions ⇒ damp adaptation to avoid overshoot.
Potential improvements: EMA smoothing, symmetric up/down scheme, percentile pair (q_high - q_low) dispersion metric.

---
## 8. Telemetry & Diagnostics
| Key | Meaning | Action if Anomalous |
|-----|---------|--------------------|
| Semantics/Shaping | Mean shaping term | If > ~0.2 * reward early → reduce β or scale. |
| Semantics/ShapingRewardRatio | Influence magnitude | Rising late in training → shorten decay. |
| Semantics/NormMargin | Post-normalization margin | Flat ~0 → weak prompt separation; adjust prompts. |
| Semantics/ClampFrac | Fraction clipped | >0.3 early → margin_scale too large or prompts extreme. |
| Semantics/Beta | Curriculum position | If decays too soon → increase `beta_end_step_fraction`. |
| Risk/Loss | Aux convergence | Plateau high → adjust `risk_lr` or horizon. |
| Risk/Corr | Predictive utility | <0.1 after warmup → horizon/prompt mismatch. |
| Risk/ModulationScale | λ lr factor | Stuck extremes → tune percentile/slope/α. |
| Semantics/EmbedLatencyMs | Embedding cost | Too high → raise capture_interval / reduce batch_max. |

---
## 9. Current Limitations
1. λ modulation heuristic unsmoothed (potential jitter).
2. Risk targets still biased by horizon truncation (no bootstrap tail).
3. No integration of semantic embeddings into policy/critics (late fusion deferred).
4. Lack of semantic-specific unit tests (margin sign, beta schedule, risk correlation).
5. Prompt engineering manual; no automated pruning/scoring pipeline yet.
6. No temporal micro-batching or async capture (may benefit CPU-bound envs).

---
## 10. Remaining High-Level Tasks
| Priority | Task | Goal | Notes |
|----------|------|------|-------|
| P1 | Unit tests suite | Guard correctness & regressions | Focus margin, β schedule, masking, modulation finiteness. |
| P1 | Config & env snapshot export | Reproducibility | Hash prompts, record torch/transformers versions. |
| P2 | Modulation smoothing (EMA) | Reduce scale noise | Keep responsiveness via dual-timescale. |
| P2 | Risk target bootstrap option | Lower bias | Mix with learned cost critic tail. |
| P2 | Automated prompt scoring | Improve signal quality | Use offline probe stats (margin variance, separation angle). |
| P3 | Mini-batch risk training | Better convergence | Only if loss plateau persists. |
| P3 | Temporal micro-batching | Efficiency on low env counts | Buffer frames then batch. |
| P4 | Embedding-policy fusion | Potential performance lift | Requires arch & obs normalization changes. |
| P4 | Async embedding worker | Hide latency | Complexity vs marginal gain. |

---
## 11. Risk & Mitigation Matrix
| Risk | Exposure | Mitigation |
|------|----------|-----------|
| Over-shaping biases policy | Early strong β | Cosine decay to 0; ratio telemetry; potential mode. |
| Prompt irrelevance | Poor margin separability | Offline probe iteration, prompt pruning. |
| Modulation instability | No smoothing | Add EMA, clamp percentile shifts. |
| Memory growth | Large window_size | Ring buffers with maxlen; configurable window. |
| Numerical issues (NaN scale) | Extreme preds / division | Guard rails & fallbacks (return 1.0). |

---
## 12. Validation Plan
1. Parity runs (`enable=False`) across 3 seeds → compare returns & costs (KS test / overlapping CIs).
2. Shaping-only vs baseline early-phase (first 20% steps) learning speed (steps-to-threshold metric).
3. Risk-only: monitor correlation & loss drop; ensure overhead within target.
4. Modulation ablation: variance of λ and episodic cost with vs without modulation.
5. Prompt sweeps: aggregate time-to-threshold & shaping ratio curves.

---
## 13. Implementation Health Checklist (Current)
| Item | Status |
|------|--------|
| Baseline parity path | Verified manual (needs automated test) |
| Shaping additive | Stable |
| Shaping potential | Stable |
| Spatial batching | Stable |
| Episode masking | Stable |
| Risk backward targets | Stable |
| λ modulation | Experimental (guarded) |
| Offline probe | Stable |
| Unit tests | Pending |
| Snapshot export | Pending |

---
## 14. Quick Usage Recipes
Baseline vs parity:
```
python examples/train_policy.py --algo PPOLag --env-id SafetyCarGoal1-v0 --total-steps 500000
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal1-v0 --total-steps 500000 --semantic-cfgs:enable False
```
Shaping + risk + modulation trial:
```
python examples/train_policy.py \
  --algo PPOLagSem --env-id SafetyCarGoal1-v0 \
  --semantic-cfgs:enable True \
  --semantic-cfgs:shaping_enable True \
  --semantic-cfgs:risk_enable True \
  --semantic-cfgs:modulation_enable True \
  --semantic-cfgs:potential_enable False
```

---
## 15. Future Exploration Ideas
- Dual-head (reward/cost) semantic embedding fine-tune via lightweight adapter (LoRA) while keeping base CLIP frozen.
- Contrastive risk pair mining (high vs low future cost frames) to sharpen predictor.
- Adaptive β schedule conditioned on observed shaping reward ratio trajectory.
- Prompt evolution via genetic or Bayesian optimization loop over offline probe metrics.
- Cross-environment semantic transfer (evaluate prompt generality).

---
## 16. Changelog vs Baseline PPO-Lag (Code Points)
| Area | Added / Modified | File(s) |
|------|------------------|---------|
| Semantic config block | New fields & toggles | `PPOLagSem.yaml` |
| Adapter swap logic | Conditional env adapter | `ppo_lag_semantic.py` |
| CLIP management & shaping | New class | `semantic_manager.py` |
| Spatial batching capture | Batch embedding path | `semantic_onpolicy_adapter.py` |
| Risk head & training hook | Aux MLP + update segment | `policy_gradient.py`, `risk_head.py` |
| Episode-aware risk masking | Target construction | `policy_gradient.py`, `semantic_manager.py` |
| Lagrange modulation | LR scaling wrapper call | `ppo_lag_semantic.py`, `semantic_manager.py` |
| Telemetry keys | Logging additions | multiple (logger registration) |
| Offline probe script | Diagnostics | `offline_clip_semantic_probe.py` |

---
## 17. Glossary
| Term | Definition |
|------|------------|
| Shaping | Additional reward term guiding early exploration (annealed). |
| Potential-based shaping | Difference of discounted potentials ensuring neutrality. |
| Margin | Safe vs unsafe centroid cosine similarity difference. |
| β schedule | Time-varying shaping coefficient. |
| Risk target | Truncated (episode-aware) discounted future cost estimate. |
| Modulation scale | Factor adjusting λ optimizer learning rate. |

---
## 18. Document Version
v1.9 (progress.md snapshot aligned with thesis v1.9)

---
End of progress compendium.
