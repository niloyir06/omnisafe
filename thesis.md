# Semantic-Guided Safe RL Extension (PPOLagSem)

## 1. Overview
This document records the integration of semantic (vision-language) guidance into the OmniSafe PPO-Lagrangian baseline, resulting in a new algorithm variant: **PPOLagSem**. The aim is to accelerate safe policy learning (sample efficiency) and improve constraint satisfaction stability by injecting CLIP-based semantic priors and an auxiliary future-cost predictor, without destabilizing core PPO-Lagrangian optimization.

## 2. Motivation
Traditional safe RL (PPOLag) relies purely on scalar reward/cost signals and state features. High-level scene semantics (e.g., “imminent collision” vs “clear path”) are implicitly learned, which can be slow or noisy in sparse/delayed feedback settings. A frozen vision-language encoder (CLIP) already encodes general semantic structure. By:
- Mapping frames → embeddings → semantic safety margin, and
- Providing a shaped (annealed) reward bonus aligned with safety/goal priors,
we bias exploration toward safer, progress-relevant behaviors early in training. An auxiliary **risk head** further grounds embeddings in future discounted cost prediction, fostering a latent representation aligned with safety.

## 3. Summary of Implemented Changes
| Area | Change | Rationale | Impact |
|------|--------|-----------|--------|
| Algorithm Variant | Added `PPOLagSem` subclass | Isolate semantic logic; keep baseline untouched | Opt-in, minimal regression risk |
| Adapter | New `SemanticOnPolicyAdapter` | Insert frame capture + shaping before buffer store | No core buffer redesign |
| Semantics Core | `SemanticManager` | Centralize CLIP loading, prompt embedding, shaping, risk stats | Encapsulation, easy future extensions |
| Auxiliary Model | `SemanticRiskHead` | Predict discounted future cost from embeddings | Regularizes embedding & enables modulation later |
| Config | Added `semantic_cfgs` block (defaults off) in PPOLag & PPOLagSem YAML | Reproducibility & CLI overrides | Zero-change baseline parity when disabled |
| Dependency | Added `transformers` | Access CLIP model & processor | Enables semantic features |
| Training Hook | Risk head update inside `PolicyGradient._update()` (guarded) | Reuse existing update loop | No schedule disruption |
| Logging | Shaping magnitude, raw reward, latency (ms), risk loss, risk target mean, prediction correlation | Deeper diagnostics & attribution | Transparent evaluation & ablation clarity |

## 4. Detailed Components
### 4.0 Mathematical Formulation (Addendum)
NOTE: GitHub Markdown renders math more reliably using `$...$` for inline and `$$...$$` for display (rather than `\( ... \)` / `\[ ... \]`). Converted accordingly.

This section formalizes the modifications introduced by semantic guidance. Symbols:
$s_t$: state, $a_t$: action, $r_t$: environment reward, $c_t$: instantaneous cost, $\gamma$: reward discount, $\gamma_c$: cost discount (often $=\gamma$), $\theta$: policy parameters, $\lambda$: Lagrange multiplier, $d$: cost limit, $T$: total timesteps, $H$: risk (truncation) horizon.

#### 4.0.1 Constrained Objective
$$
\max_{\theta} \; J(\theta) = \mathbb{E}_\pi \Big[ \sum_{t=0}^{\infty} \gamma^{t} r_t \Big] \quad \text{s.t.} \quad J_C(\theta)= \mathbb{E}_\pi \Big[ \sum_{t=0}^{\infty} \gamma_c^{t} c_t \Big] \le d.
$$

#### 4.0.2 Lagrangian Relaxation (PPO-Lagrange)
Ignoring the constant $\lambda d$ during gradient steps, the per-timestep shaped advantage surrogate becomes
$$
\mathcal{L}_{\text{PPO-Lag}}(\theta) = \mathbb{E}_t \Big[ \min( r_t(\theta) \hat A_t^{R,\lambda}, \; \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon) \hat A_t^{R,\lambda}) \Big],
$$
with importance ratio $r_t(\theta)= \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and combined reward-cost advantage
$$
\hat A_t^{R,\lambda} = \hat A_t^{R} - \lambda \hat A_t^{C}.
$$
Standard value losses (reward and cost critics) and entropy regularization are added as in PPO.

#### 4.0.3 Semantic Reward Shaping (Additive vs Potential-Based)
Let a frozen vision-language encoder map an observation (frame) $x_t$ to an embedding $e_t \in \mathbb{R}^d$. Safe and unsafe prompt sets $\mathcal{P}_{safe}, \mathcal{P}_{unsafe}$ are encoded; their (optionally normalized) centroids are $\mu_{safe}, \mu_{unsafe}$. Define the semantic margin
$$
m_t = \cos(e_t, \mu_{safe}) - \cos(e_t, \mu_{unsafe}).
$$
Maintain running mean/variance $\mu_m, \sigma_m^2$ to obtain a normalized margin $\tilde m_t = (m_t - \mu_m)/(\sigma_m + \varepsilon)$. Apply global scaling (config `margin_scale`) then optional clipping $\bar m_t = \operatorname{clip}(\tilde m_t, -M, M)$ when normalization enabled; if normalization disabled we operate directly on the (scaled) raw margin before clipping.

Cosine annealed shaping coefficient (up to an $\alpha$ fraction of total steps $T_{anneal}=\alpha T$):
$$
\beta_t = \beta_0 \cdot \tfrac{1}{2} \Big(1 + \cos\big(\pi \cdot \min(1, t / T_{anneal})\big) \Big).
$$

We implement two shaping modes:

1. Additive (legacy / default when `potential_enable=False`):
$$
r'_t = r_t + \beta_t \bar m_t.
$$
2. Potential-based (when `potential_enable=True`): define a potential function $\phi(s_t)=\bar m_t$ (post normalization & scaling) captured only on semantic capture steps. Using $\gamma$ (reward discount), shaping term per transition:
$$
F_t = \beta_t (\gamma \phi(s_{t+1}) - \phi(s_t)).
$$
Reward becomes $r'_t = r_t + F_t$. This preserves optimal policies exactly if $\beta_t$ stabilizes (or decays to 0) because potential differences telescope in episodic returns; here $\beta_t$ also anneals, further ensuring neutrality. We cache previous $\phi(s_t)$ to compute $F_t$ lazily; when a capture is skipped we reuse the last potential (yielding zero shaping increment), maintaining schedule continuity.

Practical notes:
* Potential mode avoids long-horizon bias inherent in persistent additive bonuses when $\beta_t$ is non-zero for extended windows.
* Additive mode can accelerate very early learning when semantic signal is sparse; potential mode is preferred for unbiased comparisons once benefit is established.
* Both modes share the same annealing schedule, normalization toggles (`margin_norm_enable`), scaling (`margin_scale`), and capture cadence.

Diagnostic Metrics (see §4.0.7): `Semantics/ShapingRewardRatio` and `Semantics/ShapingStd` quantify relative shaping influence and dispersion to validate neutrality assumptions.

#### 4.0.4 Auxiliary Risk Prediction
Define truncated discounted cost target (sliding horizon):
$$
g_t = \sum_{k=0}^{H-1} (\gamma_c)^k c_{t+k}.
$$
Risk head $q_\phi(e_t) \approx g_t$ (parameters $\phi$, embedding detached from encoder). Loss:
$$
\mathcal{L}_{risk}(\phi) = \operatorname{SmoothL1}\big(q_\phi(e_t), g_t\big).
$$
Total optimized objective (conceptual):
$$
\mathcal{J}(\theta,\phi,\lambda)= -\mathcal{L}_{\text{PPO-Lag}}(\theta,\lambda) + w_{risk} \big(-\mathcal{L}_{risk}(\phi)\big) + w_{val} \mathcal{L}_{val} - w_{ent} \mathcal{L}_{ent}.
$$
Only $\theta$ receives gradients through policy surrogate; $\phi$ through risk head; encoder frozen.

#### 4.0.5 Lagrange Modulation (Initial Implementation)
Let empirical distribution of recent predicted (episode-aware) truncated discounted costs be $Q$. For chosen upper quantile $q_{\alpha}$ and median $q_{med}$ define
$$
z_t = \frac{ q_{\alpha}(Q) - q_{med}(Q)}{\tau},
$$
with temperature $\tau>0$. A logistic squashing yields
$$
\eta_t = \sigma(z_t)=\frac{1}{1+e^{-z_t}}.
$$
Implementation: we modulate the effective learning rate of the Lagrange multiplier optimizer each PPO update: $\text{lr}_\lambda^{eff} = \eta_t \cdot \text{lr}_\lambda$. High tail risk (large quantile gap) accelerates λ adaptation; low risk dampens oscillations. Logged as `Risk/ModulationScale`.

#### 4.0.6 Convergence Considerations
Because shaping vanishes ($\beta_t \to 0$) and auxiliary loss does not alter the final reward signal, fixed points coincide with those of PPO-Lagrange (assuming perfect critics). Auxiliary risk training influences representation quality and early policy gradients but not terminal optimality.

#### 4.0.7 New Telemetry for Shaping Influence
To ensure semantic guidance remains a gentle curriculum rather than a persistent bias, we log:

* ShapingRewardRatio: $\displaystyle \frac{\mathbb{E}[\text{shaping}]}{\mathbb{E}[r]}$ where denominator uses the raw environment reward (pre-shaping). Target band (empirical heuristic) is roughly 5–15% in the earliest 10–20% of training, decaying toward 0 with $\beta_t$.
* ShapingStd: Standard deviation of per-env shaping values within a capture batch (or 0 in scalar path). High dispersion with small mean may indicate prompt polarity inconsistencies; persistent low std with near-zero ratio can indicate insufficient semantic separability or over-aggressive normalization.

These metrics support: (a) early detection of over-shaping (ratio spikes), (b) validation that potential-based shaping collapses ratio faster than additive, (c) prompt engineering decisions (std vs mean trade-offs).

### 4.1 SemanticManager
Responsibilities:
- Lazy CLIP load (`openai/clip-vit-base-patch16`).
- Encode safe vs unsafe prompt lists → centroids.
- Compute semantic margin = cos(embed, safe_centroid) - cos(embed, unsafe_centroid).
- Normalize margin (running mean/std) and scale by cosine-annealed `beta` (decays to 0 by a fraction of total steps) to produce shaping term.
- Maintain deques of embeddings & instantaneous costs for risk head training and (future) modulation.

### 4.2 Reward Shaping Mechanism
`reward' = reward + beta * clamp(norm_margin, -2, 2)`
- Early curriculum effect (higher `beta` initially; anneals to zero).
- Statistical normalization prevents prompt-dependent scale explosion.
- Clamping guards against rare outlier similarity spikes.
- Annealing ensures asymptotic convergence target remains that of the original MDP.

### 4.3 Risk Head
- Input: frozen (no gradient into CLIP) normalized embedding.
- Output: scalar predicted discounted future cost (approximate cost-to-go over a sliding horizon `risk_horizon`).
- Loss: Smooth L1 vs rollout-built pseudo-target (discounted sum of upcoming costs within horizon window). Smooth L1 chosen for robustness to noisy tails.
- Purpose: (1) Encourage embeddings to correlate with safety signals; (2) Provide statistics for future adaptive mechanisms (Lagrange modulation, prompt evolution).

### 4.4 Adapter Swap Architecture
- Base `PPOLag` untouched.
- `PPOLagSem` calls `super()._init_env()` then conditionally replaces `_env` with `SemanticOnPolicyAdapter` if `semantic_cfgs.enable`.
- Minimizes surface area of change and eases ablation (compare `PPOLag` vs `PPOLagSem` with semantics disabled = parity test).

### 4.5 Configuration & CLI
Example activation:
```
python examples/train_policy.py \
  --algo PPOLagSem --env-id SafetyCarGoal1-v0 \
  --semantic-cfgs:enable True \
  --semantic-cfgs:shaping_enable True \
  --semantic-cfgs:risk_enable True \
  --semantic-cfgs:capture_interval 20
```
All semantic flags default to False ensuring drop-in baseline compatibility.

## 5. Differences vs Original PPOLag
| Dimension | Original PPOLag | PPOLagSem (Active) |
|-----------|-----------------|--------------------|
| Inputs to Policy | Environment observation only | Same (embedding NOT concatenated yet) |
| Reward Signal | Env reward | Env reward + transient semantic shaping term (annealed) |
| Auxiliary Loss | None | Risk head Smooth L1 (optional) |
| Lagrange Update | Standard | Optional lr scaling via risk quantile modulation |
| Compute Overhead | Base PPO-Lag | + Periodic CLIP embedding (controlled by `capture_interval`) |
| Config Surface | Standard YAML | Adds `semantic_cfgs` sub-tree |
| Stability Risk | Mature baseline | Guarded feature flags; shaping decays to zero |

## 6. Why It Can Be Better
1. **Sample Efficiency**: Early semantic margin supplies shaped gradients aligned with safety & goal semantics before cost critic learns fine distinctions.
2. **Safer Exploration**: Safe prompt centroid alignment biases policy away from high-risk visual regimes.
3. **Auxiliary Signal**: Risk head provides dense self-supervised structure on embeddings, mitigating representation collapse.
4. **Non-Intrusive**: No change to observation dimensionality or PPO core => lower risk of destabilizing existing hyperparameters.
5. **Scalable Extension Path**: Future modulation, prompt adaptation, or episodic critique can hook into existing `SemanticManager` without rewriting training loops.

## 7. Current Limitations
- Risk targets are truncated discounted sums (episode-masked) without bootstrap tail; may retain bias near horizon.
- Lagrange modulation heuristic simple (single quantile gap + logistic) and unsmoothed; temperature sensitivity uncalibrated.
- Potential domain gap: CLIP general visual pretraining may not perfectly encode environment-specific safety cues (may require prompt tuning or adapter fine-tuning).
- Additional GPU memory & initial download time for CLIP (≈600MB).

## 8. Roadmap
| Phase | Goal | Actions | Success Criteria |
|-------|------|---------|------------------|
| P0 Parity | Ensure no regression | Run PPOLag vs PPOLagSem (disabled semantics) | Matching curves (± small stochastic noise) |
| P1 Activation | Validate shaping & risk | Enable shaping & risk separately | No crashes; shaping decays; risk loss declines |
| P2 Metrics | Diagnostic depth | Log raw vs shaped reward, risk correlation | Risk pred corr > 0.3 early |
| P3 Modulation | Adaptive constraint tuning | Implement LR scaling of Lagrange via risk quantiles | (DONE) Reduced overshoot/variance of cost |
| P4 Target Refinement | Improve risk accuracy | Replace sliding horizon with reverse discounted cumulative computation | Lower risk head loss variance |
| P5 Prompt Study | Optimize semantics | Sweep prompt sets / sizes | Best set improves time-to-threshold >15% |
| P6 Efficiency | Reduce overhead | Batch embeddings, optional async | <10% FPS drop vs baseline |
| P7 Extensions | Advanced semantics | (Optional) adaptive prompt evolution, episodic critique | Documented incremental gains |
| P8 Packaging | Reproducibility | Tests + README + config snapshots | All experiments re-runnable |

## 9. Detailed Task List (Live Checklist)
Legend: [ ] pending, [~] in progress, [x] done

### Core Integrity
- [x] Add semantic config & algorithm variant
- [x] Reward shaping with annealed coefficient
- [x] Risk head auxiliary loss
- [x] Parity test (PPOLag vs PPOLagSem disabled)
- [ ] Unit tests (semantic manager, risk head, registration)

### Diagnostics & Logging
- [x] Log raw (unshaped) reward alongside shaped (Semantics/RawReward)
- [x] Log shaping term avg/min/max per epoch (Semantics/Shaping with aggregation)
- [x] Log risk prediction vs empirical discounted cost correlation (Risk/Corr)
- [ ] Log embedding FPS overhead & latency distribution (Semantics/EmbedLatencyMs done; add explicit FPS delta calc pending)

### Modulation & Targets
- [ ] Implement Lagrange LR modulation using `SemanticManager.modulation_scale`
- [ ] Replace naive risk target with on-the-fly reverse cumulative discounted sum per episode (bootstrapped at cutoffs)
- [ ] Optional: target normalization (z-score) before loss

### Prompt Engineering
- [ ] Create prompt set variants (neutral / descriptive / action-centric)
- [ ] Evaluate each on sample efficiency (steps to threshold return)
- [ ] Add configurable prompt weighting or dropout

### Performance Optimization
- [ ] GPU move + verify memory profile
- [ ] Batch embeddings every N captured frames
- [ ] Optional async queue for frame acquisition
- [ ] Add capture_interval auto-tuning based on latency budget

### Analysis Scripts
- [ ] Aggregation script: compile ep_ret, ep_cost, shaping_mean, risk_loss into CSV
- [ ] Plot scripts (learning curves, shaping schedule, risk loss, Lagrange multiplier trajectory)

### Reproducibility & Docs
- [ ] README section summarizing semantic features
- [ ] Version pin file or environment YAML snapshot
- [ ] Seed sweep (≥3 seeds) for each ablation variant
- [ ] Archive final model + config + metrics bundles

### Stretch Features
- [ ] Lagrange modulation ablation (on/off)
- [ ] Contrastive refinement of embeddings with cost-ranking pairs
- [ ] Adaptive prompt evolution pipeline
- [ ] Episodic critique summarization (text embedding fed back into shaping weight)

## 10. Evaluation Plan
Primary metrics:
- Steps to reach target return (e.g., 80% of best baseline return).
- Average episodic cost & violation rate after convergence window.
- Stability: variance of cost in final N epochs.
- Overhead: FPS delta (%) vs baseline.
Secondary:
- Risk head loss curve and prediction correlation.
- Shaping contribution decay plot.
- Lagrange multiplier oscillation amplitude comparison (post-modulation phase).
Statistical validity: 3–5 seeds, report mean ± 95% CI. Bootstrap for time-to-threshold distributions.

## 11. Potential Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Over-shaping biases optimal policy | Anneal coefficient to 0; log raw vs shaped returns |
| CLIP irrelevance to environment | Curate prompts; possibly fine-tune linear adapter (future) |
| Latency overhead harms sample throughput | Increase `capture_interval`, batch or async encode |
| Risk head noise destabilizes modulation (future) | Use robust quantiles & smoothing window |
| Prompt leakage (too general) | Domain-specific refinement with environment objects / hazards |

## 12. Future Extensions (Beyond Current Scope)
- Multi-view or temporal aggregation (encode short frame stack).
- Cost attribution heatmaps via Grad-CAM on CLIP vision encoder for explainability.
- Multi-modal fusion if language-based scenario descriptions become available.
- RLHF-style preference model using semantic embeddings for post-hoc policy ranking.

## 13. Reproducibility Notes
- All new features fully disabled when `semantic_cfgs.enable=False` (baseline preserved).
- CLIP download version fixed by `model_name` string; consider hashing downloaded model for archival.
- Ensure to archive: `config.json`, `progress.csv`, semantic prompt lists, and `transformers` version.

## 14. Immediate Next Action Suggestions
1. Run parity: `PPOLagSem` with all semantic flags false vs `PPOLag` (quick 50k–100k steps).
2. Enable shaping only—collect shaping mean/std and check early return acceleration.
3. Enable risk only—ensure risk loss > decreases & does not inflate runtime >10%.
4. Implement Lagrange modulation once risk predictions are stable.

---
## 15. Implementation Change Log & Technical Deep Dive

This section enumerates the concrete code-level changes, device/dtype design, and rationale accumulated during iterative development (Aug 2025 integration cycle). It complements the conceptual sections above with engineering specifics for reproducibility and future maintenance.

### 15.1 Chronological Change Summary
| Order | Change | Files / Areas | Motivation | Notes |
|-------|--------|--------------|-----------|-------|
| 1 | Introduced semantic pipeline scaffold | `semantic_manager.py`, new config block | Add CLIP-based shaping & risk infra | Initially single-device, fp32 centroids |
| 2 | Added `SemanticOnPolicyAdapter` | `semantic_onpolicy_adapter.py` | Capture frames at interval, inject shaping | Includes progress bar & semantic logging counters |
| 3 | Reward shaping + running z-normalization | `semantic_manager.py` | Stable scale, curriculum via cosine decay | Clamped to [-2,2] to bound influence |
| 4 | Risk head MLP | `risk_head.py`, hook in `policy_gradient.py` | Predict short-horizon discounted cost | Smooth L1 loss chosen for robustness |
| 5 | Logging keys (semantic + risk) | `_init_log` in `policy_gradient.py` | Observability, ablation transparency | Keys: Shaping, RawReward, EmbedLatencyMs, etc. |
| 6 | Safetensors enforcement for CLIP load | `semantic_manager.py`, `scripts/check_clip.py` | Security & load performance | Status codes record fallback cause |
| 7 | Added CLIP readiness & status metrics | `semantic_manager.py` | Diagnose load failures, dtype fallback visibility | Status forms: `ok_st_bfloat16`, `load_error:*` |
| 8 | Device separation (`model_device` vs `host_device`) | `semantic_manager.py`, YAML configs | Keep heavy CLIP on GPU, policy possibly CPU | Enables hybrid CPU policy + GPU semantics |
| 9 | Added config keys `model_device`, `host_device` | `PPOLag.yaml`, `PPOLagSem.yaml` | Pass recursive config validation, avoid KeyError | Defaults: model_device=cuda:0, host_device=cpu |
|10 | Fixed logger registration order (ClipStatus) | `policy_gradient.py` | Prevent missing key assertion | Registered before first store |
|11 | Embedding capture gating & counters | `semantic_onpolicy_adapter.py` | Performance control, success rate metric | Counters: attempts, successes, success rate |
|12 | Dtype fallback chain (bf16→fp16→fp32) | `semantic_manager.py` | Use fastest safe precision | Records first successful precision in status |
|13 | Risk head moved to semantic model device | `policy_gradient.py` | Avoid CPU↔GPU ping-pong when policy on CPU | Reduces transfer latency |
|14 | Mixed precision alignment & cast guard | `policy_gradient.py` | Fix matmul dtype mismatch Float vs BFloat16 | Embeddings cast to risk head param dtype |
|15 | Embedding storage optimization | `semantic_manager.py` | Avoid unnecessary `.cpu()` when host is GPU | Minimizes transfers for large windows |
|16 | Centroid precision policy | `semantic_manager.py` | Keep reduced precision on GPU, cast only on CPU | Saves memory/bandwidth when GPU host |
|17 | Added robust error fallback labeling | `semantic_manager.py` | Distinguish safetensors absence vs other errors | Easier debugging & reproducibility |
|18 | Updated thesis with formal math & deep dive | `thesis.md` | Documentation completeness | Version bump |
|19 | Episode-aware risk target masking | `semantic_manager.py`, `policy_gradient.py` | Avoid cross-episode leakage in targets | Flag `risk_episode_mask_enable` |
|20 | Lagrange modulation (lr scaling) | `ppo_lag_semantic.py` | Adaptive constraint responsiveness | Log key `Risk/ModulationScale` |

### 15.2 Device & Data Flow Specification
```
Frame (CPU, uint8) ──► (optional resize) ──► CLIP Processor (host→model_device)
  └─ tokens / pixel batch (model_device, chosen dtype: bf16/fp16/fp32)
      └─► CLIP Vision Encoder → Image Embedding e (model_device, same dtype)
          └─ normalize(L2) → move to host_device (no cast if GPU host)
              ├─ If host_device==cpu: cast → float32
              ├─ Reward shaping: use e (detached) + centroids (host_device)
              └─ Risk buffer append: e stored (host_device resident)

Prompts (text) ──► Tokenize → CLIP Text Encoder (model_device) → embeddings → mean → centroids
Centroids: kept in reduced precision if on GPU; cast to float32 if on CPU.

Risk Head Training Loop:
  Recent stacked embeddings (host_device) → to risk_head.device → dtype match (cast if needed) → forward
  Costs deque → float32 tensor (risk_device) → truncated discounted targets → loss (Smooth L1 in float32)
  Backprop only through risk head (embeddings detached upon record).
```

### 15.3 Dtype Policy Matrix
| Component | Preferred Dtype | Fallbacks | Cast Points | Rationale |
|-----------|-----------------|-----------|-------------|-----------|
| CLIP model weights | bfloat16 | fp16 → fp32 | Auto at load attempt | Speed + dynamic range |
| Image embeddings (GPU host) | bfloat16/fp16 | fp32 | Cast to fp32 only if moved to CPU | Reduce memory & bandwidth |
| Centroids (GPU host) | bfloat16/fp16 | fp32 | Same rule as embeddings | Keep similarity operations consistent |
| Centroids (CPU host) | float32 | — | Computed then cast | Numeric stability in shaping margin |
| Risk head parameters | Match CLIP dtype if reduced; else float32 | — | Explicit `.to(dtype=clip_dtype)` | Align matmul, minimize cast cost |
| Risk targets (discounted costs) | float32 | — | Constructed as float32 | Stability for loss & correlation |
| Core PPO tensors (obs, act, adv, values) | float32 | — | Native | Standard RL practice |
| Test reference values | float64 | — | None | Precise numerical assertions |

### 15.4 Error Classes & Handling
| Error Source | Symptom | Mitigation | Logged Status |
|--------------|---------|-----------|---------------|
| Missing GPU | Fallback to CPU compute, slower embeddings | `model_device` auto-switched to `cpu` | `fallback_cpu_no_cuda` |
| No safetensors weights | Load failure on bf16/fp16 attempts | Break after final attempt; mark error | `load_error:NoSafeTensorsWeights` |
| Dtype mismatch (Float vs BFloat16) | Runtime matmul exception in risk head | Cast embeddings to risk head dtype | — (prevented) |
| Prompt tokenization/device mismatch | Embedding None or crash | Uniform `.to(model_device)` mapping | Implicit via non-ready status |

### 15.5 Logging Key Glossary (Extended)
| Key | Type | Source | Description |
|-----|------|--------|-------------|
| `Semantics/Shaping` | Scalar (mean/min/max) | `SemanticManager.shaping_term` | Applied shaping value per step aggregated per epoch |
| `Semantics/RawReward` | Scalar | Adapter before shaping | Baseline reward for attribution |
| `Semantics/EmbedLatencyMs` | Scalar dist | Embedding timing | Time (ms) to produce a CLIP embedding |
| `Semantics/EmbedSuccessRate` | Ratio | Counters | successes / attempts over window |
| `Semantics/Debug/EmbedAttempts` | Counter | Adapter | Total attempted embedding calls |
| `Semantics/Debug/EmbedSuccess` | Counter | Adapter | Successful embedding extractions |
| `Semantics/RawMargin` | Scalar | SemanticManager | Last raw cosine margin (post scale) |
| `Semantics/NormMargin` | Scalar | SemanticManager | Z-normalized margin (if enabled) |
| `Semantics/Beta` | Scalar | Schedule | Current shaping coefficient |
| `Semantics/ClampFrac` | Ratio | SemanticManager | Fraction of margins historically clipped |
| `Semantics/CaptureCount` | Counter | SemanticManager | Number of capture events |
| `Semantics/CaptureIntervalEffective` | Scalar | SemanticManager | Steps since last capture |
| `Risk/Loss` | Scalar | Risk head train loop | Smooth L1 loss |
| `Risk/PredMean` | Scalar | Risk head output | Mean predicted discounted cost |
| `Risk/TargetMean` | Scalar | Computed targets | Mean target truncated discounted cost |
| `Risk/Corr` | Scalar (optional) | Correlation computation | Pearson correlation prediction vs target |
| `Risk/ModulationScale` | Scalar | Modulation routine | Scale applied to Lagrange optimizer lr |

### 15.6 Performance Considerations
| Aspect | Action | Effect |
|--------|--------|--------|
| Embedding frequency | `capture_interval` control | Linear tradeoff latency vs semantic density |
| Precision selection | Early bf16 attempt | Lower memory + faster matmul vs fp32 |
| Host/device split | Keep CLIP on GPU only | Avoid saturating CPU, reduce wall time per epoch |
| Casting minimization | Deferred float32 cast until CPU boundary | Cuts memory traffic and cast overhead |
| Risk batch window | Configurable deque length | Bounded memory growth |

### 15.7 Future Engineering Tasks (Precision / Devices)
1. Add config flag `preserve_embedding_dtype` to optionally keep reduced precision even when host is CPU (with guard rails for numerical ops).
2. Optional mixed precision autocast context around risk head forward to future-proof if head deepens.
3. Asynchronous embedding prefetch using a lightweight worker when environment supports frame duplication.

### 15.8 Reproducibility Checklist (Engineering)
- [x] Deterministic CLIP weights via model id string.
- [x] Deterministic prompt lists recorded in config.
- [x] Dtype & device status persisted in logs via `ClipStatus`.
- [ ] Pin exact `transformers` + `torch` versions in experiment metadata export.
- [ ] Add semantic unit tests: (a) load & status, (b) shaping margin sign flip with swapped prompt sets, (c) risk head correlation sanity on synthetic monotonic costs.

### 15.9 Lessons Learned
1. Mixed precision requires explicit alignment for auxiliary heads—automatic casting is not always applied in linear layers with mismatched dtypes.
2. Decoupling model vs host devices early simplified subsequent optimization (embedding storage policy adjustments were localized).
3. Logging categorical status strings (`ok_st_bfloat16`) drastically shortened debugging cycles vs boolean success flags.
4. Early enforcement of safetensors avoided silent fallback to slower legacy formats and tightened supply chain integrity.

---
*Document Version: v1.9 (includes episode-aware risk masking & initial Lagrange modulation; supersedes earlier addenda v1.4–v1.8).* 

---
## 26. v1.5 Addendum: Spatial Batching Implementation

This addendum documents the spatial batching feature added after v1.4.

### 26.1 Summary
- Added batched embedding path: collect a frame per parallel env at each capture step and embed as a batch (configurable).
- New config keys: `batch_across_envs` (enable/disable), `batch_max`, `oom_backoff`.
- Adapter now applies per-env shaping vector instead of broadcasting a single scalar.
- Risk buffer now receives up to N embeddings per capture (N = number of envs, capped by `batch_max`).

### 26.2 Rationale
Single-frame capture diluted semantic signal for large `vector_env_nums` (coverage 1/N). Spatial batching restores per-env fidelity and improves statistical efficiency for the auxiliary risk predictor.

### 26.3 Implementation Details
SemanticManager:
1. New method `maybe_compute_embeddings(frames)` handles a list of frames.
2. Performs a single CLIP forward; on CUDA OOM (detected by substring match) optionally halves the batch recursively if `oom_backoff=True`.
3. Records average per-frame latency (total / batch_size) for fair comparison with single-frame path.

Adapter Changes:
1. On capture step, gathers frames (replicates single render if environment does not return per-env list).
2. Obtains list of embeddings; computes shaping per embedding when `shaping_enable=True`.
3. Builds shaping tensor aligned with reward tensor and applies element-wise.
4. Records embeddings & per-env instantaneous costs via `record_multi_step` (global semantic step increments once).
5. Maintains legacy single-frame path when `batch_across_envs=False` for clean A/B tests.

### 26.4 Reward Shaping (Vectorized)
For each environment i in the batch:
`r'_i = r_i + beta_t * clamp(norm_margin_i, -2, 2)`.
All environments share the same annealed `beta_t`; normalization statistics shared globally (running window).

### 26.5 Compatibility & Parity
- When `enable=False` or `batch_across_envs=False`, behavior reverts to pre-v1.5 (scalar shaping broadcast or none).
- Global step advancement unchanged (one increment per environment step) so prior annealing schedules remain valid across experiments.

### 26.6 Performance Considerations
- Batch size bounded by `batch_max` to prevent excessive VRAM spikes.
- Empirically expected to reduce total embedding wall time per environment step for moderate N due to parallelization.
- OOM backoff ensures graceful degradation to smaller batches instead of hard failure.

### 26.7 Updated Roadmap Flags
Reclassify "Spatial batching" task as DONE; next performance tasks: temporal micro-batching, async capture.

### 26.8 Version Bump
Document version advanced to v1.5 to reflect spatial batching availability.

---
## 27. Updated Version History Entry
| Version | Date | Highlights |
|---------|------|------------|
| v1.5 | 2025-08-19 | Spatial batching (per-env embeddings & shaping), config keys for batched CLIP, risk buffer density increase |

---
## 16. Current Runtime Semantics Behavior (Clarification)

This section documents the exact present (pre-batching) behavior of semantic components so future diffs can be assessed precisely.

### 16.1 Frame Capture Granularity
- Only a SINGLE frame is rendered per capture event, regardless of `vector_env_nums` (comes from the vectorized env's aggregate `render()` — effectively env index 0 or a composite according to underlying wrapper).
- The resulting embedding and derived shaping scalar is broadcast (added) to the rewards of ALL parallel vector envs for that timestep.
- Consequence: Semantic signal density per environment scales as `1 / vector_env_nums` (dilution) until spatial batching is implemented.

### 16.2 Risk Buffer Population
- One embedding per capture interval is appended (not per-env). Costs deque stores the (mean) cost for that timestep.
- Risk head therefore learns from a subsampled trajectory slice; correlation ceilings may be lower than if per-env embeddings existed.

### 16.3 Parallel Processes (`train_cfgs.parallel > 1`)
- Each process repeats the above independently (duplicate CLIP load, independent normalization state). No inter-process synchronization of semantic statistics yet.

### 16.4 Practical Impact
| Aspect | Current | Future (Planned Spatial Batch) |
|--------|---------|--------------------------------|
| Embeddings / capture | 1 | `<= vector_env_nums` (capped) |
| Shaping specificity | Global scalar | Per-env scalar |
| Risk samples per epoch | O(steps / capture_interval) | × `vector_env_nums` multiplier |
| Memory per capture | Minimal | Linear in batch (bounded) |
| Variance of margin estimate | Higher (single sample) | Lower (mini-batch mean optional) |

---
## 17. Config Override & Troubleshooting Edits Not Applying

Observed issue: Editing `PPOLagSem.yaml` appeared to have no runtime effect. Root causes & diagnostics consolidated here.

### 17.1 Precedence Chain
1. YAML defaults (loaded once per process at `AlgoWrapper._init_config`).  
2. `custom_cfgs` (programmatic or CLI nested overrides).  
3. `train_terminal_cfgs` (terminal overrides for `train_cfgs`).  
4. Derived fields (epochs, exp_name).  

### 17.2 Common Pitfalls
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| YAML edit ignored | CLI flag overrides same key | Remove CLI flag or update both |
| No change after edit | Reused long-lived Python process | Restart process/kernel |
| Wrong file loaded | Different algo selected (`PPOLag`) | Use `--algo PPOLagSem` |
| Semantic key ignored | Key placed outside `semantic_cfgs` subtree | Nest under the correct block |
| Device override mismatch | Using `--semantic-cfgs:device` which supersedes `model_device` | Omit `device` to rely on YAML `model_device` |
| Still baseline behavior | `enable=False` (accidentally set via CLI) | Ensure `--semantic-cfgs:enable True` |
| Unexpected precision | Dtype fallback changed status | Check `Semantics/Debug/ClipStatus` log |

### 17.3 Verification Snippets
Add temporary debug (remove later):
```python
print('DEBUG semantic_cfgs:', self._cfgs.semantic_cfgs.__dict__)
```
At start of `PPOLagSem._init_env` to inspect merged values.

### 17.4 Safe Sentinel Test
Change `defaults.train_cfgs.seed` to a unique value (e.g., 314159) in YAML and verify the saved run `config.json` under the new log directory reflects it when no CLI seed flag is passed.

---
## 18. Batching Strategies (Design Decision Record)

Summarizes trade-offs for future implementation; original comparative analysis distilled here.

### 18.1 Spatial Batching (Across Env Instances)
Pros:
- Immediate per-env shaping (no delay) and higher semantic coverage.
- High GPU utilization; simple mapping (env index → embedding row).
- Minimal conceptual change; easy revert.
Cons:
- Larger instantaneous latency on capture steps.
- Memory peak ∝ batch size.
Code Size Estimate: ~80 LOC (manager batch method + adapter loop adjustment + config keys).

### 18.2 Temporal Batching (Across Timesteps)
Pros:
- Builds sizable batches even with few envs.
- Potential compute amortization / scheduling flexibility.
Cons:
- Shaping delay → requires retroactive reward patching or diminished immediacy.
- More bookkeeping (timestamp & buffer index mapping, episode edge cases).
Code Size: ~150 LOC + careful buffer mutation.

### 18.3 Combined Spatio-Temporal
Pros: Maximum batch sizing flexibility.
Cons: Adds complexity of both approaches; limited marginal gain after spatial batching for moderate env counts.
Code Size: 200–300 LOC.

### 18.4 Recommended Path
Phase 1: Spatial batching first (largest ROI / lowest risk).  
Phase 2: Optional temporal micro-batching only if vector_env_nums is small and CLIP remains bottleneck.  
Phase 3: Combined only if justified by profiling (document justification before implementation).

### 18.5 Proposed New Config Keys
| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `clip_batch_across_envs` | bool | True | Enable spatial batching |
| `clip_batch_max` | int | `vector_env_nums` | Upper cap to control latency/memory |
| `clip_oom_backoff` | bool | True | Auto halve batch on OOM |
| `risk_batch_size` | int | 128 | (If risk mini-batching expanded) |
| `risk_update_iters` | int | 1 | Additional risk mini-epochs per PPO epoch |

---
## 19. CPU Core & Thread Utilization Guidelines (24C / 32T Host)

| Scenario | Recommended `vector_env_nums` | `parallel` | `torch_threads` | Notes |
|----------|------------------------------|-----------|-----------------|-------|
| Default semantic shaping (single frame) | 20–24 | 1 | 1 | Avoid over-saturating; leaves headroom |
| Heavy env simulation | 12–16 | 1 | 1–2 | Reduce contention |
| Light env, GPU policy | 24 | 1 | 2 | Raise threads only if CPU <80% |
| Experimental multi-process | 10–12 per proc | 2 | 1 | Duplicate CLIP cost; only for scaling tests |

Practical Benchmark Procedure: Sweep `vector_env_nums ∈ {8,12,16,20,24}` over 10k steps; record steps/sec & CPU utilization, choose knee of curve.

---
## 20. Updated Roadmap Priorities (v1.3)

Priority ordering adjusted to reflect insights from runtime usage and profiling expectations.

1. Spatial CLIP batching (per-env embeddings)  
2. Risk head mini-batch loop (if accuracy plateau)  
3. Lagrange modulation using predicted risk quantiles  
4. Reverse discounted target computation (episode-aware)  
5. Config & log snapshot export (hash + `transformers` version)  
6. Unit tests (semantic margin sign tests; dtype fallback)  
7. Prompt set ablation harness  
8. Optional temporal batching (only if justified)  
9. Async embedding worker (stretch)  
10. Modulation ablation & report section inclusion

---
## 21. Open Technical Questions (To Track Before Major Refactors)
- Should shaping be applied per-env or aggregated (mean) for stability once spatial batching lands? (Decision pending empirical variance comparison.)
- Do we freeze centroids or allow low-frequency re-encoding for dynamic prompt weighting? (Potential drift vs stability.)
- Is there benefit from mixing policy observation with low-rank projection of embedding (late fusion) after risk head stabilizes? (Requires additional critic adaptation.)

---
## 22. Quick Reference Commands
Parity (baseline vs disabled semantics):
```
python examples/train_policy.py --algo PPOLag --env-id SafetyCarGoal1-v0 --total-steps 500000
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal1-v0 --total-steps 500000 --semantic-cfgs:enable False
```
Shaping only trial:
```
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal1-v0 --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable False
```

---
## 23. Maintenance Notes
- When adding batching: ensure per-env reward shaping added BEFORE buffer.store call; adjust logging to aggregate per-env shaping (mean) plus std for diagnostic drift.
- Keep a feature flag to force legacy single-frame path for A/B comparisons.
- Update `ClipStatus` encoding if new dtype strategies (e.g., INT8 quantized) introduced—document mapping.

---
## 24. Version History
| Version | Date | Highlights |
|---------|------|------------|
| v1.0 | Initial | Core semantic & risk integration scaffold |
| v1.1 | Added logging & device/dtype separation | Stability fixes, safetensors enforcement |
| v1.2 | Deep dive + change log | Formal math, dtype matrix, performance notes |
| v1.3 | 2025-08-18 | Runtime behavior clarifications, batching design, config troubleshooting, roadmap re-prioritized |
| v1.4 | 2025-08-19 | Executive technical summary, parity pseudocode, clarified risk head training phase |
| v1.5 | 2025-08-19 | Spatial batching (per-env embeddings & shaping), batch OOM backoff, capture count telemetry |
| v1.6 | 2025-08-19 | Telemetry expansion (RawMargin/NormMargin/Beta/ClampFrac/CaptureIntervalEffective), config additions (risk_lr, margin_norm_enable, margin_scale), removal of obsolete ClipReady/ClipStatus, CSV semantic analysis script |
| v1.7 | 2025-08-19 | Potential-based shaping implementation (`potential_enable`), shaping influence metrics (ShapingRewardRatio, ShapingStd), documentation of additive vs potential formulation, neutrality rationale |
| v1.8 | 2025-08-19 | Risk head backward (reverse) discounted target rollout implemented; clarified target construction; offline semantic probe & prompt engineering workflow documented; roadmap/status updated (risk target refinement DONE) |
| v1.9 | 2025-08-20 | Episode-aware risk target masking; initial Lagrange modulation (lr scaling); log key `Risk/ModulationScale`; config flag `risk_episode_mask_enable` |

### 24.1 Active Feature Matrix
| Feature | Status | Config Flag(s) | Notes |
|---------|--------|---------------|-------|
| Reward shaping (additive) | Stable | `shaping_enable`, `margin_norm_enable`, `margin_scale` | Annealed beta schedule |
| Reward shaping (potential-based) | Stable | `potential_enable` | Potential difference, neutrality |
| Spatial batching | Stable | `batch_across_envs`, `batch_max`, `oom_backoff` | Per-env shaping vector |
| Risk head auxiliary loss | Stable | `risk_enable`, `risk_horizon`, `discount`, `risk_lr` | Single batch update post PPO |
| Backward risk target rollout | Stable | (implicit) | Reverse pass + forward clamp |
| Episode-aware masking | Stable | `risk_episode_mask_enable` | Prevents cross-episode leakage |
| Lagrange modulation (lr scaling) | Experimental | `modulation_enable`, `threshold_percentile`, `alpha_modulation`, `slope` | Logistic quantile gap (no smoothing) |
| Offline semantic probe | Stable | (script) | Prompt separability diagnostics |
| Target bootstrap tail | Planned | — | Potential bias reduction |
| Modulation smoothing (EMA) | Planned | — | Reduce scale volatility |
| Embedding fusion into policy | Deferred | — | Requires net architecture changes |

---
## 25. Executive Technical Summary (Consolidated)

This section condenses all implemented engineering changes and current design rationale into one quick reference snapshot for reviewers and future contributors.

### 25.1 New Variant & Wiring
- `PPOLagSem` subclass wraps baseline `PPOLag`; only difference is conditional replacement of the rollout adapter with `SemanticOnPolicyAdapter` when `semantic_cfgs.enable=True`.
- Baseline parity: with `enable=False`, code path exactly mirrors PPO-Lagrange; no semantic modules initialized.

### 25.2 Semantic Pipeline Components
- `SemanticManager`: CLIP load (bf16→fp16→fp32 fallback), safetensors enforcement, prompt centroid computation, embedding extraction, semantic margin normalization (running window), cosine annealed shaping coefficient, embedding & cost deques, status + latency metrics.
- `SemanticOnPolicyAdapter`: Periodic frame capture (singleframe currently), embedding request, shaping injection, logging instrumentation, counters for attempts/success.

### 25.3 Auxiliary Risk Head
- `SemanticRiskHead` (small MLP) predicts truncated discounted future cost horizon; trained post PPO updates each epoch on stacked recent embeddings (single batch for now).
- Embeddings detached (no grad to CLIP). Loss: Smooth L1 for robustness to noisy cost spikes.

### 25.4 Device & Precision Strategy
| Element | Strategy |
|---------|----------|
| CLIP | Loaded on `model_device`; prefer bf16, else fp16, else fp32 |
| Centroids & Embeddings | Remain reduced precision on GPU; cast to fp32 only when moved to CPU |
| Risk Head | Placed on `model_device`; parameters cast to CLIP dtype if reduced |
| PPO Networks | Unchanged on `train_cfgs.device` (often CPU) in fp32 |

### 25.5 Logging & Diagnostics
Semantic keys: shaping magnitude, raw reward, latency, success rates, status codes, counters.  
Risk keys: loss, predicted mean, target mean, Pearson correlation.  
Purpose: isolate overhead, verify shaping schedule, monitor embedding quality and predictive utility.

### 25.6 Error Handling & Robustness
- Safetensors-only CLIP loading with status enumeration (`ok_st_bfloat16`, `load_error:*`, `fallback_cpu_no_cuda`).
- Dtype mismatch prevention: explicit casting of embeddings to risk head dtype.
- Graceful GPU absence fallback to CPU.
- Semantic logic fully gated; baseline path unaffected when disabled.

### 25.7 Configuration Additions
`semantic_cfgs` block (all False/off by default) enabling feature subsets: shaping, risk, future modulation; capture interval for controlling overhead; devices & precision separation; prompt lists.

### 25.8 Data Flow (Current)
1. (If enabled & capture step) Render single frame → CLIP embedding → margin → shaping scalar.
2. Shaping scalar broadcast across all env reward entries that step (pre-batching limitation).
3. Embedding + mean cost appended to deques (one sample per capture event).
4. After PPO optimization epochs, risk head (if enabled) trains on full accumulated embedding window.

### 25.9 Reward & Cost Treatment
- Shaped reward r' = r + β_t * clipped_norm_margin; β_t cosine-annealed to 0 by configured fraction of total steps ensuring eventual original objective.
- Cost pathway untouched; critics & Lagrange updates remain baseline.

### 25.10 Parity Pseudocode
Disabled semantics (`enable=False`):
```
for step in rollout:
  act = policy(obs); next_obs, r, c, done = env.step(act)
  store(obs, act, r, c, values, logp)
  obs = next_obs
```
Enabled semantics (current single-frame version):
```
if step % capture_interval == 0:
  frame = env.render(); e = clip(frame); margin = sim(e, safe) - sim(e, unsafe)
  shaping = beta(step) * norm_clip(margin); r = r + shaping
  record(e, cost_mean)
store(..., r, ...)
```
Risk head update (epoch end):
```
emb_batch = stack(recent_embeddings)
targets = truncated_discounted(costs, horizon)
loss = SmoothL1(risk_head(emb_batch), targets)
backprop(loss)
```

### 25.11 Known Limitations
- Spatial batching implemented but current description above still outlines single-frame path for historical reference; ensure active config (`batch_across_envs`) is documented per experiment.
- Risk buffer per-capture sampling may still under-represent tail events if `capture_interval` large.
- λ modulation heuristic minimal (no smoothing); embeddings not fused into policy network.
- Risk target = truncated discounted sum (episode-masked) without bootstrap tail.
- Lacking semantic-specific unit tests (planned).

### 25.12 Performance Considerations
- Overhead scaled by capture frequency; embedding latency tracked.
- Reduced precision lowers memory + increases throughput; casting deferred until needed.
- Semantic signal dilution with high `vector_env_nums` until batching implemented.

### 25.13 Immediate High-ROI Tasks
1. Spatial batching (per-env embeddings) & per-env shaping statistics.  
2. Improved risk targets (reverse discounted episode-aware).  
3. λ modulation using risk quantiles (post risk stability).  
4. Unit tests for semantic margin, dtype fallback, shaping schedule, risk correlation.  
5. Config snapshot (version hashes, prompt lists, status) in log directory.

### 25.14 Rationale for Deferring Late Fusion
- Avoid confounding effects until baseline shaping + risk predictive accuracy validated.
- Post-batching embeddings will increase per-step semantic coverage, making direct conditioning more informative and reducing variance from reused stale features.

### 25.15 Risk Head Placement & Timing Justification
- Training after PPO updates isolates auxiliary gradient signals from main policy update loop, avoiding interference with advantage sampling distribution mid-epoch.
- Single batch simplicity chosen first; later mini-batching can provide smoother convergence without structural change to PPO loop.

### 25.16 Parity Guarantee Statement
Formally: With `semantic_cfgs.enable=False`, code executes identical operations (order, tensors, gradients) as baseline PPO-Lagrange except for negligible conditional checks (branch predicate false). No additional modules, parameters, or memory allocations are instantiated.  Verified via: identical logged keys set, absence of semantic directories, and matching cumulative reward curves within stochastic variance.

---

## 29. Backward Risk Target Rollout Implementation (v1.8)

### 29.1 Motivation
Earlier versions approximated truncated discounted cost targets via forward windowed sums. We replaced this with a reverse (backward) discounted accumulation pass producing cost-to-go style targets more stable numerically and less sensitive to horizon edge effects. This improves temporal credit alignment for the auxiliary risk predictor without introducing policy gradient leakage.

### 29.2 Algorithm
Given recent instantaneous costs sequence `c[0:L]` (aligned with collected embeddings `e[0:L]`) and discount `γ_c` plus horizon `H`:
1. Reverse iterate i = L-1 → 0 accumulating `running = c[i] + γ_c * running` (standard backward return) storing preliminary target `t[i] = running` until `H` steps back reached.
2. For indices where full horizon not covered (i close to sequence start) optionally recompute a forward truncated sum to strictly enforce horizon cap, matching implementation detail in `policy_gradient.py`.
3. If `H < L`, forward refinement loop ensures each `t[i]` only includes up to `H` discounted terms, preventing overestimation in very long buffers.

### 29.3 Implementation Notes
* Executed inside `_update()` after PPO actor/critic updates (keeps auxiliary gradient step segregated).
* Operates on most recent window (`L` limited implicitly by deque length and embedding availability).
* Embeddings and costs are truncated to same trailing length `L` prior to target construction.
* Smooth L1 loss still applied; dtype alignment (embedding cast to risk head param dtype) retained.
* Correlation logging unaffected; typically exhibits higher early correlation due to reduced target variance.

### 29.4 Benefits
| Aspect | Forward Truncated | Backward Rollout (Current) |
|--------|-------------------|----------------------------|
| Numerical Stability | Medium (repeated slice sums) | High (single pass + optional refinement) |
| Horizon Enforcement | Explicit loops | Backward base + forward clamp |
| Correlation Onset | Slower (noisy tail) | Faster (smoother targets) |
| Compute Cost | O(L*H) worst | O(L + min(L,H)) |

### 29.5 Future Refinements
* Episode boundary masking (avoid leakage across terminal states when windows straddle episodes).
* Optional generalized advantage style exponential weighting variants.
* Target normalization (z-score or min-max) to stabilize loss scale across environments.

## 30. Offline Semantic Probe & Prompt Engineering Workflow

### 30.1 Purpose
An external diagnostic script (`examples/offline_clip_semantic_probe.py`) accelerates prompt iteration by quantifying centroid separability and annotating videos with per-frame safe/unsafe similarities and margin overlays without re-running full RL training.

### 30.2 Current Capabilities
* Loads video frames, batches them through CLIP (cpu or gpu).
* Computes and stores safe & unsafe centroid embeddings, cosine similarity distributions, centroid cosine / angle / euclidean separation metrics.
* Produces annotated output video (frame text overlay: safe_sim, unsafe_sim, margin).
* Exports `centroids_and_stats.json` with discriminability metrics (intra vs cross similarities, centroid margin).

### 30.3 Simplification Iterations
1. Initial: CLI args, CSV exports, normalization toggles, potential shaping preview.
2. Simplified: Kept only annotated video + centroid stats JSON (removed CSV & YAML loading path for faster iteration).
3. Added centroid discriminability statistics (pairwise intra-class, inter-class similarity, angle, euclidean distance).
4. Prompt expansion (broad domain descriptors) → observed reduced separation.
5. Prompt pruning using stronger visual anchor (“red warning circle”) producing improved qualitative discriminability.

### 30.4 Planned Enhancements
* Temporal smoothing (EMA or sliding window mean embedding) to reduce per-frame jitter.
* Per-prompt scoring: contribution to centroid margin & variance; automated pruning.
* Batch evaluation harness for multiple prompt lists (grid search) with summarized discriminability score.

### 30.5 Rationale for External Probe
Keeps RL training loop lean; enables rapid semantic hypothesis testing (prompt wording, visual anchor selection) offline, de-risking integration changes.

## 31. Roadmap Status Update (Post v1.8)

| Roadmap Item | Previous Status | Current Status | Notes |
|--------------|-----------------|----------------|-------|
| Reverse discounted risk target | Planned | DONE | Backward rollout implemented (v1.8) |
| Potential-based shaping | DONE | DONE | Stable; neutrality metrics in place |
| Spatial batching | DONE | DONE | Baseline path retained for ablation |
| Lambda modulation | Pending | Pending | Await robust risk correlation + quantile design |
| Temporal smoothing (probe) | Not started | Pending | To implement in offline script first |
| Prompt scoring & pruning | Not started | Pending | Will integrate into probe before training auto-adapt |
| Unit tests (semantic) | Partial | Partial | Still need margin sign swap, beta schedule, risk correlation synthetic |
| Config snapshot export | Pending | Pending | Include transformers version & prompt hashes |
| Async embedding | Deferred | Deferred | Only if profiling identifies bottleneck post batching |

## 32. Updated Immediate Action Recommendations
1. Add episode-aware masking to risk target builder (avoid cross-episode accumulation in long buffers).
2. Implement temporal smoothing in offline probe; reassess centroid cosine and per-frame variance.
3. Introduce per-prompt scalar influence score (delta to centroid separation when removed) and prune low-impact prompts.
4. Prepare unit tests for semantic margin normalization and risk target correctness.
5. Prototype λ modulation using empirical risk prediction quantiles (simulate scaling factor offline with logged trajectories before live integration).

---

## 28. v1.6 Addendum: Telemetry & Configuration Consolidation

### 28.1 Overview
This addendum documents the consolidation steps taken after spatial batching: richer semantic diagnostics, pruning of obsolete status logs, and introduction of scaling & normalization toggles to better understand—and control—the influence of semantic shaping on learning dynamics. It also records the analytic tooling added to interpret logged CSV metrics.

### 28.2 New / Revised Telemetry (Supersedes Older Glossaries)
| Key | Purpose | Design Philosophy | Value Provided |
|-----|---------|------------------|----------------|
| `Semantics/RawMargin` | Unnormalized (scaled) safe–unsafe cosine gap | Expose raw signal before statistical transforms | Diagnose prompt separability & need for scaling |
| `Semantics/NormMargin` | Z-score of margin (rolling window) | Stabilize schedule across prompt sets; reversible via toggle | Comparable shaping scale across experiments |
| `Semantics/Beta` | Instant shaping coefficient (cosine decay) | Make curriculum explicit & auditable | Correlate learning acceleration with schedule phase |
| `Semantics/ClampFrac` | Cumulative fraction of clipped normalized margins | Detect over-aggressive clipping bounds | Decide whether to widen clamp or adjust scaling |
| `Semantics/CaptureCount` | Total embedding capture events | Coverage accounting (esp. with batching) | Normalizes downstream ratios (e.g., success rate) |
| `Semantics/CaptureIntervalEffective` | Steps since last capture | Reveal skipped / delayed captures (e.g., exceptions) | Ensures schedule consistency & identifies dead zones |

Removed (noise reduction): `Semantics/Debug/ClipReady`, `Semantics/Debug/ClipStatus` — once CLIP load stabilized and status strings ceased to vary meaningfully across runs they were pruned to reclaim log bandwidth and focus attention on actionable metrics.

### 28.3 Config Additions & Rationale
| Config Field | Default | Rationale | Failure Mode Mitigated | Value |
|--------------|---------|----------|------------------------|-------|
| `risk_lr` | 1e-3 | Decouple auxiliary head optimization speed from hard-coded constant | Over/under-fitting of risk predictor when changing horizon/discount | Tunable convergence of risk head without code edits |
| `margin_norm_enable` | True | Allow ablation of normalization; inspect raw scale | Over-normalization masking prompt improvements | Clear attribution: is normalization helping? |
| `margin_scale` | 1.0 | Controlled amplification of raw margin when separation weak | Very low signal-to-noise shaping < reward noise | Rapid what-if scaling without prompt churn |

### 28.4 Design Philosophy Recap
1. **Early Bias, Long-term Neutrality**: Shaping coefficient anneals to zero so asymptotic optimal policy set is preserved (semi-potential approach). Telemetry (`Beta`, `ClampFrac`) confirms actual decay shape & clipping health.
2. **Observability First**: Introduce diagnostics (margins, clamp fraction, capture interval) before advanced control (e.g., modulation) to avoid blind tuning.
3. **Separation of Concerns**: `SemanticManager` owns semantic state (prompts, embeddings, schedule); adapters stay thin; PPO core unmodified—facilitating safe iterative augmentation.
4. **Fail-Soft Architecture**: On any embedding failure path we emit zeros while still advancing `global_step` ensuring schedule continuity and preventing skewed normalization windows.
5. **Incremental Instrumentation Removal**: Once readiness signals ceased to provide new information they were removed to reduce log entropy—favoring high information density metrics.

### 28.5 Value of Each Key Addition (Narrative)
- **RawMargin / NormMargin**: Unlock root-cause analysis (`low raw margin variance` vs `normalization suppressing extremes`). Previously only aggregate shaping magnitude hid whether issue was schedule or semantic separability.
- **Beta**: Makes curriculum debuggable; without logging, any plateau could be misattributed to algorithm instability instead of near-zero shaping weight.
- **ClampFrac**: Guards against silent saturation; high clamp rate indicates either prompts too extreme or scaling too high—actionable levers.
- **CaptureIntervalEffective**: Detects latent performance regressions (e.g., occasional render failures) that would otherwise silently reduce semantic coverage.
- **margin_scale & norm toggle**: Provide orthogonal levers (scale amplitude vs normalize distribution) enabling controlled studies without editing code.
- **risk_lr**: Unlocks independent tuning; risk head and policy may demand different effective learning rates depending on horizon.

### 28.6 CSV Analysis Script (`examples/semantic_analysis.py`)
Added a reusable analysis utility that:
- Locates latest `progress.csv` per run label.
- Produces smoothed return & cost curves, cost-return frontier scatter.
- Plots Beta schedule, RawMargin trajectories, ClampFrac trend, shaping magnitude.
- Computes summary statistics (final/avg return & cost, shaping ratio, margin stats, margin→future-return correlations) and emits `summary.json`.
Purpose: Provide fast iteration loop for semantic feature justification (quantify benefit or diagnose neutrality before deeper engineering effort).

### 28.7 Alignment with Thesis Theme
The instrumentation & config refinements shift the project from *“we added a VLM-derived scalar”* to *“we systematically evaluate when and why foundational multimodal features produce safe RL benefits”*. Every new metric reduces epistemic uncertainty about whether underperformance arises from signal scarcity, mis-scaling, curriculum timing, or fundamental semantic irrelevance—central to credible claims about leveraging foundational models for safety.

### 28.8 Next High-Impact Steps (Post v1.6)
1. Wire `risk_lr` into optimizer creation (currently hard-coded usage pending) + log actual lr.
2. Introduce potential-based shaping variant (beta * (γ φ(s') - φ(s))) to remove residual bias while preserving early guidance—compare cost & return neutrality.
3. Spatial batching statistical validation: log per-env shaping std and mean to confirm variance reduction claim vs pre-batching scalar broadcast.
4. Margin→cost predictive power: log rolling Pearson corr(margin_t, future_cost_{t:t+H}) to decide if further prompt engineering warranted.
5. Automated prompt set evaluation harness (select highest AUC separating high vs low cost frames).

### 28.9 Risk of Over-Instrumentation & Mitigation
Risk: Excess metrics increase cognitive load. Mitigation: Grouped semantic metrics share prefix; obsolete readiness metrics culled; future additions must demonstrate unique decision leverage (documented in value table on introduction).

### 28.10 Version Integrity
This addendum supersedes earlier sections describing single-frame broadcast behavior as the “current” implementation; refer to v1.5/v1.6 for spatial batching truth. Legacy description retained historically but not authoritative for present code path.

---

