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

#### 4.0.3 Semantic Reward Shaping
Let a frozen vision-language encoder map an observation (frame) $x_t$ to an embedding $e_t \in \mathbb{R}^d$. Safe and unsafe prompt sets $\mathcal{P}_{safe}, \mathcal{P}_{unsafe}$ are encoded; their (optionally normalized) centroids are $\mu_{safe}, \mu_{unsafe}$. Define the semantic margin
$$
m_t = \cos(e_t, \mu_{safe}) - \cos(e_t, \mu_{unsafe}).
$$
Maintain running mean/variance $\mu_m, \sigma_m^2$ to obtain a normalized margin $\tilde m_t = (m_t - \mu_m)/(\sigma_m + \varepsilon)$. Apply clipping $\bar m_t = \operatorname{clip}(\tilde m_t, -M, M)$.

Annealed shaping coefficient (cosine schedule up to an $\alpha$-fraction of training steps $T_{anneal}=\alpha T$):
$$
\beta_t = \beta_0 \cdot \tfrac{1}{2} \Big(1 + \cos\big(\pi \cdot \min(1, t / T_{anneal})\big) \Big).
$$
Shaped reward:
$$
r'_t = r_t + \beta_t \bar m_t.
$$
Because $\beta_t \to 0$ as $t \to T_{anneal}$, asymptotic optimality under the original MDP is preserved (potential-based shaping like guarantees are approximated via annealing to null influence).

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

#### 4.0.5 Planned Lagrange Modulation (Scaffold)
Let empirical distribution of recent predicted truncated costs be $Q$. For chosen upper quantile $q_{\alpha}$, define a modulation scale (example logistic transform):
$$
\eta_t = \sigma\left( \frac{ q_{\alpha}(Q) - q_{med}(Q)}{\tau} \right), \quad \sigma(z)=\frac{1}{1+e^{-z}}.
$$
Future work: adjust effective learning rate of $\lambda$ update: $ \lambda \leftarrow \lambda + \eta_t \cdot \Delta \lambda_{base} $. This adaptively accelerates constraint enforcement when tail risk rises.

#### 4.0.6 Convergence Considerations
Because shaping vanishes ($\beta_t \to 0$) and auxiliary loss does not alter the final reward signal, fixed points coincide with those of PPO-Lagrange (assuming perfect critics). Auxiliary risk training influences representation quality and early policy gradients but not terminal optimality.

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
| Lagrange Update | Standard | (Currently same; modulation scaffold only) |
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
- Risk target is a sliding approximation (not exact bootstrapped cost-to-go).
- No Lagrange modulation yet (feature scaffold only).
- Potential domain gap: CLIP general visual pretraining may not perfectly encode environment-specific safety cues (may require prompt tuning or adapter fine-tuning).
- Additional GPU memory & initial download time for CLIP (≈600MB).

## 8. Roadmap
| Phase | Goal | Actions | Success Criteria |
|-------|------|---------|------------------|
| P0 Parity | Ensure no regression | Run PPOLag vs PPOLagSem (disabled semantics) | Matching curves (± small stochastic noise) |
| P1 Activation | Validate shaping & risk | Enable shaping & risk separately | No crashes; shaping decays; risk loss declines |
| P2 Metrics | Diagnostic depth | Log raw vs shaped reward, risk correlation | Risk pred corr > 0.3 early |
| P3 Modulation | Adaptive constraint tuning | Implement LR scaling of Lagrange via risk quantiles | Reduced overshoot/variance of cost |
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
|18 | Updated thesis with formal math & deep dive | `thesis.md` | Documentation completeness | Current version bump |

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
| `Semantics/Debug/ClipReady` | Binary | Manager flag | 1 if model loaded and centroids computed |
| `Semantics/Debug/ClipStatus` | Categorical string | Manager | Encodes dtype success or error class |
| `Risk/Loss` | Scalar | Risk head train loop | Smooth L1 loss |
| `Risk/PredMean` | Scalar | Risk head output | Mean predicted discounted cost |
| `Risk/TargetMean` | Scalar | Computed targets | Mean target truncated discounted cost |
| `Risk/Corr` | Scalar (optional) | Correlation computation | Pearson correlation prediction vs target |

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
*Document Version: v1.4 (adds executive technical summary, baseline vs semantic pseudocode, explicit parity guarantee, clarified risk head update timing).* 

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
- Single global frame per capture (no per-env spatial batching).
- Subsampled risk buffer (one embedding per capture interval regardless of env count).
- No λ modulation, no late fusion of embeddings into policy network.
- Risk target = truncated forward sum only (no reverse cumulative or bootstrap).
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

