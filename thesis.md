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
| Logging | Shaping magnitude, latency, risk loss | Diagnose overhead & learning effect | Transparent evaluation |

## 4. Detailed Components
### 4.0 Mathematical Formulation (Addendum)
This section formalizes the modifications introduced by semantic guidance. Symbols:
\(s_t\): state, \(a_t\): action, \(r_t\): environment reward, \(c_t\): instantaneous cost, \(\gamma\): reward discount, \(\gamma_c\): cost discount (often =\nobreakspace\(\gamma\)), \(\theta\): policy parameters, \(\lambda\): Lagrange multiplier, \(d\): cost limit, \(T\): total timesteps, \(H\): risk (truncation) horizon.

#### 4.0.1 Constrained Objective
\[
\max_{\theta} \; J(\theta) = \mathbb{E}_\pi \Big[ \sum_{t=0}^{\infty} \gamma^{t} r_t \Big] \quad \text{s.t.} \quad J_C(\theta)= \mathbb{E}_\pi \Big[ \sum_{t=0}^{\infty} \gamma_c^{t} c_t \Big] \le d.
\]

#### 4.0.2 Lagrangian Relaxation (PPO-Lagrange)
Ignoring the constant \(\lambda d\) during gradient steps, the per-timestep shaped advantage surrogate becomes
\[
\mathcal{L}_{\text{PPO-Lag}}(\theta) = \mathbb{E}_t \Big[ \min( r_t(\theta) \hat A_t^{R,\lambda}, \; \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon) \hat A_t^{R,\lambda}) \Big],
\]
with importance ratio \(r_t(\theta)= \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\) and combined reward-cost advantage
\[
\hat A_t^{R,\lambda} = \hat A_t^{R} - \lambda \hat A_t^{C}.
\]
Standard value losses (reward and cost critics) and entropy regularization are added as in PPO.

#### 4.0.3 Semantic Reward Shaping
Let a frozen vision-language encoder map an observation (frame) \(x_t\) to an embedding \(e_t \in \mathbb{R}^d\). Safe and unsafe prompt sets \(\mathcal{P}_{safe}, \mathcal{P}_{unsafe}\) are encoded; their (optionally normalized) centroids are \(\mu_{safe}, \mu_{unsafe}\). Define the semantic margin
\[
m_t = \cos(e_t, \mu_{safe}) - \cos(e_t, \mu_{unsafe}).
\]
Maintain running mean/variance \(\mu_m, \sigma_m^2\) to obtain a normalized margin \(\tilde m_t = (m_t - \mu_m)/(\sigma_m + \varepsilon)\). Apply clipping \(\bar m_t = \operatorname{clip}(\tilde m_t, -M, M)\).

Annealed shaping coefficient (cosine schedule up to an \(\alpha\)-fraction of training steps \(T_{anneal}=\alpha T\)):
\[
\beta_t = \beta_0 \cdot \tfrac{1}{2} \Big(1 + \cos\big(\pi \cdot \min(1, t / T_{anneal})\big) \Big).
\]
Shaped reward:
\[
r'_t = r_t + \beta_t \bar m_t.
\]
Because \(\beta_t \to 0\) as \(t \to T_{anneal}\), asymptotic optimality under the original MDP is preserved (potential-based shaping like guarantees are approximated via annealing to null influence).

#### 4.0.4 Auxiliary Risk Prediction
Define truncated discounted cost target (sliding horizon):
\[
g_t = \sum_{k=0}^{H-1} (\gamma_c)^k c_{t+k}.
\]
Risk head \(q_\phi(e_t) \approx g_t\) (parameters \(\phi\), embedding detached from encoder). Loss:
\[
\mathcal{L}_{risk}(\phi) = \operatorname{SmoothL1}\big(q_\phi(e_t), g_t\big).
\]
Total optimized objective (conceptual):
\[
\mathcal{J}(\theta,\phi,\lambda)= -\mathcal{L}_{\text{PPO-Lag}}(\theta,\lambda) + w_{risk} \big(-\mathcal{L}_{risk}(\phi)\big) + w_{val} \mathcal{L}_{val} - w_{ent} \mathcal{L}_{ent}.
\]
Only \(\theta\) receives gradients through policy surrogate; \(\phi\) through risk head; encoder frozen.

#### 4.0.5 Planned Lagrange Modulation (Scaffold)
Let empirical distribution of recent predicted truncated costs be \(Q\). For chosen upper quantile \(q_{\alpha}\), define a modulation scale (example logistic transform):
\[
\eta_t = \sigma\left( \frac{ q_{\alpha}(Q) - q_{med}(Q)}{\tau} \right), \quad \sigma(z)=\frac{1}{1+e^{-z}}.
\]
Future work: adjust effective learning rate of \(\lambda\) update: \( \lambda \leftarrow \lambda + \eta_t \cdot \Delta \lambda_{base} \). This adaptively accelerates constraint enforcement when tail risk rises.

#### 4.0.6 Convergence Considerations
Because shaping vanishes (\(\beta_t \to 0\)) and auxiliary loss does not alter the final reward signal, fixed points coincide with those of PPO-Lagrange (assuming perfect critics). Auxiliary risk training influences representation quality and early policy gradients but not terminal optimality.

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
- [ ] Parity test (PPOLag vs PPOLagSem disabled)
- [ ] Unit tests (semantic manager, risk head, registration)

### Diagnostics & Logging
- [ ] Log raw (unshaped) reward alongside shaped
- [ ] Log shaping term avg/std per epoch
- [ ] Log risk prediction vs empirical discounted cost correlation
- [ ] Log embedding FPS overhead & latency distribution (min/mean/p95)

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
*Document Version: v1.0 (initial integration summary). Update this file as tasks are completed.*
