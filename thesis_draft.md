# Thesis Draft Outline: Semantic & Risk-Augmented Safe Reinforcement Learning (PPOLagSem)

## 0. Metadata
- Working Title: Accelerating Safe Reinforcement Learning with Frozen Vision–Language Semantics and Auxiliary Risk Prediction
- Author: <Your Name>
- Date: <To fill>
- Repository Commit Hash: <To fill at submission>

## 1. Abstract (1 paragraph)
This thesis investigates how to train reinforcement learning agents that learn efficiently while satisfying explicit safety constraints. In many domains, safety feedback is sparse and delayed, which slows learning and can lead to unstable behavior that alternates between over‑caution and excessive risk. The work proposes PPOLagSem, a practical extension to PPO‑Lagrangian that leverages off‑the‑shelf vision–language models to supply high‑level cues about whether an observation appears safe or unsafe. These cues provide a gentle guidance signal that aids early exploration and then decays over time, and they also support a lightweight auxiliary predictor of near‑term safety risk. An adaptive mechanism uses the predicted risk to adjust how strongly the algorithm enforces the constraint, promoting steadier progress. The design requires no finetuning of the vision–language model, integrates without modifying the core algorithm, and keeps overhead modest by sampling observations at intervals and batching work when helpful. On car‑navigation safety tasks, PPOLagSem matches the baseline when semantic guidance is disabled and, when enabled, learns faster and violates constraints less often while maintaining low runtime overhead. The contributions are a simple, model‑agnostic semantic guidance mechanism compatible with safe RL; an auxiliary, episode‑aware risk estimation head; a risk‑quantile‑based modulation of constraint updates; and a parity‑preserving implementation with diagnostics and reproducibility in mind. The thesis concludes with limitations and directions for uncertainty‑aware risk modeling, adaptive schedules, and automatic prompt selection to broaden generalization and strengthen safety guarantees.
- Motivation: Slow safe RL convergence; underutilized pretrained semantics.
- Approach: Integrate frozen vision–language encoder + semantic reward shaping + auxiliary risk predictor + adaptive constraint modulation.
- Key Results: Faster early learning, stable constraint adherence, low overhead, model-agnostic (CLIP/SigLIP).
- Contributions: (i) unified semantic shaping & risk framework, (ii) episode-aware discounted risk formulation, (iii) modulation via risk quantile gap, (iv) robust NaN-safe training pipeline.

**Key Points:**
- Condense to ~180–220 words highlighting problem, method, results, contributions.
- Emphasize neutrality (no degradation when disabled) + low overhead.
- Quantify gains (e.g., X% AUC improvement, Y% violation reduction) once results finalized.
- Stress model-agnostic usage of frozen VLM encoder.
- Mention adapter-based integration that requires no PPO core changes and preserves existing training APIs.
- Call out parity check: PPOLag vs PPOLagSem(enable=False) learning curves overlap within CI.
- Include overhead figure: encoder amortized by capture_interval and optional cross-env batching.

**Notes:**
- Add concrete metrics after experiments finalize.
- Ensure contribution phrasing matches numbered list in Introduction 2.5.
- Define any specialized terms if used (semantic margin, risk modulation).

## Chapter 1: Introduction
Chapter overview: Background and motivation, objectives and research questions, contributions and achievements, and an overview of the thesis structure.
1.1 Problem Setting & Challenges in Safe RL
**Key Points:** Safe RL is sample-inefficient; constraint cost signals sparse/delayed; balancing exploration vs safety leads to slow convergence.
**Notes:** Cite canonical benchmarks; clarify distinction cost variance vs mean.
**Suggested Figures & Tables:**
- Fig. 1: Problem schematic showing exploration vs safety trade-off (timeline with sparse cost events).
- Table 1: Summary of challenges in safe RL and their practical implications.
1.2 Opportunity: Pretrained Vision–Language Priors
**Key Points:** Frozen VLMs encode object & relational semantics zero-shot; can yield dense safety context. No finetuning needed; prompts define task-relevant safe/unsafe concepts. Semantics can provide dense hints early when extrinsic rewards/costs are sparse.
**Notes:** Briefly contrast CLIP vs SigLIP alignment.
**Suggested Figures & Tables:**
- Fig. 2: Illustration of VLM zero-shot similarity (safe vs unsafe prompt examples over sample frames).
- Table 2: CLIP vs SigLIP characteristics relevant to this thesis (alignment, resolution, latency).
1.3 Limitations of Existing Shaping & Risk Estimation Approaches
**Key Points:** Manual shaping brittle; prior semantic shaping not safety-targeted; risk estimators cold-start unstable.
**Notes:** Include 2–3 representative citations.
1.4 High-Level Idea & Intuition
**Key Points:** Use semantic margin as curriculum shaped by annealed β; risk head predicts future cost; modulation adapts constraint pressure. Running example: car agent gets positive shaping when margin indicates “on-road, far from pedestrian” and negative when “off-road, near obstacle”; β anneals to zero to avoid biasing final policy.
**Notes:** Add running car hazard illustrative example.
**Suggested Figures & Tables:**
- Fig. 3: Running example storyboard (3–4 frames with annotations of margin sign and magnitude).
- Fig. 4: Conceptual flowchart of PPOLagSem (observe → embed → margin → shaping/risk → modulation → PPO update).
1.5 Contributions (numbered) – mapping to later chapters
1) Semantic shaping with frozen VLMs: safe/unsafe prompt centroids and a cosine margin with running z-normalization and clamp to [-2, 2]; cosine-annealed shaping coefficient β ensures early guidance with eventual neutrality. (Sec. 5.2–5.4, 6.1–6.2)
2) Auxiliary risk prediction: a lightweight risk head trained with SmoothL1 on truncated discounted episodic cost targets; episode-aware masking at terminals and optional mini-batching/iterations for stability. (Sec. 5.5, 7.4)
3) Risk-aware modulation: scale Lagrange multiplier updates via a quantile-based logistic gate computed from predicted risk distribution, reducing oscillations and constraint violations. (Sec. 5.6, 6.4)
4) Minimal-intrusion architecture: adapter + manager injection into PPO-Lag without touching core PPO update; strict parity when disabled; robust dtype/device fallbacks and NaN/Inf sanitization. (Sec. 5.7, 7.1)
5) Telemetry for fast diagnosis: comprehensive Semantics/* and Risk/* metrics (latency, clamp frac, shaping ratio, risk corr) enabling proactive fixes and compute control. (Sec. 5.8, 9.6)
**Notes:** Numbering must match Abstract & Conclusion.
**Suggested Figures & Tables:**
- Table 3: Contributions and the sections/figures where each is validated (traceability matrix).
1.6 Thesis Structure / Chapter Roadmap
**Key Points:** One sentence per major section summarizing role.
**Notes:** Keep ≤6 lines.
**Suggested Figures & Tables:**
- Fig. 5: Chapter roadmap diagram (swimlane style linking methods→experiments→results).

## Chapter 2: Background Theory and Literature Review
2.1 Safe Reinforcement Learning & Constrained MDPs
**Key Points:** Quick RL/MDP recap (MDP tuple ⟨S,A,P,r,γ⟩; return J_r; value/advantage); formal CMDP with cost signal c_t and constraint E[J_c] ≤ d; enforce via Lagrangian relaxation and dual ascent on λ.
**Notes:** Introduce symbols reused later (r_t, c_t, J_r, J_c, V, A, λ); clarify that safe RL optimizes reward subject to a cost budget.
**Suggested Figures & Tables:**
- Fig. 6: CMDP diagram (states/actions/rewards/costs; constraint on expected cumulative cost).
2.2 PPO-Lagrangian Fundamentals
**Key Points:** PPO basics (clipped surrogate with advantage Â, value baseline, entropy bonus) and PPOLag extension (replace Â with reward advantage in objective and add λ·cost term with dual ascent on λ); entropy helps stabilize updates; cost advantage can be estimated with a separate cost critic.
Formulas (concise):
- Reward objective: maximize L_PPO(θ) = E[min(r_t(θ)·Â_r,t, clip(r_t(θ), 1−ε, 1+ε)·Â_r,t)] + α·E[H(π_θ(·|s_t))].
- Cost objective: J_c(θ) = E[∑ γ_c^t c_t], with cost advantage Â_c,t from a cost critic V_c.
- Lagrangian penalized objective: maximize L_total(θ,λ) = L_PPO(θ) − λ·E[Â_c,t] (or penalty on empirical cost), subject to λ ≥ 0.
- Dual update: λ ← [λ + η_λ·(E[c_t] − d)]_+, optionally using cost advantage or episodic costs.
Algorithm sketch:
1) Collect rollouts; compute reward and cost advantages (GAE for both); 2) Optimize θ with PPO on L_total for K epochs; 3) Update λ via projected gradient/ascent toward meeting the cost budget d; 4) Repeat.
**Notes:** Mention common heuristics: normalize advantages, clip value loss, use separate γ and γ_c, and optionally damp λ updates to reduce oscillations.
**Suggested Figures & Tables:**
- Fig. 7: PPO-Lagrangian update loop (actor/critic, λ ascent) as a block diagram.
- Alg. 1: PPOLag training loop (pseudo-code) with reward/cost critics, advantage estimates, and λ update.

Algorithm 1: PPO‑Lagrangian (PPOLag) — pseudo‑code

```
Inputs:
	Policy π_θ, reward value critic V_ψ, cost value critic V^c_ξ (optional),
	cost limit d ≥ 0, entropy coef α, clip ε, RL discounts γ, γ_c,
	GAE λ_gae, λ_gae_c, PPO epochs K, minibatch size B, λ step size η_λ.
	Initialize λ ← λ_0 ≥ 0.

repeat for iterations k = 1,2,... do
	# 1) Collect rollouts
	D ← { (s_t, a_t, r_t, c_t, logp_old_t, s_{t+1}, done_t) } using π_θ.

	# 2) Compute returns and advantages (reward & cost)
	Ĝ_r,t, Â_r,t ← GAE(r_t, V_ψ, γ, λ_gae) over each trajectory in D.
	Ĝ_c,t, Â_c,t ← GAE(c_t, V^c_ξ, γ_c, λ_gae_c) (or Monte Carlo if no cost critic).
	Normalize Â_r,t and (optionally) Â_c,t per batch.

	# 3) PPO optimization (K epochs)
	for epoch = 1..K do
		for minibatch M ⊂ D of size B do
			r_t(θ) ← exp(log π_θ(a_t|s_t) − logp_old_t).
			L_clip_r ← E_M [ min(r_t · Â_r,t, clip(r_t,1−ε,1+ε) · Â_r,t) ].

			# Penalize predicted constraint violation via λ
			L_cost_adv ← E_M [ r_t · Â_c,t ]   # or use mean cost as surrogate

			# Value losses and entropy bonus
			L_v ← E_M [ (V_ψ(s_t) − Ĝ_r,t)^2 ]
			L_vc ← E_M [ (V^c_ξ(s_t) − Ĝ_c,t)^2 ]  # if using cost critic
			H ← E_M [ entropy(π_θ(·|s_t)) ]

			# Maximize total objective (implemented as minimizing negative)
			L_total ← −L_clip_r + λ · L_cost_adv + c1 · L_v + c2 · L_vc − α · H
			Update θ, ψ, (and ξ) by taking one optimizer step on L_total.
		end for
	end for

	# 4) Dual variable (λ) update toward meeting cost budget d
	J_c_emp ← mean episodic cost over D (or moving average/critic estimate)
	λ ← max(0, λ + η_λ · (J_c_emp − d))   # project to λ ≥ 0
	(Optional) λ ← clip(λ, 0, λ_max) and/or apply momentum/damping.

	Log metrics: returns, costs, λ, losses, KL, clip frac; check constraint satisfaction.
until convergence
```
2.3 Reward Shaping & Potential-Based Shaping Theory
**Key Points:** Potential shaping preserves optimal policy; additive shaping can bias but speeds exploration.
**Notes:** Provide Ng et al. lemma reference.
**Suggested Figures & Tables:**
- Fig. 8: Telescoping illustration of potential-based shaping across a short trajectory.
2.4 Auxiliary Prediction Tasks in RL (Value/Risk/Critic augmentation)
**Key Points:** Auxiliary heads improve representation stability; risk prediction distinct from value baseline.
**Notes:** Clarify difference between cost value vs truncated horizon risk.
**Suggested Figures & Tables:**
- Table 4: Comparison of auxiliary tasks (value vs cost vs risk horizon) and training targets.
2.5 Vision–Language Models (CLIP, SigLIP) & Embedding Properties
**Key Points:** Contrastive embeddings enable zero-shot similarity; pros (transfer) vs cons (bias, resolution limits).
**Notes:** Cite zero-shot accuracy benchmarks.
**Suggested Figures & Tables:**
- Fig. 9: Embedding space sketch and cosine similarity to prompt centroids.
2.6 Notation Summary Table
**Key Points:** Consolidate symbols: s,a,r,c, m (margin), φ potential, β schedule, λ, H_risk.
- s,a: state and action; r: environment reward; c: environment cost; γ, γ_c: reward and cost discounts.
- e: semantic embedding; μ_safe/μ_unsafe: prompt centroids; m: raw margin; m̂: normalized/clamped margin.
- β_t: shaping coefficient over time; φ: potential function; λ: Lagrange multiplier; H_risk: risk horizon.
- q_τ: risk τ-quantile; g: logistic gate for λ update; σ_m: running std of margins; ϵ: small constant for stability.
**Notes:** Table before heavy equations.
**Suggested Figures & Tables:**
- Table 5: Notation summary (symbols, definitions, default values if any).

### Literature Review (continued)
2.7 Safe RL with Auxiliary Signals
**Key Points:** Prior methods add critics/shields; ours integrates semantic shaping + risk modulation.
**Notes:** Stress novelty of VLM semantics in safety context.
**Suggested Figures & Tables:**
- Table 6: Related safe RL methods with auxiliary signals (inputs, objectives, safety mechanism).
2.8 Semantic or Visual Priors in RL
**Key Points:** Use of pretrained backbones/prompts; gap: lack of safety-focused integration.
**Notes:** Optionally taxonomy figure.
**Suggested Figures & Tables:**
- Fig. 10: Taxonomy of semantic prior integration (frozen guidance vs finetuned features vs fused inputs).
2.9 Risk Prediction & Uncertainty Estimation
**Key Points:** Distributional RL & ensembles; our simpler truncated cost predictor offers low overhead.
**Notes:** Position as pragmatic baseline.
**Suggested Figures & Tables:**
- Table 7: Comparison of risk prediction approaches (distributional vs point prediction; compute cost).
2.10 Reward Shaping Methodologies (Potential vs Additive vs Curriculum)
**Key Points:** Neutrality vs speed trade-off; annealing bridges; curriculum parallels margin schedule.
**Notes:** Cross-reference Section 5.4.
**Suggested Figures & Tables:**
- Fig. 11: Schedules for β (constant vs cosine anneal vs piecewise), with area under curve.
2.11 Constraint Adaptation & Lagrange Multiplier Tuning
**Key Points:** Standard gradient ascent vs heuristic adaptors; introduce quantile logistic gating.
**Notes:** Mention bounded derivative advantage.
**Suggested Figures & Tables:**
- Fig. 12: λ adaptation strategies (baseline vs quantile-gated) schematic.
2.12 Positioning: How PPOLagSem Differs
**Key Points:** Unified minimal-intrusion pipeline, neutrality when disabled, modular flags.
**Notes:** Prepare comparison table (maybe Appendix).
**Suggested Figures & Tables:**
- Table 8: Positioning table comparing PPOLagSem against closest baselines (inputs, compute, neutrality, results).

2.13 Baseline RL Algorithms Under Test (overview)
**Key Points:** Summarize all benchmarked algorithms and their safety mechanisms to contextualize PPOLagSem among standard approaches.
**Notes:** Keep concise; detailed evaluation appears in Chapter 4.

Algorithms:
- PPO (unconstrained): Clipped surrogate with entropy; no safety constraint.
- TRPO (unconstrained trust region): Maximizes reward surrogate under KL trust region (E[KL] ≤ δ).
- PPOLag (PPO‑Lagrangian): Lagrangian penalty with dual ascent λ ← [λ + η_λ (J_c − d)]_+; uses cost advantage Â_c.
- TRPOLag: TRPO step with Lagrangian penalty while enforcing KL ≤ δ.
- CPO: Constrained Policy Optimization; solves a local constrained problem each update (reward surrogate max s.t. cost surrogate ≤ d and KL ≤ δ), typically via QP.
- P3O: Primal–dual PPO variant coupling policy update with dual variable dynamics (close to PPOLag, different update details/penalties).
- CPPOPID: PID‑controlled Lagrange multiplier: λ_{t+1} = [λ_t + K_p e_t + K_i ∑ e + K_d Δe]_+, e_t = J_c − d.
- PPOSaute: Budget‑augmented safe RL; augments state with remaining budget b, with dynamics b_{t+1} = b_t − c_t; policy conditions on (s,b).
- PPOSimmerPID: Simmer‑style PID safety controller that gradually adjusts penalty/budget for smooth constraint tracking.

**Suggested Figures & Tables:**
- Fig. 12b: Constraint handling families schematic (Lagrangian, trust‑region, constrained line‑search, PID, budget augmentation) with representative methods.
- Table 8b: Comparison table (algorithm, safety mechanism, constraint handling, key hyperparameters).

## Chapter 3: Methodology
3.1 Problem Formulation and Design Principles
**Key Points:** CMDP framing; goals: early guidance, neutrality, low overhead; modular, adapter-based integration; minimal intrusion to PPO core.
**Notes:** Connect objectives to design constraints and parity guarantees.
**Suggested Figures & Tables:**
- Table 9: Design principles and how they map to module choices.

3.1.1 Research Questions
**Key Points:** When do semantic cues improve early learning? How does risk modulation affect violations and λ stability? What is the overhead/efficiency trade-off?
**Notes:** Tie to evaluation metrics in Chapter 4.
3.1.2 Research Hypothesis
**Statement:** Foundational VLMs encode broad world knowledge and safety‑relevant cues; distilling this prior into RL via annealed semantic shaping and a short‑horizon discounted cost predictor enables faster acquisition of safety behavior (higher early sample efficiency, fewer violations) than relying solely on environment feedback. Because VLMs are too costly for high‑rate control, we employ sparse online captures and ultimately distill to lightweight students for real‑time deployment.
3.2 System Overview & Data Flow Diagram
**Key Points:** Pipeline: env → capture (every capture_interval) → embedding (VLM) → margin → shaping/risk/modulation → PPO update; capture_interval reduces cost. Optional cross-env micro-batching amortizes encoder calls.
**Notes:** Insert overview figure.
**Suggested Figures & Tables:**
- Fig. 13: End-to-end system data flow (with capture interval and cross-env batching callouts).
3.3 Semantic Manager
- Prompt design, centroid embedding, normalization strategy
- Temporal pooling & capture interval
**Key Points:** Precompute centroids; running margin stats; temporal pooling smooths noise. Safe/unsafe prompts grouped and embedded once; centroids stored on host device. Use `clip_probe/centroids.json` when available; otherwise compute on first run and cache. Maintain deques of recent embeddings and margins for diagnostics and risk data sufficiency checks.
**Notes:** Document prompt selection heuristics.
**Suggested Figures & Tables:**
- Fig. 14: Prompt→centroid workflow (cache, normalize, host/GPU placement).
- Table 9: Safe/unsafe prompt sets (examples) and centroid stats (norms, counts).
3.4 Margin Computation & Processing
- Raw margin definition m = cos(e, μ_safe) − cos(e, μ_unsafe)
- Running z-normalization & clamping; margin scale parameter
**Key Points:** Scale then optional normalize; clamp [-2,2] to bound shaping; track clamp fraction. Normalization: m̂ = ((m − μ_m) / (σ_m + ϵ)) · scale, then clamp; log `Semantics/RawMargin`, `Semantics/NormMargin`, and `Semantics/ClampFrac`.
**Notes:** Provide histogram later.
**Suggested Figures & Tables:**
- Fig. 15: Histogram of raw vs normalized margins over training windows.
- Fig. 16: Clamp fraction over steps (line plot) and effect on shaping magnitude.
3.5 Reward Shaping Mechanisms
- Additive shaping formula & annealed β schedule
- Potential-based shaping derivation & neutrality proof sketch
- Shaping neutrality monitoring (ratio metrics)
**Key Points:** Cosine/capped schedule drives β→0; potential form ensures invariance; monitor shaping/total reward ratio. Additive shaping: r′_t = r_t + β_t · m̂_t. Cosine anneal: β_t = β_0 · (1 + cos(π · t / T)) / 2 with clamp to [0, β_0]. Potential-based variant: F(s_t,s_{t+1}) = γ·φ(s_{t+1}) − φ(s_t) yields telescoping sum and exact neutrality.
**Notes:** Include explicit equations.
Add: Optional uncertainty gating of shaping using prompt‑ensemble variance or temperature sweep confidence; log gating rate and effectiveness.
**Suggested Figures & Tables:**
- Fig. 17: β schedule over training steps (cosine anneal) and shaping/total reward ratio decay.
- Fig. 18: Potential-based shaping telescoping diagram (neutrality sketch).
3.6 Auxiliary Risk Prediction
- Target: episode-aware truncated discounted cost horizon
- Horizon parameter & masking algorithm pseudocode
- Loss (SmoothL1) & mini-batch multi-iter schedule
- NaN/Inf sanitation & float32 casting rationale
**Key Points:** Truncated discounted cost resets at terminals; SmoothL1 robust to spikes; iterative updates improve fit. Target: R_t^risk = Σ_{k=0}^{H_risk−1} γ_c^k · c_{t+k}, truncated at episode end. Pseudocode: (1) build per-trajectory masks, (2) compute discounted prefix sums up to horizon, (3) cast to float32, (4) SmoothL1 to prediction. Optional mini-batch size and iterations per PPO epoch control compute.
**Notes:** Ablate horizon length & batch size.
Add: Optional calibration layer (Platt or isotonic) trained on a held‑out slice of rollouts; report ECE and reliability curves.
**Suggested Figures & Tables:**
- Fig. 19: Flowchart/pseudocode block for risk target computation with episode masking.
- Fig. 20: Risk loss (SmoothL1) and correlation trajectory over training.
3.7 Modulation of Lagrange Multiplier Updates
- Quantile gap logistic gating derivation
- Stability safeguards (episode min gate, clamping)
**Key Points:** Compute risk percentile threshold; logistic gating scales λ update; safeguards prevent oscillation. Let q_τ be τ-quantile of predicted risk; define gap g_raw = (q_τ − d)/s, where d is cost limit and s a scale; gate g = sigmoid(k · g_raw). Scale λ update Δλ ← g · Δλ_base. Safeguards: require min samples, clamp g ∈ [g_min, g_max], and damp updates near threshold.
**Notes:** Provide pseudocode & complexity.
**Suggested Figures & Tables:**
- Fig. 21: Gate curve g vs quantile gap with different slopes k (bounded derivative highlight).
- Fig. 22: λ update flowchart showing gating and safeguards (min samples, clamping).
3.8 Precision & Device Strategy
- Mixed precision for encoder vs float32 risk head
- Memory & latency trade-offs
**Key Points:** bf16/fp16→fp32 fallback; embeddings moved to CPU; risk head colocated for efficiency. Status codes/logs indicate dtype fallback success; on CPU cast embeddings to fp32; if GPU available, keep encoder on model_device and copy embeddings to host_device for shaping/logging.
**Notes:** Justify no encoder finetuning.
**Suggested Figures & Tables:**
- Table 10: Precision/device configurations and observed effects (latency, stability).
3.9 Logging & Telemetry Design
- Key metrics categories (semantic, shaping, risk, modulation, performance)
**Key Points:** Metrics enable early failure detection (latency, clamp frac, risk corr). Log keys include: Semantics/Shaping, Semantics/RawReward, Semantics/EmbedLatencyMs, Semantics/EmbedSuccessRate, Semantics/Debug/EmbedAttempts, Semantics/Debug/EmbedSuccess, Semantics/RawMargin, Semantics/NormMargin, Semantics/Beta, Semantics/ClampFrac, Semantics/CaptureCount, Semantics/CaptureIntervalEffective, Semantics/ShapingRewardRatio, Semantics/ShapingStd; and Risk/Loss, Risk/PredMean, Risk/TargetMean, Risk/Corr.
**Notes:** Map metrics to debugging actions.
**Suggested Figures & Tables:**
- Fig. 23: Example telemetry dashboard layout (Semantics/*, Risk/*, steps/sec) with callouts.
3.10 Computational Complexity & Overhead Analysis
**Key Points:** Encoder dominates; overhead amortized by interval & batching; target <10% wall time.
**Notes:** Provide measured ms/frame table.
**Suggested Figures & Tables:**
- Table 11: Component-wise latency (ms/frame) and contribution to overhead.
- Fig. 24: Steps/sec vs capture_interval and batch size (lines with error bands).
- Fig. 25: Stacked bar chart of overhead breakdown (encoder, transfers, risk head).
3.11 Failure & Degradation Modes (graceful fallback design)
**Key Points:** Failures trigger automatic disable; NaN detection resets shaping; logging ensures visibility.
**Notes:** Enumerate trigger conditions.
**Suggested Figures & Tables:**
- Fig. 26: State machine for fail-safe toggles and fallback paths.
- Table 12: Trigger conditions and resulting actions (disable shaping, reset counters, warnings).

## 6. Theoretical Considerations
6.1 Potential-Based Neutrality Argument
**Key Points:** Telescoping potential differences cancel in episodic return.
**Notes:** Reference Ng et al. 1999 theorem.
**Suggested Figures & Tables:**
- Fig. 27: Telescoping sum illustration across episode boundaries.
6.2 Bias Analysis of Additive Shaping with Annealing
**Key Points:** Early bias proportional to β; integral of β over time bounds cumulative bias. With cosine anneal, ∫_0^T β(t) dt = β_0·T/2, bounding cumulative shaping influence and motivating large-T schedules for neutrality.
**Notes:** Provide schedule integral bound.
**Suggested Figures & Tables:**
- Fig. 28: Area under β(t) schedule and bound on cumulative bias (shaded integral).
6.3 Variance Reduction via Risk Auxiliary Signal (qualitative)
**Key Points:** Shared representation stabilizes value gradients; risk head encourages disentangled safety features.
**Notes:** Optional empirical variance figure.
6.4 Stability of Quantile-Based Modulation (bounded derivative)
**Key Points:** Logistic gating derivative bounded ⇒ prevents λ explosion; percentile adapts to distribution shifts.
**Notes:** Show derivative max.
**Suggested Figures & Tables:**
- Fig. 29: Derivative of gate vs gap; highlight maximum slope region.
6.5 Discussion of Convergence Guarantees (limitations)
**Key Points:** Non-stationarity (annealing, modulation) complicates formal guarantees; neutrality partially mitigates.
**Notes:** Scope limited to empirical validation.

## 7. Implementation
7.1 Code Architecture Mapping (module diagram)
**Key Points:** Adapter pattern injects semantics without altering PPO core. Key modules: `omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag_semantic.PPOLagSem`, `omnisafe.adapter.semantic_onpolicy_adapter.SemanticOnPolicyAdapter`, `omnisafe.common.semantics.semantic_manager.SemanticManager`, `omnisafe.common.semantics.risk_head.SemanticRiskHead`, config in `omnisafe/configs/on-policy/PPOLagSem.yaml`.
**Notes:** Provide module diagram.
**Suggested Figures & Tables:**
- Fig. 30: Module/adapter diagram showing where semantics connect to PPO pipeline.
7.2 Configuration Surface & Defaults Table
**Key Points:** Flags gate each feature; conservative defaults maintain parity when disabled. Examples: `--semantic-cfgs:enable`, `--semantic-cfgs:shaping_enable`, `--semantic-cfgs:risk_enable`, `--semantic-cfgs:modulation_enable`, `--semantic-cfgs:capture_interval`, `--semantic-cfgs:batch_across_envs`, `--semantic-cfgs:batch_max`, prompt lists, devices.
**Notes:** Cross-check CLI vs YAML overrides.
**Suggested Figures & Tables:**
- Table 13: Configuration flags and defaults (CLI and YAML), with recommended ranges.
7.3 Buffer Structures & Memory Footprint Estimation
**Key Points:** Deques bounded; memory O(window * dim); negligible vs model.
**Notes:** Include concrete example numbers.
**Suggested Figures & Tables:**
- Table 14: Memory footprint examples for typical window and embedding dims.
7.4 Mini-Batching & Iterative Risk Updates
**Key Points:** Multiple iters refine fit; optional mini-batch balances compute. Schedule: per PPO epoch, if samples ≥ min_required, run `risk_update_iters` passes; each pass: sample mini-batch (or full batch) of (embedding, target), SmoothL1 step, log Risk/*.
**Notes:** Pseudocode snippet.
**Suggested Figures & Tables:**
- Fig. 31: Sequence diagram of risk head updates within a PPO epoch.
7.5 Numerical Safeguards (sanitization, event counters)
**Key Points:** Replace NaN/Inf; counters logged; prevents silent corruption.
**Notes:** Table of safeguards.
**Suggested Figures & Tables:**
- Table 15: Numerical safeguards, detection conditions, and mitigations.
7.6 Reproducibility Hooks (seeds, version logging, prompt hashing)
**Key Points:** Prompt hashing ensures experiment provenance. Hash safe/unsafe prompt lists and centroids file; log repo commit, config diff, devices, dtype status; store seeds and environment versions.
**Notes:** Document seed utilities.
**Suggested Figures & Tables:**
- Fig. 32: Reproducibility pipeline (from config snapshot to artifact archiving).

## Chapter 4: Results and Discussion
This chapter presents the experimental setup, results, and interpretive discussion. It consolidates what were previously separate setup, results, and discussion sections.
4.1 Environments & Safety Constraints
**Key Points:** List tasks (e.g., SafetyCarGoal1-v0, SafetyCarGoal2-v0), cost thresholds, horizon specifics; note image observation specs if applicable and frame capture policy.
**Notes:** Provide summary table.
**Suggested Figures & Tables:**
- Table 16: Environment specs (observations, action spaces, cost limits, horizons).
- Fig. 33: Example frames from each environment with safe/unsafe regions highlighted.
4.2 Baselines & Ablations (baseline PPO-Lag, shaping-only, risk-only, full, potential shaping; plus external baselines: PPO, TRPO, TRPOLag, CPO, P3O, CPPOPID, PPOSaute, PPOSimmerPID)
**Key Points:** Isolate effect of each PPOLagSem component and compare against a broad baseline suite trained under identical evaluation protocol.
**Notes:** Pre-register seeds for all methods; ensure identical envs, horizons, and evaluation windows.

Baseline suite (trained):
- PPO — unconstrained PPO (return-only baseline).
- TRPO — unconstrained trust-region policy optimization.
- TRPOLag — TRPO with Lagrangian constraint handling.
- CPO — Constrained Policy Optimization (safety-first line search).
- P3O — primal–dual PPO variant for constrained optimization.
- CPPOPID — PPO with PID Lagrange multiplier controller.
- PPOSaute — SAUTE-style budget-augmented safe RL.
- PPOSimmerPID — Simmer-style PID safety controller.

Safety mechanism summary:
- None: PPO, TRPO.
- Lagrangian (gradient ascent): PPOLag, TRPOLag.
- Constrained line-search: CPO.
- Primal–dual: P3O.
- PID Lagrange: CPPOPID, PPOSimmerPID.
- Budget augmentation: PPOSaute.

**Suggested Figures & Tables:**
- Fig. 35a: Learning curves overlay across baseline suite (return and cost).
- Table 17: Experimental conditions and toggled components per ablation and baseline suite membership (algorithm, safety mechanism, key hyperparameters).
4.3 Hyperparameter Grid Summary
**Key Points:** Focus on β schedule (β_0, T), margin_scale, clamp range, normalization on/off, risk_horizon ∈ {8, 16, 32}, mini-batch size, risk_update_iters, modulation τ ∈ {0.7, 0.8, 0.9}, gate slope k, devices.
**Notes:** Report total runs & compute budget.
**Suggested Figures & Tables:**
- Table 18: Hyperparameter grid (β_0, T, margin_scale, H_risk, τ, k, batch sizes) and budgets.
4.4 Metrics & Evaluation Criteria
- Sample efficiency (AUC early phase)
- Constraint violation rate & cost variance
- Shaping reward ratio decay
- Risk prediction accuracy (Loss, Corr)
- Overhead (steps/sec change)
**Key Points:** Metrics map directly to contributions; AUC captures early efficiency.
**Notes:** Specify statistical tests (bootstrap CI, nonparametric tests).
Add: Teacher validity checks — calibration error (ECE), reliability curves; counterfactual baselines (random prompts, polarity swaps); segmentation/object‑removal ablations to probe causality vs correlation.
**Suggested Figures & Tables:**
- Table 19: Metric definitions, units, and computation windows.
4.5 Hardware & Runtime Environment
**Key Points:** Report GPU/CPU; justify device placement choices.
**Notes:** Provide container spec.
**Suggested Figures & Tables:**
- Table 20: Hardware summary (GPUs, CPUs, RAM) and software stack (containers, drivers).
4.6 Statistical Methodology (seeds, CI computation, tests)
**Key Points:** Seed count justification; CI method (BCa bootstrap?); multiple comparison handling.
**Notes:** Address reproducibility.
**Suggested Figures & Tables:**
- Fig. 34: Analysis workflow (data → bootstrap → significance tests → plots/tables).

4.7 Baseline Parity (disabled semantics) – validation of neutrality path
**Key Points:** Curves overlap with baseline; statistical tests show no significant difference.
**Notes:** Provide p-values / effect sizes.
**Suggested Figures & Tables:**
- Fig. 35: Learning curves (return and cost) for PPO‑Lag vs PPOLagSem (enable=False).
- Table 21: Parity test results (effect sizes, p-values, confidence intervals).
4.8 Early Sample Efficiency Gains (learning curves, AUC tables)
**Key Points:** Report % improvement at fixed step thresholds; faster crossing of performance landmarks.
**Notes:** Include AUC table.
**Suggested Figures & Tables:**
- Fig. 36: Early-phase learning curves with step thresholds marked.
- Table 22: AUC improvements at K steps across seeds (mean ± CI).
4.9 Constraint Stability & Modulation Impact (λ variance, cost stats)
**Key Points:** Modulation lowers λ variance & violation rate.
**Notes:** Table summarizing metrics.
**Suggested Figures & Tables:**
- Fig. 37: λ trajectories and variance across seeds (with/without modulation).
- Table 23: Violation rates, cost means/variances by condition.
4.10 Risk Head Effectiveness (Loss/Corr trajectories; scheduling impact)
**Key Points:** Rising correlation; loss plateau indicates convergence; schedule impacts speed.
**Notes:** Scatter plot predicted vs realized cost.
**Suggested Figures & Tables:**
- Fig. 38: Risk loss and correlation over training.
- Fig. 39: Scatter of predicted vs realized near-term cost (with line of best fit).
4.11 Ablations
- Beta schedule
- Margin normalization & scaling
- Risk horizon & mini-batch scheduling
- Modulation parameter sweeps
- Backend model choice (CLIP vs SigLIP variants)
**Key Points:** Identify most sensitive hyperparameters; show robustness.
**Notes:** Consistent visualization style.
**Suggested Figures & Tables:**
- Fig. 40: Ablation panels (β schedule, margin normalization, H_risk, modulation τ/k).
- Table 24: Best configs per ablation with mean return/cost and overhead.
4.12 Overhead & Latency Analysis (profiling breakdown)
**Key Points:** Overhead within <10% target; breakdown by component.
**Notes:** Provide profiling table. Use `Semantics/EmbedLatencyMs` and steps/sec deltas; report capture_interval effective and batch sizes.
**Suggested Figures & Tables:**
- Fig. 41: Steps/sec vs semantic settings; highlight <10% overhead region.
- Table 25: Profiling breakdown by component and configuration.
4.13 Robustness (noise injections, NaN event counts)
**Key Points:** System degrades gracefully; fallback triggers rare.
**Notes:** Margin distribution shift figure.
**Suggested Figures & Tables:**
- Fig. 42: Robustness curves under noise injections (return/cost impact).
- Table 26: NaN/Inf event counts and fallback activations.
4.14 Prompt Sensitivity & Generalization
**Key Points:** Performance stable across prompt variants.
**Notes:** Include alternate prompt results.
**Suggested Figures & Tables:**
- Fig. 43: Performance across alternative prompt sets.
- Table 27: Prompt variants and corresponding centroid similarity stats.
4.15 Summary Table (best configs vs baseline)
**Key Points:** Consolidated gains & overhead; highlight neutrality maintained.
**Notes:** Bold significant improvements.
**Suggested Figures & Tables:**
- Table 28: Summary of best-performing configurations vs baseline (return, cost, overhead).

4.16 Discussion
4.16.1 Interpreting the Gains: Where They Come From
**Key Points:** Curriculum from semantics + stabilized safety pressure.
**Notes:** Tie to earlier variance claims.
4.16.2 Trade-offs: Complexity vs Performance vs Neutrality
**Key Points:** Added modules vs maintainability; overhead justified by gains.
**Notes:** Potential simplifications future work.
4.16.3 When Semantic Shaping Helps vs Hurts
**Key Points:** Helps in visually grounded tasks; risk with misaligned prompts or domain shift.
**Notes:** Reference failure case appendix.
4.16.4 Limitations (frozen encoder, prompt dependency, one-sided modulation)
**Key Points:** Static encoder may underperform in domain shift; prompts handcrafted.
**Notes:** Suggest adaptive prompt learning.
4.16.5 Threats to Validity (environment diversity, hyperparameter overfitting, statistical power)
**Key Points:** Limited task diversity; potential tuning bias.
**Notes:** Plan additional unseen env tests.

## 11. Future Work
11.1 Distributional / Uncertainty-Aware Risk Head
**Key Points:** Move to quantile/CVaR predictions for tail safety.
**Notes:** Align with modulation improvements.
11.2 Adaptive Capture Interval & β Scheduling
**Key Points:** Dynamic compute allocation based on embedding utility.
**Notes:** Use latency & shaping efficacy metrics.
11.3 Automated Prompt Optimization & Curriculum
**Key Points:** Search prompts to maximize early AUC while preserving neutrality.
**Notes:** Could use evolutionary / RL search.
11.4 Semantic Embedding Fusion into Policy (Late / Mid-level)
**Key Points:** Direct feature fusion may further accelerate learning.
**Notes:** Warn about non-stationarity.
11.5 Distillation & Lightweight On-Policy Encoder
**Key Points:** Distill to smaller model for latency reduction.
**Notes:** Track accuracy vs speed trade-off.
11.6 Embedding-Based Novelty / Exploration Bonuses
**Key Points:** Semantic novelty can complement safety shaping.
**Notes:** Compare vs RND/ICM.
11.7 Distributional Safety Metrics (CVaR shaping)
**Key Points:** Tail-aware shaping could tighten safety guarantees.
**Notes:** Requires distributional model.

## Chapter 5: Conclusion and Future Work
5.1 Summary of Contributions and Findings
**Key Points:** Restate problem & approach; empirical findings; neutrality + extensibility.

5.2 Limitations
**Key Points:** Frozen encoder; prompt dependence; gating assumptions; environment scope.

5.3 Future Work
- Restate problem & proposed approach
- Highlight core empirical findings
- Emphasize neutrality + extensibility
- Vision for integration of broader foundation model priors in safe RL
**Key Points:** Summarize gains, neutrality, modularity, future horizon.
**Notes:** No new claims; keep concise.

## Scope Status: Implemented vs Pending
Implemented (done):
- Semantic shaping (additive and potential-based) with cosine-annealed β, normalization and clamping; neutrality telemetry (ShapingRewardRatio, ShapingStd).
- Architecture and wiring: `PPOLagSem` variant with `SemanticOnPolicyAdapter` and `SemanticManager`; baseline parity when disabled.
- VLM backend generalization (CLIP → SigLIP via AutoModel/AutoProcessor), safetensors, dtype fallback (bf16→fp16→fp32).
- Performance controls: capture interval, temporal pooling, spatial batching with OOM backoff.
- Risk head: truncated discounted horizon, episode-aware masking, reverse rollout, SmoothL1 loss, mini-batch scheduling; float32 stability and dtype alignment.
- Telemetry & robustness: Semantics/* and Risk/* logging, NaN/Inf sanitation, fail-soft fallbacks.
- Modulation (experimental): LR scaling via risk-quantile gate with episode-count gating; `Risk/ModulationScale` logged.
- Offline probe tooling for prompt/centroid analysis and annotated videos.

Pending (not done or experimental):
- Figures/tables generation and scripted regeneration from logs (parity, AUC improvements, overhead breakdown).
- Modulation refinements: EMA smoothing, correlation-quality gating, acceleration option, per-env gating.
- Adaptive schedules: auto-tuned capture_interval and β schedule.
- Prompt automation & uncertainty gating by default.
- Risk head: distributional outputs (quantile/CVaR) and integrated calibration training.
- Deployment path: policy feature fusion and student distillation to remove online VLM dependence.
- Testing & reproducibility: broader unit tests (margin/clamp, beta schedule, dtype fallback, batching parity, risk targets); config/version+prompt hash snapshots.
- Performance: async embedding worker, optional temporal micro-batching, cross-process semantic stats.

## Reproducibility & Artifact Checklist
- Code commit hash
- Conda/pip environment (versions)
- Config files & seeds list
- Prompt sets (safe/unsafe) with hashes
- Scripts for figure regeneration
- Raw logs & plotting notebook references
**Key Points:** Full environment + prompt hashing ensures replicability. Include `clip_probe/centroids.json` (or script to regenerate), YAML config snapshot (`omnisafe/configs/on-policy/PPOLagSem.yaml`), and experiment command scripts.
**Notes:** Provide hash algorithm details.
**Suggested Figures & Tables:**
- Table 29: Artifact manifest (file, description, path/hash) and figure regeneration script mapping.

## Ethics & Broader Impact (optional)
- Responsible use of large pretrained models
- Safety shaping misuse considerations
**Key Points:** Potential misuse via biased semantics; transparency mitigations.
**Notes:** Address dataset bias & fairness.

## Appendices
A. Detailed Derivations (potential shaping algebra; modulation bound)  
B. Extended Hyperparameter Tables  
C. Additional Learning Curves & Ablation Figures  
D. Risk Target Pseudocode & Complexity  
E. Prompt Set Variants & Margin Statistics  
F. Failure Case Studies (NaN events, extreme scaling)  
G. License & Third-Party Model Attributions
**Key Points:** Expanded proofs, tables, robustness artifacts.
**Notes:** Keep main text lean; reference selectively.
**Suggested Figures & Tables:**
- Appendix figures A–G: Extended derivations (A), hyperparameter tables (B), additional curves (C), risk pseudocode block (D), prompt variant galleries and margin histograms (E), failure case screenshots and logs (F), license/attribution table (G).

H. Unit Test Plan  
- Margin normalization and clamp behavior, beta schedule endpoints, dtype fallback chain  
- Spatial batching parity (batch vs single), shaping broadcast correctness  
- Risk head target correctness on synthetic episodes, correlation sanity checks

## Glossary (consolidated)
- Provide quick-reference of symbols & terms.
**Key Points:** Alphabetized cross-referenced symbols & terms.
**Notes:** Consider auto-generation script.

## References
- Academic citations for safe RL, shaping, VLMs, auxiliary learning, modulation techniques.
**Key Points:** Comprehensive coverage incl. recent works; consistent style.
**Notes:** Use reference manager for consistency.

---
Placeholder outline subject to refinement after initial experimental consolidation.
