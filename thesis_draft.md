# Thesis Draft Outline: Semantic & Risk-Augmented Safe Reinforcement Learning (PPOLagSem)

## 0. Metadata
- Working Title: Accelerating Safe Reinforcement Learning with Frozen Vision–Language Semantics and Auxiliary Risk Prediction
- Author: <Your Name>
- Date: <To fill>
- Repository Commit Hash: <To fill at submission>

## 1. Abstract (1 paragraph)
- Motivation: Slow safe RL convergence; underutilized pretrained semantics.
- Approach: Integrate frozen vision–language encoder + semantic reward shaping + auxiliary risk predictor + adaptive constraint modulation.
- Key Results: Faster early learning, stable constraint adherence, low overhead, model-agnostic (CLIP/SigLIP).
- Contributions: (i) unified semantic shaping & risk framework, (ii) episode-aware discounted risk formulation, (iii) modulation via risk quantile gap, (iv) robust NaN-safe training pipeline.

**Key Points:**
- Condense to ~180–220 words highlighting problem, method, results, contributions.
- Emphasize neutrality (no degradation when disabled) + low overhead.
- Quantify gains (e.g., X% AUC improvement, Y% violation reduction) once results finalized.
- Stress model-agnostic usage of frozen VLM encoder.

**Notes:**
- Add concrete metrics after experiments finalize.
- Ensure contribution phrasing matches numbered list in Introduction 2.5.
- Define any specialized terms if used (semantic margin, risk modulation).

## 2. Introduction
2.1 Problem Setting & Challenges in Safe RL
**Key Points:** Safe RL is sample-inefficient; constraint cost signals sparse/delayed; balancing exploration vs safety leads to slow convergence.
**Notes:** Cite canonical benchmarks; clarify distinction cost variance vs mean.
2.2 Opportunity: Pretrained Vision–Language Priors
**Key Points:** Frozen VLMs encode object & relational semantics zero-shot; can yield dense safety context.
**Notes:** Briefly contrast CLIP vs SigLIP alignment.
2.3 Limitations of Existing Shaping & Risk Estimation Approaches
**Key Points:** Manual shaping brittle; prior semantic shaping not safety-targeted; risk estimators cold-start unstable.
**Notes:** Include 2–3 representative citations.
2.4 High-Level Idea & Intuition
**Key Points:** Use semantic margin as curriculum shaped by annealed β; risk head predicts future cost; modulation adapts constraint pressure.
**Notes:** Add running car hazard illustrative example.
2.5 Contributions (bullet list) – numbered, mapping to later sections
**Key Points:** Precise 4–6 contributions (semantic shaping framework, risk horizon target, quantile modulation, neutrality, robust pipeline).
**Notes:** Numbering must match Abstract & Conclusion.
2.6 Paper Structure / Chapter Roadmap
**Key Points:** One sentence per major section summarizing role.
**Notes:** Keep ≤6 lines.

## 3. Background
3.1 Safe Reinforcement Learning & Constrained MDPs
**Key Points:** Formal CMDP; constraint via expected cumulative cost ≤ d; Lagrangian relaxation.
**Notes:** Introduce symbols reused later (c_t, J_c, λ).
3.2 PPO-Lagrangian Fundamentals
**Key Points:** PPO clipped surrogate + λ update; entropy bonus stabilizes policy updates.
**Notes:** Mention convergence heuristics.
3.3 Reward Shaping & Potential-Based Shaping Theory
**Key Points:** Potential shaping preserves optimal policy; additive shaping can bias but speeds exploration.
**Notes:** Provide Ng et al. lemma reference.
3.4 Auxiliary Prediction Tasks in RL (Value/Risk/Critic augmentation)
**Key Points:** Auxiliary heads improve representation stability; risk prediction distinct from value baseline.
**Notes:** Clarify difference between cost value vs truncated horizon risk.
3.5 Vision–Language Models (CLIP, SigLIP) & Embedding Properties
**Key Points:** Contrastive embeddings enable zero-shot similarity; pros (transfer) vs cons (bias, resolution limits).
**Notes:** Cite zero-shot accuracy benchmarks.
3.6 Notation Summary Table
**Key Points:** Consolidate symbols: s,a,r,c, m (margin), φ potential, β schedule, λ, H_risk.
**Notes:** Table before heavy equations.

## 4. Related Work
4.1 Safe RL with Auxiliary Signals
**Key Points:** Prior methods add critics/shields; ours integrates semantic shaping + risk modulation.
**Notes:** Stress novelty of VLM semantics in safety context.
4.2 Semantic or Visual Priors in RL
**Key Points:** Use of pretrained backbones/prompts; gap: lack of safety-focused integration.
**Notes:** Optionally taxonomy figure.
4.3 Risk Prediction & Uncertainty Estimation
**Key Points:** Distributional RL & ensembles; our simpler truncated cost predictor offers low overhead.
**Notes:** Position as pragmatic baseline.
4.4 Reward Shaping Methodologies (Potential vs Additive vs Curriculum)
**Key Points:** Neutrality vs speed trade-off; annealing bridges; curriculum parallels margin schedule.
**Notes:** Cross-reference Section 5.4.
4.5 Constraint Adaptation & Lagrange Multiplier Tuning
**Key Points:** Standard gradient ascent vs heuristic adaptors; introduce quantile logistic gating.
**Notes:** Mention bounded derivative advantage.
4.6 Positioning: How PPOLagSem Differs
**Key Points:** Unified minimal-intrusion pipeline, neutrality when disabled, modular flags.
**Notes:** Prepare comparison table (maybe Appendix).

## 5. Method
5.1 System Overview & Data Flow Diagram
**Key Points:** Pipeline: env → capture → embedding → margin → shaping/risk/modulation → PPO update; capture_interval reduces cost.
**Notes:** Insert overview figure.
5.2 Semantic Manager
- Prompt design, centroid embedding, normalization strategy
- Temporal pooling & capture interval
**Key Points:** Precompute centroids; running margin stats; temporal pooling smooths noise.
**Notes:** Document prompt selection heuristics.
5.3 Margin Computation & Processing
- Raw margin definition m = cos(e, μ_safe) − cos(e, μ_unsafe)
- Running z-normalization & clamping; margin scale parameter
**Key Points:** Scale then optional normalize; clamp [-2,2] to bound shaping; track clamp fraction.
**Notes:** Provide histogram later.
5.4 Reward Shaping Mechanisms
- Additive shaping formula & annealed β schedule
- Potential-based shaping derivation & neutrality proof sketch
- Shaping neutrality monitoring (ratio metrics)
**Key Points:** Cosine/capped schedule drives β→0; potential form ensures invariance; monitor shaping/total reward ratio.
**Notes:** Include explicit equations.
5.5 Auxiliary Risk Prediction
- Target: episode-aware truncated discounted cost horizon
- Horizon parameter & masking algorithm pseudocode
- Loss (SmoothL1) & mini-batch multi-iter schedule
- NaN/Inf sanitation & float32 casting rationale
**Key Points:** Truncated discounted cost resets at terminals; SmoothL1 robust to spikes; iterative updates improve fit.
**Notes:** Ablate horizon length & batch size.
5.6 Modulation of Lagrange Multiplier Updates
- Quantile gap logistic gating derivation
- Stability safeguards (episode min gate, clamping)
**Key Points:** Compute risk percentile threshold; logistic gating scales λ update; safeguards prevent oscillation.
**Notes:** Provide pseudocode & complexity.
5.7 Precision & Device Strategy
- Mixed precision for encoder vs float32 risk head
- Memory & latency trade-offs
**Key Points:** bf16/fp16→fp32 fallback; embeddings moved to CPU; risk head colocated for efficiency.
**Notes:** Justify no encoder finetuning.
5.8 Logging & Telemetry Design
- Key metrics categories (semantic, shaping, risk, modulation, performance)
**Key Points:** Metrics enable early failure detection (latency, clamp frac, risk corr).
**Notes:** Map metrics to debugging actions.
5.9 Computational Complexity & Overhead Analysis
**Key Points:** Encoder dominates; overhead amortized by interval & batching; target <10% wall time.
**Notes:** Provide measured ms/frame table.
5.10 Failure & Degradation Modes (graceful fallback design)
**Key Points:** Failures trigger automatic disable; NaN detection resets shaping; logging ensures visibility.
**Notes:** Enumerate trigger conditions.

## 6. Theoretical Considerations
6.1 Potential-Based Neutrality Argument
**Key Points:** Telescoping potential differences cancel in episodic return.
**Notes:** Reference Ng et al. 1999 theorem.
6.2 Bias Analysis of Additive Shaping with Annealing
**Key Points:** Early bias proportional to β; integral of β over time bounds cumulative bias.
**Notes:** Provide schedule integral bound.
6.3 Variance Reduction via Risk Auxiliary Signal (qualitative)
**Key Points:** Shared representation stabilizes value gradients; risk head encourages disentangled safety features.
**Notes:** Optional empirical variance figure.
6.4 Stability of Quantile-Based Modulation (bounded derivative)
**Key Points:** Logistic gating derivative bounded ⇒ prevents λ explosion; percentile adapts to distribution shifts.
**Notes:** Show derivative max.
6.5 Discussion of Convergence Guarantees (limitations)
**Key Points:** Non-stationarity (annealing, modulation) complicates formal guarantees; neutrality partially mitigates.
**Notes:** Scope limited to empirical validation.

## 7. Implementation
7.1 Code Architecture Mapping (module diagram)
**Key Points:** Adapter pattern injects semantics without altering PPO core.
**Notes:** Provide module diagram.
7.2 Configuration Surface & Defaults Table
**Key Points:** Flags gate each feature; conservative defaults maintain parity when disabled.
**Notes:** Cross-check CLI vs YAML overrides.
7.3 Buffer Structures & Memory Footprint Estimation
**Key Points:** Deques bounded; memory O(window * dim); negligible vs model.
**Notes:** Include concrete example numbers.
7.4 Mini-Batching & Iterative Risk Updates
**Key Points:** Multiple iters refine fit; optional mini-batch balances compute.
**Notes:** Pseudocode snippet.
7.5 Numerical Safeguards (sanitization, event counters)
**Key Points:** Replace NaN/Inf; counters logged; prevents silent corruption.
**Notes:** Table of safeguards.
7.6 Reproducibility Hooks (seeds, version logging, prompt hashing)
**Key Points:** Prompt hashing ensures experiment provenance.
**Notes:** Document seed utilities.

## 8. Experimental Setup
8.1 Environments & Safety Constraints
**Key Points:** List tasks, cost thresholds, horizon specifics.
**Notes:** Provide summary table.
8.2 Baselines & Ablations (baseline PPO-Lag, shaping-only, risk-only, full, potential shaping)
**Key Points:** Isolate effect of each component.
**Notes:** Pre-register seeds.
8.3 Hyperparameter Grid Summary
**Key Points:** Focus on β schedule, margin_scale, risk_horizon, modulation params.
**Notes:** Report total runs & compute budget.
8.4 Metrics & Evaluation Criteria
- Sample efficiency (AUC early phase)
- Constraint violation rate & cost variance
- Shaping reward ratio decay
- Risk prediction accuracy (Loss, Corr)
- Overhead (steps/sec change)
**Key Points:** Metrics map directly to contributions; AUC captures early efficiency.
**Notes:** Specify statistical tests (bootstrap CI, nonparametric tests).
8.5 Hardware & Runtime Environment
**Key Points:** Report GPU/CPU; justify device placement choices.
**Notes:** Provide container spec.
8.6 Statistical Methodology (seeds, CI computation, tests)
**Key Points:** Seed count justification; CI method (BCa bootstrap?); multiple comparison handling.
**Notes:** Address reproducibility.

## 9. Results
9.1 Baseline Parity (disabled semantics) – validation of neutrality path
**Key Points:** Curves overlap with baseline; statistical tests show no significant difference.
**Notes:** Provide p-values / effect sizes.
9.2 Early Sample Efficiency Gains (learning curves, AUC tables)
**Key Points:** Report % improvement at fixed step thresholds; faster crossing of performance landmarks.
**Notes:** Include AUC table.
9.3 Constraint Stability & Modulation Impact (λ variance, cost stats)
**Key Points:** Modulation lowers λ variance & violation rate.
**Notes:** Table summarizing metrics.
9.4 Risk Head Effectiveness (Loss/Corr trajectories; scheduling impact)
**Key Points:** Rising correlation; loss plateau indicates convergence; schedule impacts speed.
**Notes:** Scatter plot predicted vs realized cost.
9.5 Ablations
- Beta schedule
- Margin normalization & scaling
- Risk horizon & mini-batch scheduling
- Modulation parameter sweeps
- Backend model choice (CLIP vs SigLIP variants)
**Key Points:** Identify most sensitive hyperparameters; show robustness.
**Notes:** Consistent visualization style.
9.6 Overhead & Latency Analysis (profiling breakdown)
**Key Points:** Overhead within <10% target; breakdown by component.
**Notes:** Provide profiling table.
9.7 Robustness (noise injections, NaN event counts)
**Key Points:** System degrades gracefully; fallback triggers rare.
**Notes:** Margin distribution shift figure.
9.8 Prompt Sensitivity & Generalization
**Key Points:** Performance stable across prompt variants.
**Notes:** Include alternate prompt results.
9.9 Summary Table (best configs vs baseline)
**Key Points:** Consolidated gains & overhead; highlight neutrality maintained.
**Notes:** Bold significant improvements.

## 10. Discussion
10.1 Interpreting the Gains: Where They Come From
**Key Points:** Curriculum from semantics + stabilized safety pressure.
**Notes:** Tie to earlier variance claims.
10.2 Trade-offs: Complexity vs Performance vs Neutrality
**Key Points:** Added modules vs maintainability; overhead justified by gains.
**Notes:** Potential simplifications future work.
10.3 When Semantic Shaping Helps vs Hurts
**Key Points:** Helps in visually grounded tasks; risk with misaligned prompts or domain shift.
**Notes:** Reference failure case appendix.
10.4 Limitations (frozen encoder, prompt dependency, one-sided modulation)
**Key Points:** Static encoder may underperform in domain shift; prompts handcrafted.
**Notes:** Suggest adaptive prompt learning.
10.5 Threats to Validity (environment diversity, hyperparameter overfitting, statistical power)
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

## 12. Conclusion
- Restate problem & proposed approach
- Highlight core empirical findings
- Emphasize neutrality + extensibility
- Vision for integration of broader foundation model priors in safe RL
**Key Points:** Summarize gains, neutrality, modularity, future horizon.
**Notes:** No new claims; keep concise.

## 13. Reproducibility & Artifact Checklist
- Code commit hash
- Conda/pip environment (versions)
- Config files & seeds list
- Prompt sets (safe/unsafe) with hashes
- Scripts for figure regeneration
- Raw logs & plotting notebook references
**Key Points:** Full environment + prompt hashing ensures replicability.
**Notes:** Provide hash algorithm details.

## 14. Ethics & Broader Impact (optional)
- Responsible use of large pretrained models
- Safety shaping misuse considerations
**Key Points:** Potential misuse via biased semantics; transparency mitigations.
**Notes:** Address dataset bias & fairness.

## 15. Appendices
A. Detailed Derivations (potential shaping algebra; modulation bound)  
B. Extended Hyperparameter Tables  
C. Additional Learning Curves & Ablation Figures  
D. Risk Target Pseudocode & Complexity  
E. Prompt Set Variants & Margin Statistics  
F. Failure Case Studies (NaN events, extreme scaling)  
G. License & Third-Party Model Attributions
**Key Points:** Expanded proofs, tables, robustness artifacts.
**Notes:** Keep main text lean; reference selectively.

## 16. Glossary (consolidated)
- Provide quick-reference of symbols & terms.
**Key Points:** Alphabetized cross-referenced symbols & terms.
**Notes:** Consider auto-generation script.

## 17. References
- Academic citations for safe RL, shaping, VLMs, auxiliary learning, modulation techniques.
**Key Points:** Comprehensive coverage incl. recent works; consistent style.
**Notes:** Use reference manager for consistency.

---
Placeholder outline subject to refinement after initial experimental consolidation.
