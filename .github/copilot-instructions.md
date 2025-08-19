# OmniSafe AI Assistant Instructions

Concise, project‑specific rules (semantic PPOLagSem fork). Keep edits small, parity‑safe, and testable.

## 1. Essentials
- Algorithms live under `omnisafe/algorithms/**`; on‑policy variants use an Adapter for rollouts (`OnPolicyAdapter`).
- `PPOLagSem` only differs from `PPOLag` by injecting `SemanticOnPolicyAdapter` + `SemanticManager` when `semantic_cfgs.enable`.
- Data path (on‑policy): Adapter.rollout → buffer.store → algorithm update; semantics hook only touches reward before store & logs metrics.

## 2. Implemented Semantic Features (DONE)
- CLIP load with safetensors + dtype fallback (bf16→fp16→fp32) & status codes.
- Prompt centroids (safe/unsafe) + cosine margin normalization + cosine‑annealed shaping beta (clamped margin to [-2,2]).
- Optional per‑env spatial batching (`batch_across_envs`, `batch_max`, OOM backoff) with vectorized shaping.
- Embedding & cost deques (risk buffer scaffold), latency + success counters.

## 3. Incomplete / Planned (TOP TODO ORDER)
1. Activate & train risk head (SmoothL1 on truncated discounted cost); log Risk/* metrics.
2. Add shaping distribution stats (std/min/max) & batch size metric.
3. Lagrange modulation via risk quantile gating (`modulation_enable`).
4. Unit tests: margin normalization, beta schedule, dtype fallback, risk correlation, batching parity.
5. Reverse discounted (episode‑aware) risk targets + optional normalization.
6. Temporal / async embedding micro‑batching (only if profiling justifies).

## 4. Config & CLI Pattern
- Nested override: `--semantic-cfgs:enable True --semantic-cfgs:shaping_enable True`.
- Key flags: `enable`, `shaping_enable`, `risk_enable`, `modulation_enable`, `capture_interval`, `model_device`, `host_device`, `batch_across_envs`, `batch_max`, prompts.
- Parity contract: with `enable=False` no semantic side effects (must remain true after changes).

## 5. Logging Keys
- Semantics: `Semantics/Shaping`, `RawReward`, `EmbedLatencyMs`, `EmbedSuccessRate`, `Debug/ClipReady`, `Debug/ClipStatus`, `Debug/EmbedAttempts`, `Debug/EmbedSuccess`.
- Future Risk: `Risk/Loss`, `Risk/PredMean`, `Risk/TargetMean`, `Risk/Corr` (add only when implemented).

## 6. Devices & Dtypes
- CLIP on `model_device`; embeddings/centroids moved to `host_device` (GPU keep reduced precision; CPU cast fp32).
- Never assume bf16 success; check `_clip_status` or `Semantics/Debug/ClipStatus`.
- Place risk head on CLIP device & cast embeddings to its param dtype.

## 7. Pitfalls & Quick Diagnostics
- Config edits “ignored”: overridden by CLI; print `self._cfgs.semantic_cfgs.__dict__` in `_init_env` to verify.
- Zero shaping: check clip ready (1), capture step multiple (`step % capture_interval == 0`), beta not fully annealed.
- High latency: raise `capture_interval`, enable batching, or move model to GPU.
- Missing per‑env shaping: ensure `batch_across_envs=True` and `vector_env_nums>1`.

## 8. Testing Gaps (Add First)
- Populate `tests/test_semantic_spatial_batching.py` with: (a) step count parity batch vs single, (b) shaping broadcast vs per‑env equality when batching disabled.
- Add semantic manager unit tests for: status fallback chain, margin sign (swap prompt lists), beta schedule endpoints.

## 9. Style & Safety
- Absolute imports only; keep line length 100; avoid unrelated reformat.
- New metrics: prefix `Semantics/` or `Risk/`; do not overload existing names.
- Gate every new semantic code path behind `semantic_cfgs` flags.

## 10. Performance Guardrails
- Target embedding overhead <10% wall time; monitor `EmbedLatencyMs` trend.
- Do not increment semantic global step more than once per env step (even with future temporal batching).

## 11. When Extending
- Prefer adapter or manager changes over altering PPO core update; keeps parity & minimizes regression risk.
- Update this file if parity guarantee or logging schema changes.

Feedback: request deeper detail (e.g., risk head training loop) before implementing if unclear.
