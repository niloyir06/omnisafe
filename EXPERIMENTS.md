# PPOLagSem Experiment Suite

Structured set of experiments covering parity, shaping, risk scheduling, modulation, backend choice, latency, robustness, prompts, scaling, and stress tests. Each experiment defines: purpose, configuration deltas, metrics of interest.

Use the script `scripts/run_ppolagsem_suite.sh` to generate or execute commands.

## Global Defaults
- Algo: `PPOLagSem`
- Steps: 500000 (adjust via `STEPS` env)
- Seeds: 3 (set `SEEDS="0 1 2"`)
- Environment(s): `SafetyCarGoal2-v0` (extend using `ENVS`)

Baseline flags (parity): `--semantic-cfgs:enable False`
Full semantics base: `--semantic-cfgs:enable True --semantic-cfgs:shaping_enable True`

Risk add-on: `--semantic-cfgs:risk_enable True`
Modulation add-on: `--semantic-cfgs:modulation_enable True`
Potential shaping: `--semantic-cfgs:potential_enable True`

## 1. Baseline & Parity
| ID | Description | Config Delta |
|----|-------------|--------------|
| B1 | PPO-Lag vs PPOLagSem parity | (Baseline) `--algo PPOLag`, (Parity) `--algo PPOLagSem --semantic-cfgs:enable False` |
| B2 | Shaping benefit early phase | `--semantic-cfgs:enable True --semantic-cfgs:shaping_enable True` |

Metrics: return curve AUC (first 20%), cost violation rate, steps/sec.

Full commands (environment fixed to SafetyCarGoal2-v0, fixed seed 42):
```
# B1 baseline
python examples/train_policy.py --algo PPOLag --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42
# B1 parity (semantics disabled)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 --semantic-cfgs:enable False --vector-env-nums 16
# B2 shaping
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --vector-env-nums 16
```

## 2. Shaping Ablations
| ID | Factor | Grid |
|----|--------|------|
| S1 | Additive vs Potential | `potential_enable` ∈ {False,True} |
| S2 | Beta schedule | `beta_start` ∈ {0.05,0.15,0.30}; `beta_end_step_fraction` ∈ {0.5,0.7,0.9} |
| S3 | Margin normalization & scale | `margin_norm_enable` ∈ {True,False}; `margin_scale` ∈ {0.5,1.0,1.5} |

Metrics: ShapingRewardRatio decay shape, final neutrality (last 10% returns/costs), ClampFrac.
Full commands (examples; enumerate grid as needed):
```
# S1 additive
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:potential_enable False
# S1 potential
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:potential_enable True

# S2 example (beta_start=0.15, beta_end_step_fraction=0.7)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True \
  --semantic-cfgs:beta_start 0.15 --semantic-cfgs:beta_end_step_fraction 0.7

# S3 example (no norm, margin_scale=1.5)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True \
  --semantic-cfgs:margin_norm_enable False --semantic-cfgs:margin_scale 1.5
```

## 3. Risk Scheduling
| ID | Factor | Grid |
|----|--------|------|
| R1 | Mini-batch & iters | `risk_min_samples` ∈ {5,50}; `risk_batch_size` ∈ {0,128}; `risk_update_iters` ∈ {1,3} |
| R2 | Horizon sweep | `risk_horizon` ∈ {32,64,128} |
| R3 | Episode mask | `risk_episode_mask_enable` ∈ {True,False} |

Metrics: Risk/Loss, Risk/Corr, Risk/TrainSamples, overhead.

Full commands:
```
# R1 example (risk_min_samples=50, risk_batch_size=128, risk_update_iters=3)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True \
  --semantic-cfgs:risk_min_samples 50 --semantic-cfgs:risk_batch_size 128 --semantic-cfgs:risk_update_iters 3

# R2 example (risk_horizon=128)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True \
  --semantic-cfgs:risk_horizon 128

# R3 mask off
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True \
  --semantic-cfgs:risk_episode_mask_enable False
```

## 4. Modulation
| ID | Factor | Grid |
|----|--------|------|
| M1 | Enable vs Disable | `modulation_enable` ∈ {False,True} |
| M2 | Alpha & percentile | `alpha_modulation` ∈ {1.0,2.0,4.0}; `threshold_percentile` ∈ {50,60,70} |
| M3 | Warmup episodes | `modulation_min_episodes` ∈ {0,20,50} |

Metrics: λ variance, episodic cost std, ModulationScale distribution.

Full commands:
```
# M1 modulation enabled
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True \
  --semantic-cfgs:modulation_enable True

# M2 example (alpha_modulation=4.0, threshold_percentile=70)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True \
  --semantic-cfgs:modulation_enable True --semantic-cfgs:alpha_modulation 4.0 --semantic-cfgs:threshold_percentile 70

# M3 warmup (modulation_min_episodes=50)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True \
  --semantic-cfgs:modulation_enable True --semantic-cfgs:modulation_min_episodes 50
```

## 5. Backend Model & Capture
| ID | Factor | Grid |
|----|--------|------|
| V1 | Model family | `model_name` ∈ {`openai/clip-vit-base-patch16`,`google/siglip-so400m-patch14-384`} |
| V2 | Capture interval | `capture_interval` ∈ {2,4,8} × chosen model |

Metrics: steps/sec, EmbedLatencyMs, ShapingRewardRatio, final return.

Full commands:
```
# V1 SigLIP
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True \
  --semantic-cfgs:model_name google/siglip-so400m-patch14-384

# V2 capture_interval=8
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True \
  --semantic-cfgs:capture_interval 8
```

## 6. Latency / Temporal Pooling
| ID | Factor | Grid |
|----|--------|------|
| L1 | Spatial batch size | `batch_max` ∈ {16,32,64} |
| L2 | Temporal window | `temporal_window` ∈ {1,3,5} |

Metrics: EmbedLatencyMs, NormMargin variance, ClampFrac.

Full commands:
```
# L1 batch_max=64
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:batch_max 64

# L2 temporal_window=5
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:temporal_window 5
```

## 7. Robustness & Numerical
| ID | Factor | Grid |
|----|--------|------|
| N1 | Precision | (encoder fp16 vs force fp32) (config if implemented) |
| N2 | Noise injection | External env wrapper adding Gaussian noise probability 0.1 |

Metrics: Risk/NanEvents, Risk/Loss stability, return degradation.

Full commands:
```
# N2 noise injection example (assuming external wrapper flag --env-noise-prob 0.1)
python examples.train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --env-noise-prob 0.1
```

## 8. Prompt Sensitivity
| ID | Factor | Grid |
|----|--------|------|
| P1 | Prompt sets | safe/unsafe set A vs B |
| P2 | Ablated discriminative prompt | Remove top-margin prompt |

Metrics: NormMargin statistics, early AUC, ShapingRewardRatio.

Full commands:
```
# P1 prompt set B (assuming alt YAML or override file)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --config alt_prompts.yaml

# P2 ablated prompt (assuming flag --semantic-cfgs:ablated_prompt_idx 0)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:ablated_prompt_idx 0
```

## 9. Combined Benefit
| ID | Description | Notes |
|----|-------------|-------|
| C1 | Best shaping + best risk schedule + best modulation | Compare vs shaping-only, risk-only, baseline |
| C2 | Potential shaping neutrality check | Swap additive with potential using best β |

## 10. Scaling & Windows
| ID | Factor | Grid |
|----|--------|------|
| W1 | Buffer size | `window_size` ∈ {1024,2048,4096} |
| W2 | Norm window | `norm_window` ∈ {250,500,1000} |

Metrics: memory (approx), Risk/Loss plateau speed, NormMargin stability.

Full commands:
```
# W1 window_size=4096
python examples.train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed 42 \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:window_size 4096

# W2 norm_window=250
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed $RANDOM \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:norm_window 250
```

## 11. Stress / Failure Modes
| ID | Factor | Grid |
|----|--------|------|
| F1 | Extreme margin scale | `margin_scale`=3.0 |
| F2 | Aggressive risk LR | `risk_lr` ∈ {5e-4,1e-3,5e-3} |

Metrics: NanEvents, divergence incidence, ClampFrac.

Full commands:
```
# F1 extreme margin_scale
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed $RANDOM \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:margin_scale 3.0

# F2 aggressive risk LR (5e-3)
python examples/train_policy.py --algo PPOLagSem --env-id SafetyCarGoal2-v0 --total-steps 500000 --seed $RANDOM \
  --semantic-cfgs:enable True --semantic-cfgs:shaping_enable True --semantic-cfgs:risk_enable True --semantic-cfgs:risk_lr 5e-3
```

## Metrics to Collect
- Return, Cost (mean, violation rate)
- ShapingRewardRatio curve
- Risk/Loss, Risk/Corr, Risk/TrainSamples, Risk/NanEvents
- ModulationScale (if enabled)
- Margin stats (mean/std, ClampFrac)
- EmbedLatencyMs, steps/sec
- λ variance

## Command Pattern
```
python examples/train_policy.py \
  --algo PPOLagSem \
  --env-id <ENV> \
  --total-steps ${STEPS:-500000} \
  --seed <SEED> \
  --semantic-cfgs:enable True \
  --semantic-cfgs:shaping_enable True \
  [additional semantic cfg overrides]
```

## Running via Script
```
chmod +x scripts/run_ppolagsem_suite.sh
SEEDS="42" ENVS="SafetyCarGoal2-v0" ./scripts/run_ppolagsem_suite.sh list   # show commands
SEEDS="42" ENVS="SafetyCarGoal2-v0" ./scripts/run_ppolagsem_suite.sh B1    # run specific block
./scripts/run_ppolagsem_suite.sh all                                          # run everything (careful!)
```

## Adding New Experiments
Edit `run_ppolagsem_suite.sh`, add a function `block_<ID>()` that echoes commands via `emit` helper.

---
This document should remain succinct; detailed rationale resides in `thesis.md` and `progress.md`.
