#!/usr/bin/env bash
# Run suites of PPOLagSem experiments.
# Usage examples:
#   SEEDS="0 1 2" ENVS="SafetyCarGoal1-v0" ./scripts/run_ppolagsem_suite.sh list
#   STEPS=300000 ./scripts/run_ppolagsem_suite.sh S1
#   ./scripts/run_ppolagsem_suite.sh all
set -euo pipefail

: "${SEEDS:=0 1 2}"
: "${ENVS:=SafetyCarGoal2-v0}"
: "${STEPS:=500000}"
: "${BASE_ALGO:=PPOLagSem}"
: "${BASE_CMD:=python examples/train_policy.py}"
: "${VECTOR_ENVS:=32}"

# Emit a single command (dry-run aware)
emit() {
  local tag="$1"; shift
  local cmd="$*"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "[DRY][$tag] $cmd"
  else
    echo "[RUN][$tag] $cmd"
    eval "$cmd"
  fi
}

# Core builders --------------------------------------------------------------
base_flags() {
  echo "--algo $BASE_ALGO --total-steps $STEPS --train-cfgs:vector_env_nums $VECTOR_ENVS"
}
sem_base() {
  echo "--semantic-cfgs:enable True --semantic-cfgs:shaping_enable True"
}

ppo_baseline_cmd() {
  echo "$BASE_CMD --algo PPOLag --total-steps $STEPS --train-cfgs:vector_env_nums $VECTOR_ENVS"
}
parity_cmd() {
  echo "$BASE_CMD $(base_flags) --semantic-cfgs:enable False"
}

# Blocks ---------------------------------------------------------------------
block_B1() { # PPO-Lag vs parity PPOLagSem
  for e in $ENVS; do
    for s in $SEEDS; do
  emit B1-baseline "$BASE_CMD --algo PPOLag --env-id $e --seed $s --total-steps $STEPS --train-cfgs:vector_env_nums $VECTOR_ENVS"
  emit B1-parity   "$BASE_CMD --algo $BASE_ALGO --env-id $e --seed $s --total-steps $STEPS --train-cfgs:vector_env_nums $VECTOR_ENVS --semantic-cfgs:enable False"
    done
  done
}

block_B2() { # Shaping benefit
  for e in $ENVS; do
    for s in $SEEDS; do
      emit B2-shaping "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base)"
    done
  done
}

block_S1() { # Additive vs Potential
  for e in $ENVS; do
    for s in $SEEDS; do
      for pot in False True; do
        emit S1-pot$pot "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:potential_enable $pot"
      done
    done
  done
}

block_S2() { # Beta schedule sweep
  for e in $ENVS; do
    for s in $SEEDS; do
      for bstart in 0.05 0.15 0.30; do
        for bend in 0.5 0.7 0.9; do
          emit S2-b${bstart}-f${bend} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:beta_start $bstart --semantic-cfgs:beta_end_step_fraction $bend"
        done
      done
    done
  done
}

block_S3() { # Margin norm & scale
  for e in $ENVS; do
    for s in $SEEDS; do
      for norm in True False; do
        for scale in 0.5 1.0 1.5; do
          emit S3-n${norm}-m${scale} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:margin_norm_enable $norm --semantic-cfgs:margin_scale $scale"
        done
      done
    done
  done
}

block_R1() { # Risk schedule grid
  for e in $ENVS; do
    for s in $SEEDS; do
      for mins in 5 50; do
        for bsz in 0 128; do
          for it in 1 3; do
            emit R1-m${mins}-b${bsz}-i${it} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:risk_min_samples $mins --semantic-cfgs:risk_batch_size $bsz --semantic-cfgs:risk_update_iters $it"
          done
        done
      done
    done
  done
}

block_R2() { # Risk horizon
  for e in $ENVS; do
    for s in $SEEDS; do
      for H in 32 64 128; do
        emit R2-H${H} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:risk_horizon $H"
      done
    done
  done
}

block_R3() { # Episode mask toggle
  for e in $ENVS; do
    for s in $SEEDS; do
      for mask in True False; do
        emit R3-mask$mask "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:risk_episode_mask_enable $mask"
      done
    done
  done
}

block_M1() { # Modulation enable
  for e in $ENVS; do
    for s in $SEEDS; do
      for en in False True; do
        emit M1-mod$en "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:modulation_enable $en"
      done
    done
  done
}

block_M2() { # Alpha & percentile
  for e in $ENVS; do
    for s in $SEEDS; do
      for alpha in 1.0 2.0 4.0; do
        for perc in 50 60 70; do
          emit M2-a${alpha}-p${perc} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:modulation_enable True --semantic-cfgs:alpha_modulation $alpha --semantic-cfgs:threshold_percentile $perc"
        done
      done
    done
  done
}

block_M3() { # Modulation warmup episodes
  for e in $ENVS; do
    for s in $SEEDS; do
      for warm in 0 20 50; do
        emit M3-w${warm} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:modulation_enable True --semantic-cfgs:modulation_min_episodes $warm"
      done
    done
  done
}

block_V1() { # Model family
  for e in $ENVS; do
    for s in $SEEDS; do
      for model in openai/clip-vit-base-patch16 google/siglip-so400m-patch14-384; do
        emit V1-${model//\//_} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:model_name $model"
      done
    done
  done
}

block_V2() { # Capture interval
  for e in $ENVS; do
    for s in $SEEDS; do
      for interval in 2 4 8; do
        emit V2-c${interval} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:capture_interval $interval"
      done
    done
  done
}

block_L1() { # batch_max
  for e in $ENVS; do
    for s in $SEEDS; do
      for bm in 16 32 64; do
        emit L1-bm${bm} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:batch_max $bm"
      done
    done
  done
}

block_L2() { # temporal_window
  for e in $ENVS; do
    for s in $SEEDS; do
      for tw in 1 3 5; do
        emit L2-tw${tw} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:temporal_window $tw"
      done
    done
  done
}

block_Robust() { # Stress tests
  for e in $ENVS; do
    for s in $SEEDS; do
      emit F1-scale3 "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:margin_scale 3.0"
      for lr in 5e-4 1e-3 5e-3; do
        emit F2-risklr${lr//e/} "$BASE_CMD $(base_flags) --env-id $e --seed $s $(sem_base) --semantic-cfgs:risk_enable True --semantic-cfgs:risk_lr $lr"
      done
    done
  done
}

blocks_all=(B1 B2 S1 S2 S3 R1 R2 R3 M1 M2 M3 V1 V2 L1 L2 Robust)

list_blocks() {
  printf "%s\n" "${blocks_all[@]}"
}

run_block() {
  local blk=$1
  local fn="block_${blk}"
  if declare -f "$fn" >/dev/null 2>&1; then
    echo "# Executing block $blk" >&2
    $fn
  else
    echo "Unknown block: $blk" >&2
    exit 1
  fi
}

case "${1:-list}" in
  list)
    list_blocks ;;
  all)
    for b in "${blocks_all[@]}"; do run_block "$b"; done ;;
  *)
    run_block "$1" ;;
 esac
