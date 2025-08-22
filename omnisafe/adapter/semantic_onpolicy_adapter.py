"""SemanticOnPolicyAdapter extends OnPolicyAdapter with frame capture & semantic logging.

Adds:
 - Periodic frame capture & CLIP embedding.
 - Semantic shaping reward.
 - Progress bar (rich.track) mirroring base on-policy adapter.
 - Clip readiness & status logging, embedding counters & latency.
"""
from __future__ import annotations

import torch
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.logger import Logger
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.common.semantics.semantic_manager import SemanticManager


class SemanticOnPolicyAdapter(OnPolicyAdapter):  # pragma: no cover minimal logic
    def __init__(self, *args, semantic_manager: SemanticManager, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._semantic_manager = semantic_manager
        self._embed_attempts = 0
        self._embed_success = 0
        super().__init__(*args, **kwargs)

    def rollout(self, steps_per_epoch: int, agent: ConstraintActorCritic, buffer: VectorOnPolicyBuffer, logger: Logger) -> None:  # type: ignore[override]
        # Reset per-episode logging and grab initial observation
        self._reset_log()
        obs, _ = self.reset()
        capture_interval = self._semantic_manager.cfg.capture_interval
        # Seed success rate metric to avoid NaN before any attempts
        if self._semantic_manager.cfg.enable:
            logger.store({'Semantics/EmbedSuccessRate': 0.0})
            # (ClipReady/ClipStatus removed from telemetry)
        batch_mode = self._semantic_manager.cfg.batch_across_envs and getattr(self._env, 'num_envs', 1) > 1
        for step in track(
            range(steps_per_epoch),
            description=f'Processing semantic rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)
            # Preserve raw (environment) reward before semantic shaping for logging & ratios
            base_reward = reward.clone() if torch.is_tensor(reward) else reward
            if self._semantic_manager.cfg.enable and (step % capture_interval == 0):
                # Spatial batch path
                if batch_mode:
                    frames = []
                    # Try to render once per env; fallback to single shared frame if vectorized env doesn't expose per-env render.
                    base_frame = self._env.render()
                    if isinstance(base_frame, list) and len(base_frame) == reward.shape[0]:  # already list of frames
                        frames = base_frame
                    else:
                        # replicate same frame for all envs (best-effort)
                        frames = [base_frame for _ in range(reward.shape[0])]
                    self._embed_attempts += len(frames)
                    embeddings = self._semantic_manager.maybe_compute_embeddings(frames)
                    # Always derive per-env costs & dones for risk buffer regardless of simple_mode
                    mean_costs = cost.detach().cpu().tolist()
                    try:
                        dones_list = (terminated | truncated).detach().cpu().tolist()
                    except Exception:  # noqa: BLE001
                        dones_list = [False] * len(mean_costs)
                    if getattr(self._semantic_manager.cfg, 'simple_mode', False):
                        valid_ct = sum(1 for e in embeddings if e is not None)
                        shapings = []
                        success_ct = 0
                        if self._semantic_manager.cfg.shaping_enable:
                            for emb in embeddings:
                                if emb is not None:
                                    success_ct += 1
                                    shaping_val, _ = self._semantic_manager.compute_shaping_and_eff(emb)
                                    shapings.append(shaping_val)
                                else:
                                    shapings.append(0.0)
                        else:
                            for emb in embeddings:
                                if emb is not None:
                                    success_ct += 1
                                shapings.append(0.0)
                        self._embed_success += success_ct
                        shaping_tensor = torch.as_tensor(shapings, device=reward.device, dtype=reward.dtype)
                        reward = reward + shaping_tensor
                        # logging (aggregate shaping mean, ratio, std)
                        mean_shaping = shaping_tensor.mean()
                        logger.store({'Semantics/Shaping': mean_shaping})
                        try:
                            raw_reward_mean = base_reward.mean() if hasattr(base_reward, 'mean') else base_reward
                            if abs(float(raw_reward_mean)) > 1e-6:
                                ratio = (mean_shaping / raw_reward_mean).clamp(-10, 10)
                                logger.store({'Semantics/ShapingRewardRatio': ratio})
                        except Exception:
                            pass
                        try:
                            logger.store({'Semantics/ShapingStd': shaping_tensor.std(unbiased=False)})
                        except Exception:
                            logger.store({'Semantics/ShapingStd': torch.as_tensor(0.0)})
                        latency = self._semantic_manager.last_embed_latency_ms if success_ct > 0 else 0.0
                        logger.store({'Semantics/EmbedLatencyMs': latency})
                        logger.store({
                            'Semantics/Debug/EmbedAttempts': float(self._embed_attempts),
                            'Semantics/Debug/EmbedSuccess': float(self._embed_success),
                        })
                        # Telemetry diagnostics (last embedding processed)
                        logger.store(self._semantic_manager.debug_metrics())
                    self._semantic_manager.record_multi_step(embeddings, mean_costs, dones_list)
                    # removed verbose debug print
                else:
                    # Legacy single-frame path
                    embedding = None
                    shaping = 0.0
                    try:
                        frame = self._env.render()
                        self._embed_attempts += 1
                        embedding = self._semantic_manager.maybe_compute_embedding(frame)
                        if embedding is not None:
                            self._embed_success += 1
                            if self._semantic_manager.cfg.shaping_enable:
                                shaping, _ = self._semantic_manager.compute_shaping_and_eff(embedding)
                        logger.store({
                            'Semantics/Debug/EmbedAttempts': float(self._embed_attempts),
                            'Semantics/Debug/EmbedSuccess': float(self._embed_success),
                        })
                        if embedding is not None:
                            logger.store(self._semantic_manager.debug_metrics())
                    except Exception:  # noqa: BLE001
                        pass
                    if shaping != 0.0:
                        reward = reward + shaping
                        # ratio & std (scalar path: std=0)
                        try:
                            base = reward - shaping
                            if abs(base.item()) > 1e-6:
                                logger.store({'Semantics/ShapingRewardRatio': torch.as_tensor(shaping / base.item()).clamp(-10, 10)})
                        except Exception:
                            pass
                        logger.store({'Semantics/ShapingStd': torch.as_tensor(0.0)})
                    # Single-path done if ANY env done this step (approx) for legacy; choose mean done
                    # Per-env dones list used to count episodes precisely (even in single-frame mode)
                    try:
                        dones_list = (terminated | truncated).detach().cpu().tolist()
                        any_done = any(dones_list)
                    except Exception:  # noqa: BLE001
                        dones_list = None
                        any_done = False
                    self._semantic_manager.record_step(embedding, float(cost.mean().item()), any_done, env_idx=0)
                    latency = self._semantic_manager.last_embed_latency_ms if embedding is not None else 0.0
                    logger.store({'Semantics/EmbedLatencyMs': latency})
                    logger.store({'Semantics/Shaping': torch.as_tensor(shaping)})
            else:
                # No capture this step: still advance schedule counter for semantic annealing
                if self._semantic_manager.cfg.enable:
                    self._semantic_manager.advance_step()
                    logger.store({'Semantics/Shaping': torch.as_tensor(0.0)})
                    logger.store({'Semantics/EmbedLatencyMs': 0.0})
            if self._semantic_manager.cfg.enable:
                try:
                    raw_val = base_reward.mean() if hasattr(base_reward, 'mean') else torch.as_tensor(base_reward)
                    logger.store({'Semantics/RawReward': raw_val})
                except Exception:  # pragma: no cover
                    pass
                # Always log semantic debug metrics (includes Risk/Buf/*) each step to avoid NaNs when no capture.
                try:
                    logger.store(self._semantic_manager.debug_metrics())
                except Exception:  # noqa: BLE001
                    pass
                # Live logging of modulation episode counter & active gate status
                try:
                    eps_done = float(self._semantic_manager.episodes_completed)
                    min_eps = float(getattr(self._semantic_manager.cfg, 'modulation_min_episodes', 0))
                    active = 1.0 if eps_done >= min_eps and getattr(self._semantic_manager.cfg, 'modulation_enable', False) else 0.0
                    logger.store({'Risk/ModulationActive': active})
                except Exception:  # noqa: BLE001
                    pass

            self._log_value(reward=reward, cost=cost, info=info)
            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )
            obs = next_obs

            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(info['final_observation'][idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)
                    buffer.finish_path(last_value_r, last_value_c, idx)
            # Centralized explicit episode counting (once per env per true episode end)
            if self._semantic_manager.cfg.enable:
                try:
                    done_mask = (terminated | truncated).detach().cpu()
                    ended = int(done_mask.int().sum().item())
                    if ended > 0:
                        env_indices = [i for i, flag in enumerate(done_mask.tolist()) if flag]
                        self._semantic_manager.mark_episode_end(int(ended), env_indices=env_indices)
                        eps_done = float(self._semantic_manager.episodes_completed)
                        min_eps = float(getattr(self._semantic_manager.cfg, 'modulation_min_episodes', 0))
                        active = 1.0 if eps_done >= min_eps and getattr(self._semantic_manager.cfg, 'modulation_enable', False) else 0.0
                        logger.store({'Risk/ModulationActive': active})
                except Exception:  # noqa: BLE001
                    pass

        if self._semantic_manager.cfg.enable:
            rate = (
                float(self._embed_success) / float(self._embed_attempts)
                if self._embed_attempts > 0
                else 0.0
            )
            logger.store({'Semantics/EmbedSuccessRate': rate})
            # final debug counters
            logger.store({
                'Semantics/Debug/EmbedAttempts': float(self._embed_attempts),
                'Semantics/Debug/EmbedSuccess': float(self._embed_success),
            })
