"""SemanticOnPolicyAdapter extends OnPolicyAdapter with frame capture & shaping."""
from __future__ import annotations

import torch

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.logger import Logger
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.common.semantics.semantic_manager import SemanticManager


class SemanticOnPolicyAdapter(OnPolicyAdapter):  # pragma: no cover minimal logic
    def __init__(self, *args, semantic_manager: SemanticManager, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._semantic_manager = semantic_manager
        super().__init__(*args, **kwargs)

    def rollout(self, steps_per_epoch: int, agent: ConstraintActorCritic, buffer: VectorOnPolicyBuffer, logger: Logger) -> None:  # type: ignore[override]
        self._reset_log()
        obs, _ = self.reset()
        capture_interval = self._semantic_manager.cfg.capture_interval
        for step in range(steps_per_epoch):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # frame capture & embedding every interval
            embedding = None
            shaping = 0.0
            if self._semantic_manager.cfg.enable and (step % capture_interval == 0):
                try:
                    frame = self._env.render()  # assuming safety-gymnasium returns RGB array
                    embedding = self._semantic_manager.maybe_compute_embedding(frame)
                    shaping = self._semantic_manager.shaping_term(embedding) if embedding is not None else 0.0
                except Exception:  # noqa: BLE001
                    pass
            if shaping != 0.0:
                reward = reward + shaping
                logger.store({'Semantics/Shaping': torch.as_tensor(shaping)})
            if embedding is not None:
                logger.store({'Semantics/EmbedLatencyMs': self._semantic_manager.last_embed_latency_ms})
            self._semantic_manager.record_step(embedding, float(cost.mean().item()))

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
