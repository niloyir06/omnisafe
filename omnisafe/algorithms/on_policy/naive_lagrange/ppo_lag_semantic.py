"""Semantic-enabled PPOLag variant.

This subclass conditionally swaps the environment adapter for a semantic-aware adapter
when ``semantic_cfgs.enable`` is True. Core PPOLag logic remains unchanged; semantic
features (reward shaping, risk head) are activated via existing hooks added to
``PolicyGradient``.
"""
from __future__ import annotations

from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.algorithms import registry


@registry.register
class PPOLagSem(PPOLag):  # pragma: no cover simple wiring
    """PPOLag with optional semantic guidance (CLIP embedding based).

    Activate by setting in custom cfgs (CLI examples):
        --semantic-cfgs:enable True
        --semantic-cfgs:shaping_enable True
        --semantic-cfgs:risk_enable True
        --semantic-cfgs:modulation_enable True
    """

    def _init_env(self) -> None:  # type: ignore[override]
        super()._init_env()
        # After base env init we can optionally replace adapter.
        sem_cfg = getattr(self._cfgs, 'semantic_cfgs', None)
        if sem_cfg and getattr(sem_cfg, 'enable', False):
            from omnisafe.common.semantics.semantic_manager import SemanticManager, SemanticConfig
            from omnisafe.adapter.semantic_onpolicy_adapter import SemanticOnPolicyAdapter

            sem_conf = SemanticConfig(
                enable=getattr(sem_cfg, 'enable', False),
                capture_interval=getattr(sem_cfg, 'capture_interval', 4),
                frame_size=getattr(sem_cfg, 'frame_size', 224),
                model_name=getattr(sem_cfg, 'model_name', 'openai/clip-vit-base-patch16'),
                device=self._cfgs.train_cfgs.device,
                beta_start=getattr(sem_cfg, 'beta_start', 0.05),
                beta_end_step_fraction=getattr(sem_cfg, 'beta_end_step_fraction', 0.4),
                shaping_enable=getattr(sem_cfg, 'shaping_enable', False),
                risk_enable=getattr(sem_cfg, 'risk_enable', False),
                modulation_enable=getattr(sem_cfg, 'modulation_enable', False),
                risk_horizon=getattr(sem_cfg, 'risk_horizon', 64),
                discount=getattr(sem_cfg, 'discount', 0.99),
                alpha_modulation=getattr(sem_cfg, 'alpha_modulation', 2.0),
                threshold_percentile=getattr(sem_cfg, 'threshold_percentile', 60),
                slope=getattr(sem_cfg, 'slope', 5.0),
                window_size=getattr(sem_cfg, 'window_size', 2048),
                safe_prompts=getattr(sem_cfg, 'safe_prompts', []),
                unsafe_prompts=getattr(sem_cfg, 'unsafe_prompts', []),
            )
            self._semantic_manager = SemanticManager(sem_conf, self._cfgs.train_cfgs.total_steps)
            # replace env adapter keeping same env id & num envs
            self._env = SemanticOnPolicyAdapter(  # type: ignore[attr-defined]
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
                semantic_manager=self._semantic_manager,
            )
