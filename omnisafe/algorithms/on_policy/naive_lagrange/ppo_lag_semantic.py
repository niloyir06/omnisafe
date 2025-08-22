"""Semantic-enabled PPOLag variant.

This subclass conditionally swaps the environment adapter for a semantic-aware adapter
when ``semantic_cfgs.enable`` is True. Core PPOLag logic remains unchanged; semantic
features (reward shaping, risk head) are activated via existing hooks added to
``PolicyGradient``.
"""
from __future__ import annotations

from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
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

            # Allow CLIP to live on its own device (e.g., CUDA) while policy stays on CPU or another device.
            sem_conf = SemanticConfig(
                enable=getattr(sem_cfg, 'enable', False),
                capture_interval=getattr(sem_cfg, 'capture_interval', 4),
                frame_size=getattr(sem_cfg, 'frame_size', 224),
                model_name=getattr(sem_cfg, 'model_name', 'openai/clip-vit-base-patch16'),
                model_device=getattr(sem_cfg, 'device', getattr(sem_cfg, 'model_device', self._cfgs.train_cfgs.device)),
                host_device=getattr(sem_cfg, 'host_device', 'cpu'),
                beta_start=getattr(sem_cfg, 'beta_start', 0.05),
                beta_end_step_fraction=getattr(sem_cfg, 'beta_end_step_fraction', 0.4),
                shaping_enable=getattr(sem_cfg, 'shaping_enable', False),
                risk_enable=getattr(sem_cfg, 'risk_enable', False),
                modulation_enable=getattr(sem_cfg, 'modulation_enable', False),
                modulation_min_episodes=getattr(sem_cfg, 'modulation_min_episodes', 0),
                risk_horizon=getattr(sem_cfg, 'risk_horizon', 64),
                discount=getattr(sem_cfg, 'discount', 0.99),
                risk_episode_mask_enable=getattr(sem_cfg, 'risk_episode_mask_enable', True),
                risk_min_samples=getattr(sem_cfg, 'risk_min_samples', 5),
                risk_batch_size=getattr(sem_cfg, 'risk_batch_size', 0),
                risk_update_iters=getattr(sem_cfg, 'risk_update_iters', 1),
                alpha_modulation=getattr(sem_cfg, 'alpha_modulation', 2.0),
                threshold_percentile=getattr(sem_cfg, 'threshold_percentile', 60),
                slope=getattr(sem_cfg, 'slope', 5.0),
                window_size=getattr(sem_cfg, 'window_size', 2048),
                norm_window=getattr(sem_cfg, 'norm_window', 1000),
                margin_norm_enable=getattr(sem_cfg, 'margin_norm_enable', True),
                margin_scale=getattr(sem_cfg, 'margin_scale', 1.0),
                potential_enable=getattr(sem_cfg, 'potential_enable', False),
                safe_prompts=getattr(sem_cfg, 'safe_prompts', []),
                unsafe_prompts=getattr(sem_cfg, 'unsafe_prompts', []),
                batch_across_envs=getattr(sem_cfg, 'batch_across_envs', True),
                batch_max=getattr(sem_cfg, 'batch_max', 32),
                oom_backoff=getattr(sem_cfg, 'oom_backoff', True),
            )
            # Debug: print semantic configuration once at init so user can verify values.
            try:
                from dataclasses import asdict  # local import (std lib)
                print('[SemanticConfig Init]', asdict(sem_conf))
            except Exception:  # noqa: BLE001
                # Fallback to repr if asdict fails for any reason
                print('[SemanticConfig Init]', sem_conf)

            self._semantic_manager = SemanticManager(sem_conf, self._cfgs.train_cfgs.total_steps)
            # replace env adapter keeping same env id & num envs
            self._env = SemanticOnPolicyAdapter(  # type: ignore[attr-defined]
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
                semantic_manager=self._semantic_manager,
            )

    def _update(self) -> None:  # type: ignore[override]
        """Extend PPOLag update with optional Lagrange modulation using risk predictions.

        If semantic modulation enabled and risk head exists with sufficient samples, scale the
        effective lambda optimizer step size for this update only by multiplying gradients.
        """
        # Standard cost statistic
        import numpy as np  # local import to avoid top-level changes
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # Determine modulation scale
        scale = 1.0
        sem_cfg = getattr(self._cfgs, 'semantic_cfgs', None)
        if sem_cfg and getattr(sem_cfg, 'modulation_enable', False) and hasattr(self, '_semantic_manager'):
            try:
                if hasattr(self, '_risk_head'):
                    mscale = self._semantic_manager.modulation_scale(self._risk_head)  # type: ignore[attr-defined]
                    if mscale is not None:
                        scale = float(mscale)
            except Exception:  # noqa: BLE001
                pass
        # Apply scaled update: temporarily adjust optimizer lr
        base_lr = self._lagrange.lambda_optimizer.param_groups[0]['lr']  # type: ignore[attr-defined]
        if scale != 1.0:
            self._lagrange.lambda_optimizer.param_groups[0]['lr'] = base_lr * scale  # type: ignore[attr-defined]
        self._lagrange.update_lagrange_multiplier(Jc)  # type: ignore[attr-defined]
        if scale != 1.0:
            # restore base lr
            self._lagrange.lambda_optimizer.param_groups[0]['lr'] = base_lr  # type: ignore[attr-defined]
        # Proceed with normal actor/critic + risk head update via superclass chain (PolicyGradient)
        # Invoke base PolicyGradient update (actor/critic + risk head hook) directly
        PolicyGradient._update(self)  # type: ignore[misc]
        # Log lambda & modulation (always log scale for contiguous series)
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})  # type: ignore[attr-defined]
        self._logger.store({'Risk/ModulationScale': scale})
        if hasattr(self, '_semantic_manager'):
            # Episodes completed & whether gating active (1 if modulation allowed, else 0)
            eps_done = getattr(self._semantic_manager, 'episodes_completed', 0)
            min_eps = getattr(getattr(self._cfgs, 'semantic_cfgs', object()), 'modulation_min_episodes', 0)
            active = 1.0 if eps_done >= min_eps else 0.0
            self._logger.store({'Risk/ModulationActive': active})
