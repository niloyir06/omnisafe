# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class PolicyGradient(BaseAlgo):
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `PG <https://proceedings.neurips.cc/paper/1999/file64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        # Semantic adapter swap (inactive by default) - algorithms can override _env after calling super().
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        # Original explicit registration style preserved; only additional semantic/risk keys appended.
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {'pi': self._actor_critic.actor}
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        # Core metrics
        wl = self._cfgs.logger_cfgs.window_lens
        self._logger.register_key('Metrics/EpRet', window_length=wl)
        self._logger.register_key('Metrics/EpCost', window_length=wl)
        self._logger.register_key('Metrics/EpLen', window_length=wl)

        # Training stats
        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')
        self._logger.register_key('TotalEnvSteps')

        # Loss/value metrics
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        # Timing
        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        # Semantic & risk diagnostics
        self._logger.register_key('Semantics/Shaping', min_and_max=True)
        self._logger.register_key('Semantics/RawReward')
        self._logger.register_key('Semantics/EmbedLatencyMs', min_and_max=True)
        self._logger.register_key('Semantics/EmbedSuccessRate')
        self._logger.register_key('Semantics/Debug/EmbedAttempts')
        self._logger.register_key('Semantics/Debug/EmbedSuccess')
        self._logger.register_key('Risk/Loss')
        self._logger.register_key('Risk/PredMean')
        self._logger.register_key('Risk/TargetMean')
        self._logger.register_key('Risk/Corr')
        self._logger.register_key('Risk/LR')
        self._logger.register_key('Risk/ModulationScale')
        # Additional modulation gating telemetry (episode-count gate & active flag)
        self._logger.register_key('Risk/ModulationActive')
        # Risk buffer diagnostics (semantic extension)
        self._logger.register_key('Risk/Buf/Embeddings')
        self._logger.register_key('Risk/Buf/Costs')
        self._logger.register_key('Risk/Buf/Dones')
        # Risk training status diagnostics (added after initial set; keep short window)
        self._logger.register_key('Risk/TrainStatusCode', window_length=100)
        self._logger.register_key('Risk/TrainStatusMsgLen', window_length=100)
        self._logger.register_key('Risk/TrainSamples')
        # NaN / Inf event counter for risk head (sanity diagnostics)
        self._logger.register_key('Risk/NanEvents')

        # Semantic telemetry (raw margin diagnostics & schedule)
        win = wl
        self._logger.register_key('Semantics/RawMargin', min_and_max=True, window_length=win)
        self._logger.register_key('Semantics/NormMargin', min_and_max=True, window_length=win)
        self._logger.register_key('Semantics/Beta', window_length=win)
        self._logger.register_key('Semantics/ClampFrac', window_length=win)
        self._logger.register_key('Semantics/CaptureCount', window_length=win)
        self._logger.register_key('Semantics/CaptureIntervalEffective', window_length=win)
        self._logger.register_key('Semantics/ShapingRewardRatio', window_length=win)
        self._logger.register_key('Semantics/ShapingStd', window_length=win)

        # Environment-specific keys
        for env_spec_key in self._env.env_spec_keys:
            self._logger.register_key(env_spec_key)

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )
        # Semantic risk head update hook (only if semantic manager attached by subclass/override)
        if hasattr(self, '_semantic_manager') and hasattr(self._cfgs, 'semantic_cfgs') and getattr(self, '_semantic_manager') is not None:  # type: ignore[attr-defined]
            sem_cfg = self._cfgs.semantic_cfgs  # type: ignore[attr-defined]
            # removed verbose entry debug
            if getattr(sem_cfg, 'risk_enable', False):
                if not hasattr(self, '_risk_head'):
                    from omnisafe.common.semantics.risk_head import SemanticRiskHead
                    example_emb = self._semantic_manager.get_recent_risk_batch()  # type: ignore[attr-defined]
                    in_dim = example_emb.shape[1] if example_emb is not None else 512
                    risk_device = getattr(self._semantic_manager, 'model_device', self._device)  # type: ignore[attr-defined]
                    # Keep risk head in float32 for numerical stability even if encoder is fp16/bf16.
                    self._risk_head = SemanticRiskHead(in_dim).to(risk_device).float()
                    risk_lr = getattr(sem_cfg, 'risk_lr', 1e-3)
                    self._risk_opt = torch.optim.Adam(self._risk_head.parameters(), lr=risk_lr)
                batch_embs = self._semantic_manager.get_recent_risk_batch()  # type: ignore[attr-defined]
                train_status_code = None  # 0=trained, 1=no_embeddings, 2=too_few_embeddings, 3=targets_none
                train_status_msg = ''
                samples_used = 0
                nan_events = 0
                min_samples = int(getattr(sem_cfg, 'risk_min_samples', 5))
                if batch_embs is not None and batch_embs.shape[0] >= min_samples:
                    risk_device = next(self._risk_head.parameters()).device
                    full_embs = batch_embs.to(risk_device, dtype=torch.float32)
                    # Sanitize embeddings (replace NaN/Inf with zeros)
                    if not torch.isfinite(full_embs).all():
                        full_embs = torch.nan_to_num(full_embs, nan=0.0, posinf=0.0, neginf=0.0)
                        nan_events += 1
                    horizon = getattr(sem_cfg, 'risk_horizon', 64)
                    gamma = getattr(sem_cfg, 'discount', 0.99)
                    episode_mask_enable = getattr(sem_cfg, 'risk_episode_mask_enable', True)
                    update_iters = max(1, int(getattr(sem_cfg, 'risk_update_iters', 1)))
                    batch_size = int(getattr(sem_cfg, 'risk_batch_size', 0))
                    total_loss = 0.0
                    last_preds = None
                    last_targets_mean = 0.0
                    corr_val = None
                    for it in range(update_iters):
                        if batch_size > 0:
                            # Random sample without replacement if possible, else with replacement
                            if full_embs.shape[0] <= batch_size:
                                embs = full_embs
                            else:
                                idx = torch.randperm(full_embs.shape[0], device=full_embs.device)[:batch_size]
                                embs = full_embs[idx]
                        else:
                            embs = full_embs
                        # Build targets
                        with torch.no_grad():
                            if episode_mask_enable:
                                targets_full = self._semantic_manager.build_episode_masked_targets(gamma=gamma, horizon=horizon)  # type: ignore[attr-defined]
                                if targets_full is not None:
                                    L = min(len(targets_full), embs.shape[0])
                                    targets = targets_full[-L:].to(risk_device)
                                    embs = embs[-L:]
                                else:
                                    targets = None
                            else:
                                costs = torch.tensor(list(self._semantic_manager.costs), dtype=torch.float32, device=risk_device)  # type: ignore[attr-defined]
                                if costs.numel() > 0:
                                    L = min(len(costs), embs.shape[0])
                                    costs = costs[-L:]
                                    embs = embs[-L:]
                                    disc = torch.tensor([gamma ** k for k in range(horizon)], device=risk_device)
                                    targets = torch.zeros(L, dtype=torch.float32, device=risk_device)
                                    for i in range(L):
                                        seg = costs[i:i + horizon]
                                        targets[i] = (seg * disc[: seg.shape[0]]).sum()
                                else:
                                    targets = None
                        if targets is None:
                            train_status_code = 3
                            train_status_msg = 'targets_none'
                            break
                        # Sanitize targets
                        if not torch.isfinite(targets).all():
                            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                            nan_events += 1
                        preds = self._risk_head(embs.float())
                        if not torch.isfinite(preds).all():
                            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                            nan_events += 1
                        loss_t = torch.nn.functional.smooth_l1_loss(preds.float(), targets.float())
                        if not torch.isfinite(loss_t):
                            # Skip this iteration; count event and continue.
                            nan_events += 1
                            continue
                        self._risk_opt.zero_grad()
                        loss_t.backward()
                        self._risk_opt.step()
                        total_loss += loss_t.item()
                        last_preds = preds.detach()
                        last_targets_mean = targets.mean().item()
                        samples_used += embs.shape[0]
                        # Correlation (use last iteration batch)
                        if preds.numel() > 10 and targets.numel() > 10 and preds.std() > 1e-6 and targets.std() > 1e-6:
                            try:
                                corr_mat = torch.corrcoef(torch.stack([preds.flatten().detach(), targets.flatten().detach()]))
                                corr_val = corr_mat[0, 1].item()
                            except Exception:  # pragma: no cover
                                pass
                    if train_status_code != 3:
                        avg_loss = total_loss / update_iters
                        train_status_code = 0
                        train_status_msg = 'trained'
                        log_dict = {
                            'Risk/Loss': avg_loss,
                            'Risk/PredMean': 0.0 if last_preds is None else last_preds.mean().item(),
                            'Risk/TargetMean': last_targets_mean,
                            'Risk/TrainSamples': float(samples_used),
                        }
                        try:
                            if getattr(sem_cfg, 'modulation_enable', False):
                                scale = self._semantic_manager.modulation_scale(self._risk_head)  # type: ignore[attr-defined]
                                if scale is not None:
                                    log_dict['Risk/ModulationScale'] = scale
                        except Exception:  # noqa: BLE001
                            pass
                        if hasattr(self, '_risk_opt'):
                            for pg in self._risk_opt.param_groups:
                                if 'lr' in pg:
                                    log_dict['Risk/LR'] = pg['lr']
                                    break
                        if corr_val is not None:
                            log_dict['Risk/Corr'] = corr_val
                        log_dict['Risk/TrainStatusCode'] = float(train_status_code)
                        log_dict['Risk/TrainStatusMsgLen'] = float(len(train_status_msg))
                        if nan_events > 0:
                            log_dict['Risk/NanEvents'] = float(nan_events)
                        self._logger.store(log_dict)
                else:
                    if batch_embs is None:
                        train_status_code = 1
                        train_status_msg = 'no_embeddings'
                        emb_count = 0
                    else:
                        emb_count = batch_embs.shape[0]
                        train_status_code = 2
                        train_status_msg = f'too_few_embeddings:{emb_count}'
                    self._logger.store({
                        'Risk/Loss': 0.0,
                        'Risk/PredMean': 0.0,
                        'Risk/TargetMean': 0.0,
                        'Risk/TrainStatusCode': float(train_status_code),
                        'Risk/TrainStatusMsgLen': float(len(train_status_msg)),
                        'Risk/Buf/Embeddings': float(0 if batch_embs is None else batch_embs.shape[0]),
                        'Risk/TrainSamples': float(0),
                        'Risk/NanEvents': float(0),
                    })

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} [
                \sum_{t=0}^T ( \frac{\pi^{'}_{\theta}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_{\theta}` is the policy network, :math:`\pi^{'}_{\theta}`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss
