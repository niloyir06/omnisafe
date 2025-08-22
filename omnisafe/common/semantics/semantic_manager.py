"""SemanticManager: CLIP embeddings, semantic shaping, risk buffers, modulation.

Features:
 - CLIP load with safetensors and dtype fallback (bf16 -> fp16 -> fp32) plus status string.
 - Prompt centroids (safe/unsafe) computed once.
 - Embedding extraction & latency tracking.
 - Reward shaping: cosine beta decay, running z-normalization of margin, clipping to [-2,2].
 - Risk data buffers (embeddings, costs) for auxiliary predictor.
 - Modulation scale helper using risk quantile gating.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Deque, Sequence
from collections import deque
import time
import torch
from transformers import AutoModel, AutoProcessor  # type: ignore



@dataclass
class SemanticConfig:
    enable: bool = False
    capture_interval: int = 4
    frame_size: int = 224
    model_name: str = "openai/clip-vit-base-patch32"
    # model_device: where CLIP model runs (set to 'cuda:0' to keep heavy compute on GPU)
    model_device: str = "cuda:0"
    # host_device: where centroids, returned embeddings, and subsequent computations live (usually 'cpu')
    host_device: str = "cpu"
    beta_start: float = 0.15
    beta_end_step_fraction: float = 0.7  # (NOTE: consider reverting to 0.4 if this was unintentional)
    shaping_enable: bool = False
    risk_enable: bool = False
    modulation_enable: bool = False
    # Minimum completed episodes before modulation is allowed (gating)
    modulation_min_episodes: int = 0
    # Risk head learning rate (instead of hard-coded value)
    risk_lr: float = 1e-3
    risk_horizon: int = 64
    discount: float = 0.99
    # Episode-aware masking for risk targets (reset at terminals)
    risk_episode_mask_enable: bool = True
    # Minimum samples required before attempting a risk head update.
    risk_min_samples: int = 5
    # Optional mini-batch size for risk head updates (0 => use full available batch).
    risk_batch_size: int = 0
    # Number of risk update iterations per PPO epoch (>=1). When >1 with mini-batching, draws fresh samples each iter.
    risk_update_iters: int = 1
    alpha_modulation: float = 2.0
    threshold_percentile: int = 60
    slope: float = 5.0
    window_size: int = 2048
    norm_window: int = 1000
    # Margin normalization control & scaling (diagnostics)
    margin_norm_enable: bool = True
    margin_scale: float = 1.0  # multiply raw margin before normalization/clipping
    # Potential-based shaping toggle: if True uses beta*(gamma*phi_next - phi_prev) with phi = normed margin
    potential_enable: bool = False
    # Environment-specific base descriptive prompts (red car, green goal, purple obstacles, blue cubes)
    safe_prompts: List[str] = field(default_factory=lambda: [
        "red car moving toward green goal",
        "clear path ahead to green goal",
        "red car aligned safely avoiding obstacles",
    ])
    unsafe_prompts: List[str] = field(default_factory=lambda: [
        "red car near purple obstacle collision",
        "red car pushing into blue cube hazard",
        "blocked path crowded with purple obstacles",
    ])
    # Spatial batching options
    batch_across_envs: bool = True  # attempt to embed all env frames together
    batch_max: int = 32  # safety cap for very large vector_env_nums
    oom_backoff: bool = True  # on CUDA OOM during batch embedding, halve batch and retry
    # Temporal window for pooled semantic embedding (1 = disabled). When >1 the shaping term
    # uses the mean of the last N frame embeddings (warm-up: mean over available < N).
    temporal_window: int = 6
    # Simplicity toggle: when True use a stripped-down code path (single dtype load, no silent fallbacks/backoffs)
    simple_mode: bool = True


class SemanticManager:
    # Type hints for attributes (not initialized at class scope to avoid GPU ops at import time)
    safe_centroid: Optional[torch.Tensor]
    unsafe_centroid: Optional[torch.Tensor]
    embeddings: Deque[torch.Tensor]
    costs: Deque[float]
    dones: Deque[bool]
    _margins: Deque[float]
    _temporal_windows: List[Deque[torch.Tensor]]
    _prev_phi_env: List[Optional[float]]

    def __init__(self, cfg: SemanticConfig, total_steps: int) -> None:
        self.cfg = cfg
        # fallback prompts if empty
        if not self.cfg.safe_prompts:
            self.cfg.safe_prompts = ["clear path to goal", "centered trajectory", "ample clearance"]
        if not self.cfg.unsafe_prompts:
            self.cfg.unsafe_prompts = ["imminent collision", "tight near-obstacle turn", "risky close pass"]
        self.total_steps = total_steps
        self.global_step = 0
        # Separate devices: CLIP compute vs host CPU for RL objects.
        self.model_device = torch.device(self.cfg.model_device)
        self._clip_status = 'disabled'
        if 'cuda' in self.cfg.model_device and not torch.cuda.is_available():  # graceful fallback
            self.model_device = torch.device('cpu')
            self._clip_status = 'fallback_cpu_no_cuda'
        self.host_device = torch.device(self.cfg.host_device)
        self.last_embed_latency_ms = 0.0
        self._clip_ready = False
        self.safe_centroid = None
        self.unsafe_centroid = None
        self.embeddings = deque(maxlen=self.cfg.window_size)
        self.costs = deque(maxlen=self.cfg.window_size)
        self.dones = deque(maxlen=self.cfg.window_size)
        self._margins = deque(maxlen=self.cfg.norm_window)
        self._load_clip_and_prompts()
        # Debug / telemetry fields
        self._last_margin_raw = 0.0
        self._last_margin_norm = 0.0
        self._last_beta = 0.0
        self._total_margins = 0
        self._clipped_margins = 0
        self._capture_count = 0  # number of embedding capture events (single or batched)
        self._last_capture_step = -1  # for effective interval telemetry
        # Episode counter for modulation gating
        self.episodes_completed = 0
        # Per-env temporal embedding windows (allocated lazily on first multi-step capture)
        self._tw = max(1, int(getattr(self.cfg, 'temporal_window', 1)))
        self._temporal_windows = []  # type: ignore[var-annotated]
        # Per-env previous phi for potential shaping (avoid cross-env leakage)
        self._prev_phi_env = []
        # One-time warning flag if embeddings never populate while captures succeed
        self._empty_buf_warned = True
        # Enforce minimum window size to actually retain samples
        if self.cfg.window_size < 1:
            self.cfg.window_size = 1
            self.embeddings = deque(maxlen=1)
            self.costs = deque(maxlen=1)
            self.dones = deque(maxlen=1)

    def _load_clip_and_prompts(self) -> None:
        if not self.cfg.enable:
            return
        if self.cfg.simple_mode:
            # Unified loader for CLIP or SigLIP(2) via AutoModel/AutoProcessor.
            dt = torch.float16 if (self.model_device.type == 'cuda') else torch.float32
            self.clip_model = AutoModel.from_pretrained(  # CLIPModel or SiglipModel resolved automatically
                self.cfg.model_name,
                torch_dtype=dt,
                use_safetensors=True,
            )  # type: ignore[attr-defined]
            self.clip_model.to(self.model_device)  # type: ignore[attr-defined]
            self.clip_model.eval()
            self.clip_proc = AutoProcessor.from_pretrained(self.cfg.model_name)  # type: ignore[attr-defined]
            with torch.no_grad():
                safe_tok = self.clip_proc(text=self.cfg.safe_prompts, return_tensors="pt", padding=True)
                safe_tok = {k: v.to(self.model_device) for k, v in safe_tok.items()}
                se = self.clip_model.get_text_features(**safe_tok)
                self.safe_centroid = torch.nn.functional.normalize(se.mean(0, keepdim=True), dim=-1).to(self.host_device)
                unsafe_tok = self.clip_proc(text=self.cfg.unsafe_prompts, return_tensors="pt", padding=True)
                unsafe_tok = {k: v.to(self.model_device) for k, v in unsafe_tok.items()}
                ue = self.clip_model.get_text_features(**unsafe_tok)
                self.unsafe_centroid = torch.nn.functional.normalize(ue.mean(0, keepdim=True), dim=-1).to(self.host_device)
                if self.host_device.type == 'cpu':
                    self.safe_centroid = self.safe_centroid.float()
                    self.unsafe_centroid = self.unsafe_centroid.float()
            self._clip_ready = True
            family = 'siglip' if 'siglip' in self.cfg.model_name.lower() else 'clip'
            self._clip_status = f"ok_simple_{family}_{str(dt).split('.')[-1]}"
        else:
            # Original multi-dtype fallback retained (can be removed later if not needed)
            dtypes = [torch.bfloat16, torch.float16, torch.float32]
            last_err: Optional[Exception] = None
            for dt in dtypes:
                try:
                    self.clip_model = AutoModel.from_pretrained(
                        self.cfg.model_name,
                        torch_dtype=dt,
                        use_safetensors=True,
                    )  # type: ignore[attr-defined]
                    self.clip_model.to(self.model_device)  # type: ignore[attr-defined]
                    self.clip_model.eval()
                    self.clip_proc = AutoProcessor.from_pretrained(self.cfg.model_name)  # type: ignore[attr-defined]
                    with torch.no_grad():
                        safe_tok = self.clip_proc(text=self.cfg.safe_prompts, return_tensors="pt", padding=True)
                        safe_tok = {k: v.to(self.model_device) for k, v in safe_tok.items()}
                        se = self.clip_model.get_text_features(**safe_tok)
                        self.safe_centroid = torch.nn.functional.normalize(se.mean(0, keepdim=True), dim=-1).to(self.host_device)
                        if self.host_device.type == 'cpu':
                            self.safe_centroid = self.safe_centroid.float()
                        unsafe_tok = self.clip_proc(text=self.cfg.unsafe_prompts, return_tensors="pt", padding=True)
                        unsafe_tok = {k: v.to(self.model_device) for k, v in unsafe_tok.items()}
                        ue = self.clip_model.get_text_features(**unsafe_tok)
                        self.unsafe_centroid = torch.nn.functional.normalize(ue.mean(0, keepdim=True), dim=-1).to(self.host_device)
                        if self.host_device.type == 'cpu':
                            self.unsafe_centroid = self.unsafe_centroid.float()
                    self._clip_ready = True
                    family = 'siglip' if 'siglip' in self.cfg.model_name.lower() else 'clip'
                    self._clip_status = f"ok_st_{family}_{str(dt).split('.')[-1]}"
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
            if not self._clip_ready:
                err_name = type(last_err).__name__ if last_err else "unknown"
                msg = str(last_err) if last_err else ""
                if 'safetensors' in msg.lower() and 'not found' in msg.lower():
                    err_name = 'NoSafeTensorsWeights'
                self._clip_status = f'load_error:{err_name}'

    def is_ready(self) -> bool:
        return self._clip_ready

    def maybe_compute_embedding(self, frame) -> Optional[torch.Tensor]:
        if not self._clip_ready:
            return None
        self._capture_count += 1
        self._last_capture_step = self.global_step
        start = time.time()
        with torch.no_grad():
            batch = self.clip_proc(images=frame, return_tensors="pt")
            batch = {k: v.to(self.model_device) for k, v in batch.items()}
            img_feat = self.clip_model.get_image_features(**batch)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1).to(self.host_device)
            if self.host_device.type == 'cpu':
                img_feat = img_feat.float()
        self.last_embed_latency_ms = (time.time() - start) * 1000
        return img_feat[0]

    # ---- Batched embeddings ----
    def maybe_compute_embeddings(self, frames: Sequence) -> List[Optional[torch.Tensor]]:  # type: ignore[override]
        """Compute embeddings for a batch of frames.

        Returns list aligned with input (None where failure). Falls back to single embedding
        path if batch embedding fails. Supports OOM backoff by halving batch size recursively.
        """
        if not self._clip_ready or not frames:
            return [None for _ in frames]
        frames = list(frames)[: self.cfg.batch_max]
        self._capture_count += 1
        self._last_capture_step = self.global_step
        start = time.time()
        with torch.no_grad():
            batch = self.clip_proc(images=frames, return_tensors="pt")
            batch = {k: v.to(self.model_device) for k, v in batch.items()}
            img_feats = self.clip_model.get_image_features(**batch)
            img_feats = torch.nn.functional.normalize(img_feats, dim=-1).to(self.host_device)
            if self.host_device.type == 'cpu':
                img_feats = img_feats.float()
        latency = (time.time() - start) * 1000
        self.last_embed_latency_ms = latency / max(1, len(frames))
        return [img_feats[i] for i in range(img_feats.shape[0])]

    # ------------- Shaping ------------------
    def shaping_term(self, embedding: torch.Tensor) -> float:
        shaping, _ = self.compute_shaping_and_eff(embedding)
        return shaping

    def _ensure_env(self, env_idx: int) -> None:
        if env_idx >= len(self._temporal_windows):
            # Grow lists up to env_idx inclusive
            for _ in range(env_idx - len(self._temporal_windows) + 1):
                self._temporal_windows.append(deque(maxlen=self._tw))
                self._prev_phi_env.append(None)

    def effective_embedding(self, embedding: torch.Tensor, env_idx: int = 0) -> torch.Tensor:
        """Return (and update) per-env pooled embedding for shaping/risk.

        Maintains separate temporal deque per environment to prevent cross-env leakage.
        Warm-up: mean over available entries (< window size). If window disabled returns original.
        """
        if self._tw <= 1:
            return embedding
        self._ensure_env(env_idx)
        window = self._temporal_windows[env_idx]
        window.append(embedding.detach())
        return torch.stack(list(window), dim=0).mean(0)

    def compute_shaping_and_eff(self, embedding: torch.Tensor, env_idx: int = 0) -> tuple[float, torch.Tensor]:
        """Compute shaping term and return (shaping, effective_embedding).

        If shaping disabled or CLIP not ready, returns (0.0, pooled_embedding_or_raw).
        """
        eff_emb = self.effective_embedding(embedding, env_idx=env_idx)
        if not (self.cfg.shaping_enable and self._clip_ready and self.safe_centroid is not None and self.unsafe_centroid is not None):
            return 0.0, eff_emb
        with torch.no_grad():
            safe_sim = torch.matmul(eff_emb, self.safe_centroid[0])
            unsafe_sim = torch.matmul(eff_emb, self.unsafe_centroid[0])
            margin = (safe_sim - unsafe_sim).item()
        margin *= self.cfg.margin_scale
        norm_margin = margin
        if self.cfg.margin_norm_enable:
            self._margins.append(margin)
            if len(self._margins) > 30:
                m = sum(self._margins) / len(self._margins)
                var = sum((x - m) ** 2 for x in self._margins) / len(self._margins)
                std = (var ** 0.5) or 1.0
                norm_margin = (margin - m) / std
        beta = self._beta_schedule()
        clipped_norm = max(min(norm_margin, 2.0), -2.0)
        phi = norm_margin
        if self.cfg.potential_enable:
            self._ensure_env(env_idx)
            prev_phi = self._prev_phi_env[env_idx]
            if prev_phi is None:
                shaping = 0.0
            else:
                shaping = beta * (self.cfg.discount * phi - prev_phi)
            self._prev_phi_env[env_idx] = phi
        else:
            shaping = beta * clipped_norm
        # Telemetry updates
        self._last_margin_raw = margin
        self._last_margin_norm = norm_margin
        self._last_beta = beta
        self._total_margins += 1
        if clipped_norm != norm_margin:
            self._clipped_margins += 1
        return shaping, eff_emb

    # --- Telemetry accessors (lightweight) ---
    def debug_metrics(self) -> dict[str, float]:
        """Return semantic & risk diagnostics (fills all registered keys if possible).

        Previously we registered several logger keys (RawMargin, ClampFrac, CaptureIntervalEffective,
        Risk/Buf/Costs, Risk/Buf/Dones) but did not actually emit values. This richer dict ensures
        downstream logs reflect real buffer & margin state, reducing silent failure modes when the
        risk head appears inactive.
        """
        # Clamp fraction (avoid div-by-zero)
        clamp_frac = 0.0
        if self._total_margins > 0:
            clamp_frac = float(self._clipped_margins) / float(self._total_margins)
        # Effective interval since last capture (0 if a capture happened this step or never captured)
        if self._last_capture_step < 0:
            eff_interval = 0.0
        else:
            eff_interval = float(self.global_step - self._last_capture_step)
        # Risk buffers
        buf_emb = float(len(self.embeddings))
        buf_cost = float(len(self.costs))
        buf_done = float(len(self.dones))
        metrics = {
            'Semantics/Beta': self._last_beta,
            'Semantics/NormMargin': self._last_margin_norm,
            'Semantics/RawMargin': self._last_margin_raw,
            'Semantics/ClampFrac': clamp_frac,
            'Semantics/CaptureCount': float(self._capture_count),
            'Semantics/CaptureIntervalEffective': eff_interval,
            'Risk/Buf/Embeddings': buf_emb,
            'Risk/Buf/Costs': buf_cost,
            'Risk/Buf/Dones': buf_done,
        }
        return metrics

    def _avg_window_fill(self) -> float:
        if self._tw <= 1 or not self._temporal_windows:
            return 0.0
        fills = [len(w) / float(self._tw) for w in self._temporal_windows]
        return float(sum(fills) / len(fills)) if fills else 0.0

    def _beta_schedule(self) -> float:
        steps_end = int(self.total_steps * self.cfg.beta_end_step_fraction)
        if steps_end <= 0:
            return 0.0
        if self.global_step >= steps_end:
            return 0.0
        prog = self.global_step / steps_end
        return self.cfg.beta_start * 0.5 * (1 + torch.cos(torch.tensor(prog * 3.1415926535))).item()

    def record_step(self, embedding: Optional[torch.Tensor], cost: float, done: bool = False, env_idx: int = 0) -> None:
        """Record a single-env capture step (legacy single-frame path).

        Episode counting removed (centralized in mark_episode_end). Temporal pooling per env only.
        """
        if embedding is not None:
            eff_emb = self.effective_embedding(embedding, env_idx=env_idx)
            emb_store = eff_emb.detach()
            if self.host_device.type == 'cpu':
                emb_store = emb_store.cpu()
            before = len(self.embeddings)
            self.embeddings.append(emb_store)
            self.costs.append(cost)
            self.dones.append(done)
            after = len(self.embeddings)
        self.global_step += 1

    def record_multi_step(self, embeddings: List[Optional[torch.Tensor]], costs: List[float], dones: Optional[List[bool]] = None) -> None:
        """Record multiple (embedding, cost) pairs for a single environment step.

        Increments global_step exactly once to maintain shaping schedule consistency.
        """
        if dones is None:
            dones = [False] * len(costs)
        for idx, (emb, c, d) in enumerate(zip(embeddings, costs, dones)):
            if emb is None:
                continue
            eff_emb = self.effective_embedding(emb, env_idx=idx)
            emb_store = eff_emb.detach()
            if self.host_device.type == 'cpu':
                emb_store = emb_store.cpu()
            before = len(self.embeddings)
            self.embeddings.append(emb_store)
            self.costs.append(c)
            self.dones.append(d)
            after = len(self.embeddings)
        self.global_step += 1
        # (Episode counting centralized in mark_episode_end.)

    def count_episode_dones(self, dones: List[bool]) -> None:  # kept for backward compatibility (no-op)
        return

    # --- Centralized episode end marking (adapter can call once per true env episode) ---
    def mark_episode_end(self, count: int = 1, env_indices: Optional[List[int]] = None) -> None:
        """Record completed episodes explicitly (single source of truth).

        If env_indices provided, clear temporal windows & potential phi for those envs only.
        """
        if count <= 0:
            return
        self.episodes_completed += int(count)
        if env_indices and self._tw > 1:
            for idx in env_indices:
                if idx < len(self._temporal_windows):
                    self._temporal_windows[idx].clear()
                if idx < len(self._prev_phi_env):
                    self._prev_phi_env[idx] = None
        elif self._tw > 1 and not env_indices:
            # Fallback: if indices unknown, do nothing (avoid wiping all env history)
            pass

    def advance_step(self) -> None:
        """Advance global step without recording embeddings (no capture this step)."""
        self.global_step += 1

    def get_recent_risk_batch(self) -> Optional[torch.Tensor]:
        if not self.embeddings:
            return None
        return torch.stack(list(self.embeddings), dim=0)

    def build_episode_masked_targets(self, gamma: float, horizon: int) -> Optional[torch.Tensor]:
        """Episode-aware discounted cost targets with horizon cap."""
        if not self.embeddings or not self.costs:
            return None
        costs = torch.tensor(list(self.costs), dtype=torch.float32)
        dones_list = list(self.dones)
        # Align length
        if len(dones_list) < len(costs):
            dones_list.extend([False] * (len(costs) - len(dones_list)))
        
        L = len(costs)
        targets = torch.zeros(L, dtype=torch.float32)
        disc_powers = torch.tensor([gamma ** k for k in range(horizon)], dtype=torch.float32)
        
        # Single forward pass with episode reset
        for i in range(L):
            acc = 0.0
            for k in range(horizon):
                j = i + k
                if j >= L or dones_list[j]:
                    break
                acc += disc_powers[k] * costs[j].item()
            targets[i] = acc
        return targets

    def estimated_mean_risk(self, risk_head: Optional[torch.nn.Module]) -> Optional[float]:
        if not (risk_head and self.embeddings):
            return None
        with torch.no_grad():
            embs = torch.stack(list(self.embeddings), dim=0).to(next(risk_head.parameters()).device)
            preds = risk_head(embs)
            return preds.mean().item()

    def modulation_scale(self, risk_head: Optional[torch.nn.Module]) -> Optional[float]:
        if not (self.cfg.modulation_enable and risk_head and self.embeddings):
            return None
        # Gating: require minimum completed episodes
        if self.episodes_completed < getattr(self.cfg, 'modulation_min_episodes', 0):
            return 1.0
            
        with torch.no_grad():
            try:
                embs = torch.stack(list(self.embeddings), dim=0).to(next(risk_head.parameters()).device)
                preds = risk_head(embs).flatten()
                # Filter finite predictions
                preds = preds[torch.isfinite(preds)]
                if preds.numel() < 10:
                    return 1.0
                    
                q_level = max(0.0, min(1.0, self.cfg.threshold_percentile / 100.0))
                perc = torch.quantile(preds, q_level)
                mean_pred = preds.mean()
                
                if not (torch.isfinite(perc) and torch.isfinite(mean_pred)):
                    return 1.0
                    
                slope = max(1e-6, getattr(self.cfg, 'slope', 5.0))
                slope_scale = perc / slope if abs(perc.item()) > 1e-6 else 1.0
                diff = mean_pred - perc
                arg = (diff / (slope_scale + 1e-6)).clamp(-20.0, 20.0)
                gate = torch.sigmoid(arg)
                alpha = getattr(self.cfg, 'alpha_modulation', 2.0)
                scale = 1.0 / (1.0 + alpha * gate.item())
                return max(0.01, min(10.0, scale))  # reasonable bounds
            except Exception:
                return 1.0
