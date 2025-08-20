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
from transformers import CLIPModel, CLIPProcessor  # type: ignore



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
    # Risk head learning rate (instead of hard-coded value)
    risk_lr: float = 1e-3
    risk_horizon: int = 64
    discount: float = 0.99
    # Episode-aware masking for risk targets (reset at terminals)
    risk_episode_mask_enable: bool = True
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


class SemanticManager:
    # Type hints for attributes (not initialized at class scope to avoid GPU ops at import time)
    safe_centroid: Optional[torch.Tensor]
    unsafe_centroid: Optional[torch.Tensor]
    embeddings: Deque[torch.Tensor]
    costs: Deque[float]
    dones: Deque[bool]
    _margins: Deque[float]

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
        # Potential shaping state
        self._prev_phi = None

    def _load_clip_and_prompts(self) -> None:
        if not self.cfg.enable:
            return
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
        last_err: Optional[Exception] = None
        for dt in dtypes:
            try:
                # Enforce safetensors usage for security/performance.
                self.clip_model = CLIPModel.from_pretrained(
                    self.cfg.model_name,
                    torch_dtype=dt,
                    use_safetensors=True,
                )  # type: ignore[attr-defined]
                self.clip_model.to(self.model_device)  # type: ignore[attr-defined]
                self.clip_model.eval()
                self.clip_proc = CLIPProcessor.from_pretrained(self.cfg.model_name)  # type: ignore[attr-defined]
                with torch.no_grad():
                    safe_tok = self.clip_proc(text=self.cfg.safe_prompts, return_tensors="pt", padding=True)
                    safe_tok = {k: v.to(self.model_device) for k, v in safe_tok.items()}
                    se = self.clip_model.get_text_features(**safe_tok)
                    self.safe_centroid = torch.nn.functional.normalize(se.mean(0, keepdim=True), dim=-1).to(self.host_device)
                    if self.host_device.type == 'cpu':  # keep GPU centroids in reduced precision
                        self.safe_centroid = self.safe_centroid.float()
                    unsafe_tok = self.clip_proc(text=self.cfg.unsafe_prompts, return_tensors="pt", padding=True)
                    unsafe_tok = {k: v.to(self.model_device) for k, v in unsafe_tok.items()}
                    ue = self.clip_model.get_text_features(**unsafe_tok)
                    self.unsafe_centroid = torch.nn.functional.normalize(ue.mean(0, keepdim=True), dim=-1).to(self.host_device)
                    if self.host_device.type == 'cpu':
                        self.unsafe_centroid = self.unsafe_centroid.float()
                self._clip_ready = True
                self._clip_status = f'ok_st_{str(dt).split(".")[-1]}'
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if not self._clip_ready:
            # Distinguish safetensors absence from other errors if possible
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
        start = time.time()
        try:
            self._capture_count += 1
            self._last_capture_step = self.global_step
            with torch.no_grad():
                batch = self.clip_proc(images=frame, return_tensors="pt")
                batch = {k: v.to(self.model_device) for k, v in batch.items()}
                img_feat = self.clip_model.get_image_features(**batch)
                img_feat = torch.nn.functional.normalize(img_feat, dim=-1).to(self.host_device)
                if self.host_device.type == 'cpu':
                    img_feat = img_feat.float()
            self.last_embed_latency_ms = (time.time() - start) * 1000
            return img_feat[0]
        except Exception:  # noqa: BLE001
            return None

    # ---- Batched embeddings ----
    def maybe_compute_embeddings(self, frames: Sequence) -> List[Optional[torch.Tensor]]:  # type: ignore[override]
        """Compute embeddings for a batch of frames.

        Returns list aligned with input (None where failure). Falls back to single embedding
        path if batch embedding fails. Supports OOM backoff by halving batch size recursively.
        """
        if not self._clip_ready or not frames:
            return [None for _ in frames]
        # Clamp batch size
        frames = list(frames)[: self.cfg.batch_max]
        self._capture_count += 1
        self._last_capture_step = self.global_step

        def _embed(list_frames: List):  # recursive helper
            start = time.time()
            try:
                with torch.no_grad():
                    batch = self.clip_proc(images=list_frames, return_tensors="pt")
                    batch = {k: v.to(self.model_device) for k, v in batch.items()}
                    img_feats = self.clip_model.get_image_features(**batch)
                    img_feats = torch.nn.functional.normalize(img_feats, dim=-1).to(self.host_device)
                    if self.host_device.type == 'cpu':
                        img_feats = img_feats.float()
                latency = (time.time() - start) * 1000
                # average latency per frame for logging (approx)
                self.last_embed_latency_ms = latency / max(1, len(list_frames))
                return [img_feats[i] for i in range(img_feats.shape[0])]
            except RuntimeError as e:  # potential OOM
                if self.cfg.oom_backoff and 'out of memory' in str(e).lower() and len(list_frames) > 1:
                    mid = len(list_frames) // 2
                    left = _embed(list_frames[:mid])
                    right = _embed(list_frames[mid:])
                    return left + right
                return [None for _ in list_frames]
            except Exception:  # noqa: BLE001
                return [None for _ in list_frames]

        embeddings_list = _embed(frames)
        # Pad if recursive splitting shrunk? (Should remain aligned.)
        if len(embeddings_list) != len(frames):  # safety
            return [None for _ in frames]
        # Ensure Optional typing consistency
        typed_list: List[Optional[torch.Tensor]] = [e if isinstance(e, torch.Tensor) else None for e in embeddings_list]
        return typed_list

    # ------------- Shaping ------------------
    def shaping_term(self, embedding: torch.Tensor) -> float:
        if not (self.cfg.shaping_enable and self._clip_ready and self.safe_centroid is not None and self.unsafe_centroid is not None):
            return 0.0
        with torch.no_grad():
            safe_sim = torch.matmul(embedding, self.safe_centroid[0])
            unsafe_sim = torch.matmul(embedding, self.unsafe_centroid[0])
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
        phi = norm_margin  # un-clipped potential
        if self.cfg.potential_enable:
            if self._prev_phi is None:
                shaping = 0.0
            else:
                shaping = beta * (self.cfg.discount * phi - self._prev_phi)
            self._prev_phi = phi
        else:
            shaping = beta * clipped_norm
        # Telemetry updates
        self._last_margin_raw = margin
        self._last_margin_norm = norm_margin
        self._last_beta = beta
        self._total_margins += 1
        if clipped_norm != norm_margin:
            self._clipped_margins += 1
        return shaping

    # --- Telemetry accessors (lightweight) ---
    def debug_metrics(self) -> dict[str, float]:
        """Return latest semantic telemetry metrics.

        Consumers (adapter) can log these without recomputation.
        """
        clamp_frac = (
            float(self._clipped_margins) / float(self._total_margins)
            if self._total_margins > 0
            else 0.0
        )
        effective_interval = (
            float(self.global_step - self._last_capture_step)
            if self._last_capture_step >= 0
            else 0.0
        )
        return {
            'Semantics/RawMargin': self._last_margin_raw,
            'Semantics/NormMargin': self._last_margin_norm,
            'Semantics/Beta': self._last_beta,
            'Semantics/ClampFrac': clamp_frac,
            'Semantics/CaptureCount': float(self._capture_count),
            'Semantics/CaptureIntervalEffective': effective_interval,
        }

    def _beta_schedule(self) -> float:
        steps_end = int(self.total_steps * self.cfg.beta_end_step_fraction)
        if steps_end <= 0:
            return 0.0
        if self.global_step >= steps_end:
            return 0.0
        prog = self.global_step / steps_end
        return self.cfg.beta_start * 0.5 * (1 + torch.cos(torch.tensor(prog * 3.1415926535))).item()

    def record_step(self, embedding: Optional[torch.Tensor], cost: float, done: bool = False) -> None:
        if embedding is not None and self.cfg.risk_enable:
            # Keep embedding on host_device; if host is GPU we avoid cpu sync.
            emb_store = embedding.detach()
            if self.host_device.type == 'cpu':
                emb_store = emb_store.cpu()
            self.embeddings.append(emb_store)
            self.costs.append(cost)
            self.dones.append(done)
        self.global_step += 1

    def record_multi_step(self, embeddings: List[Optional[torch.Tensor]], costs: List[float], dones: Optional[List[bool]] = None) -> None:
        """Record multiple (embedding, cost) pairs for a single environment step.

        Increments global_step exactly once to maintain shaping schedule consistency.
        """
        if self.cfg.risk_enable:
            if dones is None:
                dones = [False] * len(costs)
            filtered_dones: List[bool] = []
            for emb, c, d in zip(embeddings, costs, dones):
                if emb is None:
                    continue
                emb_store = emb.detach()
                if self.host_device.type == 'cpu':
                    emb_store = emb_store.cpu()
                self.embeddings.append(emb_store)
                self.costs.append(c)
                filtered_dones.append(d)
            for d in filtered_dones:
                self.dones.append(d)
        self.global_step += 1

    def advance_step(self) -> None:
        """Advance global step without recording embeddings (no capture this step)."""
        self.global_step += 1

    def get_recent_risk_batch(self) -> Optional[torch.Tensor]:
        if not self.embeddings:
            return None
        return torch.stack(list(self.embeddings), dim=0)

    def build_episode_masked_targets(self, gamma: float, horizon: int) -> Optional[torch.Tensor]:
        """Episode-aware discounted cost targets with horizon cap.

        Resets accumulation after terminal states using recorded done flags.
        """
        if not self.embeddings or not self.costs:
            return None
        costs = torch.tensor(list(self.costs), dtype=torch.float32)
        dones_list = list(self.dones)
        # Align length (legacy entries may lack dones; assume False)
        if len(dones_list) < len(costs):
            pad = [False] * (len(costs) - len(dones_list))
            dones_list = pad + dones_list
        L = len(costs)
        targets = torch.zeros(L, dtype=torch.float32)
        # Backward pass resetting at terminals
        running = 0.0
        look = 0
        for i in range(L - 1, -1, -1):
            if dones_list[i]:
                running = 0.0
                look = 0
            if look >= horizon:
                running = 0.0
                look = 0
            running = costs[i].item() + gamma * running
            targets[i] = running
            look += 1
        # Forward refinement for strict horizon & terminal stop
        if horizon < L:
            pow_cache = torch.tensor([gamma ** k for k in range(horizon)], dtype=torch.float32)
            for i in range(L):
                acc = 0.0
                for k in range(horizon):
                    j = i + k
                    if j >= L:
                        break
                    acc += pow_cache[k] * costs[j].item()
                    if dones_list[j]:
                        break
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
        def _safe_return(val: float) -> float:
            # Ensure final value finite and positive
            if not (val > 0) or not torch.isfinite(torch.tensor(val)):
                return 1.0
            return float(val)
        with torch.no_grad():
            try:
                embs = torch.stack(list(self.embeddings), dim=0).to(next(risk_head.parameters()).device)
                preds = risk_head(embs).flatten()
            except Exception:
                return 1.0
            # Filter out non-finite predictions
            finite_mask = torch.isfinite(preds)
            preds = preds[finite_mask]
            if preds.numel() < 10:
                return 1.0
            # Quantile & mean (guard against NaN)
            try:
                q_level = float(self.cfg.threshold_percentile) / 100.0
                q_level = min(max(q_level, 0.0), 1.0)
                perc = torch.quantile(preds, q_level)
                mean_pred = preds.mean()
            except Exception:
                return 1.0
            if not (torch.isfinite(perc) and torch.isfinite(mean_pred)):
                return 1.0
            slope = float(getattr(self.cfg, 'slope', 5.0) or 0.0)
            if slope <= 0:
                slope = 5.0  # fallback default
            if abs(perc.item()) > 1e-6:
                slope_scale = perc / slope
            else:
                slope_scale = torch.tensor(1.0, device=preds.device)
            if not torch.isfinite(slope_scale):
                slope_scale = torch.tensor(1.0, device=preds.device)
            diff = mean_pred - perc
            # Normalize & clamp argument to sigmoid to avoid overflow & NaNs
            arg = (diff / (slope_scale + 1e-6)).clamp(min=-20.0, max=20.0)
            gate = torch.sigmoid(arg)
            alpha = float(getattr(self.cfg, 'alpha_modulation', 2.0))
            if not torch.isfinite(gate):
                return 1.0
            scale = 1.0 / (1.0 + alpha * gate.item())
            return _safe_return(scale)
