"""SemanticManager: handles CLIP embeddings, shaping reward, and risk data store.

Minimally invasive: stores its own buffers separate from OmniSafe buffers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
import time

import torch

try:  # lazy import transformers; degrade gracefully if missing
    from transformers import CLIPProcessor, CLIPModel  # type: ignore
except Exception:  # pragma: no cover
    CLIPProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore


@dataclass
class SemanticConfig:
    enable: bool = False
    capture_interval: int = 4
    frame_size: int = 224
    model_name: str = "openai/clip-vit-base-patch16"
    device: str = "cpu"
    beta_start: float = 0.05
    beta_end_step_fraction: float = 0.4
    shaping_enable: bool = False
    risk_enable: bool = False
    modulation_enable: bool = False
    risk_horizon: int = 64
    discount: float = 0.99
    alpha_modulation: float = 2.0
    threshold_percentile: int = 60
    slope: float = 5.0
    window_size: int = 2048
    safe_prompts: List[str] = field(default_factory=lambda: ["clear path to goal", "centered trajectory", "ample clearance"])
    unsafe_prompts: List[str] = field(default_factory=lambda: ["imminent collision", "tight near-obstacle turn", "risky close pass"])


class SemanticManager:
    """Manages semantic embeddings, reward shaping, and risk statistics."""

    def __init__(self, cfg: SemanticConfig, total_steps: int) -> None:
        self.cfg = cfg
        self.total_steps = total_steps
        self.global_step = 0
        self.device = torch.device(cfg.device)
        self._init_clip()
        self.safe_centroid, self.unsafe_centroid = self._encode_prompt_sets(
            cfg.safe_prompts, cfg.unsafe_prompts
        ) if self._clip_ready else (None, None)
        # risk storage
        self.embeddings: Deque[torch.Tensor] = deque(maxlen=cfg.window_size)
        self.costs: Deque[float] = deque(maxlen=cfg.window_size)
        self.future_cost_targets: Deque[float] = deque(maxlen=cfg.window_size)
        self._margin_stats: Deque[float] = deque(maxlen=1000)
        self.last_embed_latency_ms: float = 0.0

    # ---------------- CLIP -----------------
    def _init_clip(self) -> None:
        if not self.cfg.enable:
            self._clip_ready = False
            return
        if CLIPModel is None:
            self._clip_ready = False
            return
        try:
            if CLIPModel is None or CLIPProcessor is None:  # type: ignore
                self._clip_ready = False
                return
            self.clip_model = CLIPModel.from_pretrained(self.cfg.model_name).to(self.device)  # type: ignore[attr-defined]
            self.clip_model.eval()
            self.clip_proc = CLIPProcessor.from_pretrained(self.cfg.model_name)  # type: ignore[attr-defined]
            self._clip_ready = True
        except Exception:  # noqa: BLE001
            self._clip_ready = False

    def _encode_prompt_sets(self, safe: List[str], unsafe: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens = self.clip_proc(text=safe, images=None, return_tensors="pt", padding=True)
            safe_emb = self.clip_model.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()})
            safe_centroid = torch.nn.functional.normalize(safe_emb.mean(0, keepdim=True), dim=-1)
            tokens = self.clip_proc(text=unsafe, images=None, return_tensors="pt", padding=True)
            unsafe_emb = self.clip_model.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()})
            unsafe_centroid = torch.nn.functional.normalize(unsafe_emb.mean(0, keepdim=True), dim=-1)
        return safe_centroid, unsafe_centroid

    def maybe_compute_embedding(self, frame) -> Optional[torch.Tensor]:  # frame: HxWxC ndarray
        if not self._clip_ready:
            return None
        start = time.time()
        with torch.no_grad():
            inputs = self.clip_proc(images=frame, return_tensors="pt").to(self.device)
            img_feat = self.clip_model.get_image_features(**inputs)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)  # (1,d)
        self.last_embed_latency_ms = (time.time() - start) * 1000
        return img_feat[0]  # shape (d,)

    # ------------- Shaping ------------------
    def shaping_term(self, embedding: torch.Tensor) -> float:
        if not (self.cfg.shaping_enable and self._clip_ready and self.safe_centroid is not None and self.unsafe_centroid is not None):
            return 0.0
        with torch.no_grad():
            safe_sim = torch.matmul(embedding, self.safe_centroid[0])
            unsafe_sim = torch.matmul(embedding, self.unsafe_centroid[0])
            margin = (safe_sim - unsafe_sim).item()
        self._margin_stats.append(margin)
        norm_margin = margin
        if len(self._margin_stats) > 30:
            m = sum(self._margin_stats) / len(self._margin_stats)
            var = sum((x - m) ** 2 for x in self._margin_stats) / len(self._margin_stats)
            std = (var ** 0.5) or 1.0
            norm_margin = (margin - m) / std
        beta = self._beta_schedule()
        shaped = beta * max(min(norm_margin, 2.0), -2.0)
        return shaped

    def _beta_schedule(self) -> float:
        frac_end = self.cfg.beta_end_step_fraction
        steps_end = int(self.total_steps * frac_end)
        if steps_end <= 0:
            return 0.0
        if self.global_step >= steps_end:
            return 0.0
        # cosine decay from beta_start to 0
        progress = self.global_step / steps_end
        return self.cfg.beta_start * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    # ------------- Risk Data ---------------
    def record_step(self, embedding: Optional[torch.Tensor], cost: float) -> None:
        if embedding is not None and self.cfg.risk_enable:
            self.embeddings.append(embedding.detach().cpu())
            self.costs.append(cost)
        self.global_step += 1

    def get_recent_risk_batch(self) -> Optional[torch.Tensor]:
        if not self.embeddings:
            return None
        return torch.stack(list(self.embeddings), dim=0)

    def estimated_mean_risk(self, risk_head: Optional[torch.nn.Module]) -> Optional[float]:
        if not (risk_head and self.embeddings):
            return None
        with torch.no_grad():
            embs = torch.stack(list(self.embeddings), dim=0).to(next(risk_head.parameters()).device)
            preds = risk_head(embs)
            return preds.mean().item()

    # ---------- Modulation Helpers ---------
    def modulation_scale(self, risk_head: Optional[torch.nn.Module]) -> Optional[float]:
        if not (self.cfg.modulation_enable and risk_head and self.embeddings):
            return None
        with torch.no_grad():
            embs = torch.stack(list(self.embeddings), dim=0).to(next(risk_head.parameters()).device)
            preds = risk_head(embs).flatten()
            if preds.numel() < 10:
                return 1.0
            perc = torch.quantile(preds, self.cfg.threshold_percentile / 100.0)
            mean_pred = preds.mean()
            # sigmoid gate
            slope_scale = (perc / self.cfg.slope) if abs(perc.item()) > 1e-6 else 1.0
            gate = torch.sigmoid((mean_pred - perc) / (slope_scale + 1e-8))
            scale = 1.0 / (1.0 + self.cfg.alpha_modulation * gate.item())
            return scale
