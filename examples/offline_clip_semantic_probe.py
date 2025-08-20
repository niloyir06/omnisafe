#!/usr/bin/env python3
"""Minimal offline CLIP semantic probe.

Produces ONLY:
    1. Annotated video(s) overlaying: frame index, time(s), safe_sim, unsafe_sim, margin (safe-unsafe).
    2. centroids.json containing: safe/unsafe prompts, centroids, cosine similarity, euclidean distance, angle (deg).

All prior CSV/summary/potential/normalization/YAML logic removed for simplicity.
Edit the SAFE_PROMPTS / UNSAFE_PROMPTS lists below to iterate on discriminability.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore

from transformers import CLIPProcessor, CLIPModel


# -----------------------------
VIDEOS: List[str] = [
    '/vol/research/ni00276/RL/omnisafe/runs/PPOLagSem-{SafetyCarGoal1-v0}/seed-000-2025-08-19-02-49-08/video/epoch-500/eval-episode-0.mp4'
]

# Discriminative prompt candidates (initial active subset below). Expand/refine iteratively.
# Additional suggestions (commented) kept for convenience / experimentation.
SAFE_PROMPTS: List[str] = [
    # High-contrast safe (avoid "circle", "ring", contact verbs)
    "clear open track no warning markers",
    "car progressing smoothly with wide clearance",
    "unobstructed path toward green goal",
    "vehicle in low risk position far from obstacles",
    "stable motion no hazard indicator present",
    "open lane ahead free of obstacles",
]
UNSAFE_PROMPTS: List[str] = [
    # Unsafe prompts leverage visual cue: red circle appears during over/ on obstacle events
    "red warning circle around car indicating obstacle contact",
    "car encircled by red ring while over obstacle",
    "hazard alert red circle shows collision engagement",
    "vehicle inside red warning halo stuck on obstacle",
    "persistent red circle highlighting unsafe overlap",
    "red ring signaling car pressing onto obstacle surface",
]
# High-contrast recommended initial subset (optional):
# ACTIVE_SAFE_PROMPTS = [
#     "red car on clear track far from any obstacle",
#     "unobstructed path ahead toward green goal",
#     "stable controlled progress with no nearby obstacles",
#     "car approaching goal without interference",
#     "hazard free route to green goal",
# ]
# ACTIVE_UNSAFE_PROMPTS = [
#     "red car crashing into purple obstacle",
#     "imminent collision car contacting obstacle",
#     "collision impact blocking progress to goal",
#     "dangerous proximity car inches from collision",
#     "car forced to halt by blocking hazard",
# ]
# More safe ideas:
# "clear corridor to goal no obstacles nearby"
# "red car safely bypassing hazards with ample space"
# "low risk position stable progress toward target"
# More unsafe ideas:
# "red car entangled with multiple obstacles"
# "vehicle wedged against hazard unable to proceed"
# "dangerous proximity car nearly touching obstacle front"
MODEL_ID: str = 'openai/clip-vit-base-patch16'
DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE: str = 'auto'   # 'auto','bf16','fp16','fp32'
EVERY_N: int = 1      # sample every Nth frame
MAX_FRAMES: int = 5000
RESIZE: int = 0       # keep original resolution if 0
OUT_DIR: str = 'clip_probe'

# Temporal smoothing configuration
SMOOTH_ENABLE: bool = True          # master switch
SMOOTH_METHOD: str = 'window'          # 'ema' or 'window'
SMOOTH_ALPHA: float = 0.1           # EMA coefficient (effective smoothing ~1/alpha frames)
SMOOTH_WINDOW: int = 5              # Sliding window size if method == 'window'
SMOOTH_RENORM: bool = True          # Re-normalize smoothed embedding to unit length
ANNOTATE_RAW: bool = False          # If True, include raw (unsmoothed) sims/margin alongside smoothed

# Annotated video output settings
WRITE_ANNOTATED_VIDEO: bool = True
ANNOTATED_SUFFIX: str = '_annotated.mp4'
FONT_SCALE: float = 0.5
FONT_THICKNESS: int = 1
FONT: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR_SAFE = (50, 235, 50)      # BGR
TEXT_COLOR_UNSAFE = (40, 40, 230)
TEXT_COLOR_NEUTRAL = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_BG_ALPHA = 0.4
MAX_TEXT_ROWS = 6

# -----------------------------


# (Normalization/potential removed for simplicity.)


def load_clip(model_id: str, device: str, dtype: str):
    torch_dtype = None
    if dtype == 'bf16':
        torch_dtype = torch.bfloat16
    elif dtype == 'fp16':
        torch_dtype = torch.float16
    elif dtype == 'fp32':
        torch_dtype = torch.float32
    clip_model = CLIPModel.from_pretrained(model_id, use_safetensors=True)
    clip_model.to(device)  # type: ignore[call-arg]
    if torch_dtype is not None:
        try:
            clip_model.to(dtype=torch_dtype)  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            clip_model = clip_model.to(dtype=torch_dtype)  # type: ignore[call-arg]
    processor = CLIPProcessor.from_pretrained(model_id)
    clip_model.eval()
    return clip_model, processor


def video_iter(path: str, every_n: int, max_frames: int, resize: int):
    if cv2 is None:
        raise RuntimeError('opencv-python (cv2) required for video reading.')
    cap = cv2.VideoCapture(path)
    idx = 0
    used = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            if resize > 0:
                frame = cv2.resize(frame, (resize, resize))
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield used, idx / fps, Image.fromarray(frame_rgb), frame  # include original BGR frame for annotation
            used += 1
            if used >= max_frames:
                break
        idx += 1
    cap.release()


def embed_images(clip_model: CLIPModel, processor: CLIPProcessor, images: Sequence[Image.Image], device: str):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors='pt')
        try:
            inputs = inputs.to(device)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_image_features(**inputs)  # type: ignore[arg-type]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def embed_texts(clip_model: CLIPModel, processor: CLIPProcessor, prompts: Sequence[str], device: str):
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors='pt', padding=True)
        try:
            inputs = inputs.to(device)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = clip_model.get_text_features(**inputs)  # type: ignore[arg-type]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def main():
    if not VIDEOS:
        raise ValueError('Populate VIDEOS with at least one video path.')
    os.makedirs(OUT_DIR, exist_ok=True)
    clip_model, processor = load_clip(MODEL_ID, DEVICE, DTYPE)
    safe_embs = embed_texts(clip_model, processor, SAFE_PROMPTS, DEVICE)
    unsafe_embs = embed_texts(clip_model, processor, UNSAFE_PROMPTS, DEVICE)
    safe_centroid = safe_embs.mean(dim=0)
    unsafe_centroid = unsafe_embs.mean(dim=0)
    with torch.no_grad():
        cos = torch.cosine_similarity(safe_centroid, unsafe_centroid, dim=0).item()
        euclid = float((2 * (1 - cos)) ** 0.5)
        angle_deg = float(torch.rad2deg(torch.arccos(torch.clamp(torch.tensor(cos), -1.0, 1.0))).item())
    centroid_payload = {
        'safe_prompts': SAFE_PROMPTS,
        'unsafe_prompts': UNSAFE_PROMPTS,
        'cosine_similarity': cos,
        'euclidean_distance': euclid,
        'angle_degrees': angle_deg,
        'safe_centroid': safe_centroid.cpu().tolist(),
        'unsafe_centroid': unsafe_centroid.cpu().tolist(),
    }
    with open(Path(OUT_DIR) / 'centroids.json', 'w') as f:
        json.dump(centroid_payload, f, indent=2)
    print('Wrote centroid metrics to', Path(OUT_DIR)/'centroids.json')

    for vid in VIDEOS:
        path = Path(vid)
        start = time.time()
        writer = None
        frame_count = 0
        # Smoothing state
        ema_embed = None
        window_buf: list[torch.Tensor] = []
        raw_margins: list[float] = []
        smooth_margins: list[float] = []
        for frame_idx, t_sec, pil_img, bgr_frame in video_iter(str(path), EVERY_N, MAX_FRAMES, RESIZE):
            img_feat = embed_images(clip_model, processor, [pil_img], DEVICE)[0]
            raw_safe_sim = torch.cosine_similarity(img_feat, safe_centroid, dim=0).item()
            raw_unsafe_sim = torch.cosine_similarity(img_feat, unsafe_centroid, dim=0).item()
            raw_margin = raw_safe_sim - raw_unsafe_sim
            raw_margins.append(raw_margin)

            use_feat = img_feat
            if SMOOTH_ENABLE:
                if SMOOTH_METHOD == 'ema':
                    if ema_embed is None:
                        ema_embed = img_feat.clone()
                    else:
                        ema_embed = (1 - SMOOTH_ALPHA) * ema_embed + SMOOTH_ALPHA * img_feat
                    use_feat = ema_embed
                elif SMOOTH_METHOD == 'window':
                    window_buf.append(img_feat)
                    if len(window_buf) > SMOOTH_WINDOW:
                        window_buf.pop(0)
                    use_feat = torch.stack(window_buf, dim=0).mean(0)
                if SMOOTH_RENORM:
                    use_feat = use_feat / use_feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            safe_sim = torch.cosine_similarity(use_feat, safe_centroid, dim=0).item()
            unsafe_sim = torch.cosine_similarity(use_feat, unsafe_centroid, dim=0).item()
            margin = safe_sim - unsafe_sim
            smooth_margins.append(margin)
            if WRITE_ANNOTATED_VIDEO and cv2 is not None:
                if writer is None:
                    h, w, _ = bgr_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = str(Path(OUT_DIR) / f"{path.stem}_annotated.mp4")
                    writer = cv2.VideoWriter(out_path, fourcc, max(1, 30 // EVERY_N), (w, h))
                lines = [f"frame:{frame_idx}", f"t:{t_sec:.2f}s"]
                if ANNOTATE_RAW and SMOOTH_ENABLE:
                    lines += [
                        f"raw_safe:{raw_safe_sim:+.3f}",
                        f"raw_unsafe:{raw_unsafe_sim:+.3f}",
                        f"raw_margin:{raw_margin:+.3f}",
                        f"sm_safe:{safe_sim:+.3f}",
                        f"sm_unsafe:{unsafe_sim:+.3f}",
                        f"sm_margin:{margin:+.3f}",
                    ]
                else:
                    lines += [
                        f"safe:{safe_sim:+.3f}",
                        f"unsafe:{unsafe_sim:+.3f}",
                        f"margin:{margin:+.3f}",
                    ]
                y0 = 5
                x0 = 5
                line_h = 18
                box_h = line_h * min(MAX_TEXT_ROWS, len(lines)) + 6
                box_w = 0
                for ln in lines[:MAX_TEXT_ROWS]:
                    box_w = max(box_w, 6 + len(ln) * 8)
                overlay = bgr_frame.copy()
                cv2.rectangle(overlay, (x0 - 2, y0 - 2), (x0 + box_w, y0 + box_h), TEXT_BG_COLOR, -1)
                cv2.addWeighted(overlay, TEXT_BG_ALPHA, bgr_frame, 1 - TEXT_BG_ALPHA, 0, bgr_frame)
                for i, ln in enumerate(lines[:MAX_TEXT_ROWS]):
                    color = TEXT_COLOR_NEUTRAL
                    if ln.startswith('safe:'):
                        color = TEXT_COLOR_SAFE
                    elif ln.startswith('unsafe:'):
                        color = TEXT_COLOR_UNSAFE
                    cv2.putText(
                        bgr_frame,
                        ln,
                        (x0, y0 + (i + 1) * line_h),
                        FONT,
                        FONT_SCALE,
                        color,
                        FONT_THICKNESS,
                        lineType=cv2.LINE_AA,
                    )
                writer.write(bgr_frame)
            frame_count += 1
        if writer is not None:
            writer.release()
            print(f"Annotated video written for {path} ({frame_count} frames) in {time.time()-start:.1f}s")
        else:
            print(f"No frames processed for {path} (check path or EVERY_N/MAX_FRAMES settings)")
        # Variance report
        if SMOOTH_ENABLE and raw_margins and smooth_margins:
            import math
            def _std(xs: list[float]):
                m = sum(xs)/len(xs)
                return math.sqrt(sum((x-m)**2 for x in xs)/len(xs))
            raw_std = _std(raw_margins)
            smooth_std = _std(smooth_margins)
            print(f"Temporal smoothing ({SMOOTH_METHOD}) raw_margin_std={raw_std:.4f} smoothed_margin_std={smooth_std:.4f} reduction={(1 - smooth_std/raw_std)*100 if raw_std>1e-9 else 0:.1f}%")


if __name__ == '__main__':
    main()
