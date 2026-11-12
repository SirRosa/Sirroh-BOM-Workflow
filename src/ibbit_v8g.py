#!/usr/bin/env python3
"""IbBit v8G — restricted 3-color soil palette.

A side variant of IbBit v8; the general-purpose IbBit script remains unchanged.
Only these three existing v8 palette entries are used, in fixed order:
  source shadows -> neutral/dark (#222522)
  source midtones -> brown/dark (#3A291A)
  source highlights -> neutral/mid (#485246)

Before processing: scale/letterbox RGBA to 256x256. Edge-aware luminance
smoothing, global batch thresholds and small-color-island removal (<16 px).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

SIZE = 256
# Exact RGB entries from IbBit v8 — not recalculated from source colors.
PALETTE = np.array([
    [34, 37, 34],   # 0: neutral.dark (darkest pixels)
    [58, 41, 26],   # 1: brown.dark (dominant midtones)
    [72, 82, 70],   # 2: neutral.mid (brightest pixels)
], dtype=np.uint8)
LABELS = ("neutral_dark", "brown_dark", "neutral_mid")
SHADOW_FRACTION = 0.12
LIGHT_FRACTION = 0.12
MIN_COMPONENT_SIZE = 16


def validate_palette() -> None:
    assert PALETTE.shape == (3, 3)
    lum = PALETTE.astype(np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    if not (lum[0] < lum[1] < lum[2]):
        raise ValueError("Palette must go from darkest to lightest: neutral dark, brown dark, neutral mid")


validate_palette()


def fit_rgba(image: Image.Image, size: int = SIZE) -> Image.Image:
    """Aspect-safe 256x256 scaling before quantization; premultiplied alpha."""
    source = image.convert("RGBA")
    if source.size == (size, size):
        return source.copy()
    scale = min(size / source.width, size / source.height)
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    arr = np.asarray(source, dtype=np.float32)
    a = arr[..., 3:4] / 255.0
    premul = arr[..., :3] * a
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rgb = cv2.resize(premul, (width, height), interpolation=interpolation)
    alpha = cv2.resize(arr[..., 3], (width, height), interpolation=interpolation)
    if rgb.ndim == 2:
        rgb = rgb[..., None]
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint(rgb / np.maximum(alpha / 255.0, 1e-5)), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(np.rint(alpha[..., 0]), 0, 255).astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = 0
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.alpha_composite(Image.fromarray(rgba, "RGBA"), ((size - width)//2, (size - height)//2))
    return result


def prepared_lightness(img: Image.Image, *, size: int = SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgba = np.asarray(fit_rgba(img, size), dtype=np.uint8)
    alpha = rgba[..., 3].copy()
    visible = alpha >= 8
    rgb = rgba[..., :3].copy()
    if np.any(~visible):
        rgb = cv2.inpaint(rgb, (~visible).astype(np.uint8), 2, cv2.INPAINT_TELEA)
    # Preserve major structures, remove subpixel tonal fluctuation before reducing to 3 colors.
    smooth = cv2.bilateralFilter(rgb, 7, 30, 4)
    lab = cv2.cvtColor(smooth.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    light = cv2.bilateralFilter(lab[..., 0], 5, 7.5, 3)
    return light, visible, alpha


def compute_thresholds(lightness_arrays: Iterable[np.ndarray], masks: Iterable[np.ndarray],
                       shadow_fraction: float = SHADOW_FRACTION,
                       light_fraction: float = LIGHT_FRACTION) -> tuple[float, float]:
    if shadow_fraction <= 0 or light_fraction <= 0 or shadow_fraction + light_fraction >= 1:
        raise ValueError("Each accent fraction must be >0 and total <1")
    samples = [data[mask] for data, mask in zip(lightness_arrays, masks) if np.any(mask)]
    if not samples:
        return (0.0, 1.0)
    sample = np.concatenate(samples)
    t0, t1 = np.quantile(sample, (shadow_fraction, 1.0 - light_fraction)).tolist()
    if t1 <= t0:
        t1 = t0 + 1e-4
    return float(t0), float(t1)


def quantize(lightness: np.ndarray, visible: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    t0, t1 = thresholds
    levels = np.full(lightness.shape, 1, dtype=np.uint8)
    levels[visible & (lightness < t0)] = 0
    levels[visible & (lightness >= t1)] = 2
    return levels


def count_small_components(levels: np.ndarray, visible: np.ndarray, min_area: int = MIN_COMPONENT_SIZE) -> int:
    small = 0
    for c in range(3):
        mask = (visible & (levels == c)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        small += int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < min_area))
    return small


def cleanup_color_islands(levels: np.ndarray, visible: np.ndarray, *, min_area: int = MIN_COMPONENT_SIZE,
                          max_passes: int = 8) -> tuple[np.ndarray, dict]:
    """Replace each connected monochrome region smaller than min_area pixels
    with the touching color having most 8-neighbor contacts. Keep alpha unchanged.
    """
    result = levels.copy()
    h, w = result.shape
    before = count_small_components(result, visible, min_area)
    pixels_changed = 0
    passes_used = 0
    for iteration in range(max_passes):
        changed = False
        for color in range(3):
            mask = (visible & (result == color)).astype(np.uint8)
            n, comps, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for cid in range(1, n):
                area = int(stats[cid, cv2.CC_STAT_AREA])
                if area >= min_area:
                    continue
                x, y, cw, ch = (int(a) for a in stats[cid,:4])
                ys, xs = np.where(comps[y:y+ch, x:x+cw] == cid)
                ys, xs = ys+y, xs+x
                contacts = {0:0,1:0,2:0}
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                    ny, nx = ys+dy, xs+dx
                    in_bounds = (ny>=0)&(ny<h)&(nx>=0)&(nx<w)
                    ny, nx = ny[in_bounds],nx[in_bounds]
                    touching = visible[ny,nx] & (comps[ny,nx] != cid)
                    neighbors = result[ny[touching], nx[touching]]
                    for candidate in (0,1,2):
                        if candidate != color:
                            contacts[candidate] += int(np.count_nonzero(neighbors==candidate))
                contacts.pop(color)
                if max(contacts.values(), default=0)==0:
                    continue  # isolated disconnected sprite with no other color touching
                # Number of contacts first, closest tone second, stable number third.
                replacement = min(contacts, key=lambda cand: (-contacts[cand],abs(cand-color),cand))
                same = result[ys,xs] == color
                if np.any(same):
                    result[ys[same],xs[same]] = replacement
                    pixels_changed += int(np.count_nonzero(same))
                    changed = True
        passes_used=iteration+1
        if not changed:
            break
    return result, {'islands_before':before,
                    'islands_after':count_small_components(result,visible,min_area),
                    'pixels_reassigned':pixels_changed,'passes':passes_used,'min_area':min_area}


def render(lightness: np.ndarray, visible: np.ndarray, alpha: np.ndarray,
           thresholds: tuple[float,float], *, min_area: int = MIN_COMPONENT_SIZE) -> tuple[Image.Image,dict]:
    levels = quantize(lightness,visible,thresholds)
    levels,stats = cleanup_color_islands(levels,visible,min_area=min_area)
    out=np.zeros((*levels.shape,4),dtype=np.uint8)
    out[...,3]=alpha
    out[visible,:3]=PALETTE[levels[visible]]
    out[~visible,:3]=0
    counts=np.bincount(levels[visible],minlength=3)
    report={'colors':dict(zip(LABELS,[int(x) for x in counts])),
            'thresholds_lab_l':[round(t,4) for t in thresholds],
            'size':[int(out.shape[1]),int(out.shape[0])],
            'alpha_preserved':bool(np.array_equal(out[...,3],alpha)),
            **stats}
    return Image.fromarray(out,'RGBA'),report


def run_batch(input_dir: Path, output_dir: Path, *, size: int = SIZE,
              shadow_fraction: float = SHADOW_FRACTION, light_fraction: float = LIGHT_FRACTION,
              min_area: int = MIN_COMPONENT_SIZE) -> dict:
    paths=sorted(input_dir.rglob('*.png'))
    if not paths: raise ValueError('No PNG files in input directory')
    in_resolved=input_dir.resolve(); out_resolved=output_dir.resolve()
    if in_resolved == out_resolved or in_resolved in out_resolved.parents:
        raise ValueError('Output directory must not be nested in input directory')
    prepared={}
    for p in paths:
        with Image.open(p) as im:
            prepared[p.relative_to(input_dir)] = prepared_lightness(im,size=size)
    t=compute_thresholds([tr[0] for tr in prepared.values()],
                         [tr[1] for tr in prepared.values()],shadow_fraction,light_fraction)
    summary={}
    for relative,(light,visible,alpha) in prepared.items():
        result,report=render(light,visible,alpha,t,min_area=min_area)
        target=output_dir/relative;target.parent.mkdir(parents=True,exist_ok=True)
        result.save(target, 'PNG',optimize=True)
        summary[str(relative)]=report
        print(f'IbBit v8G: {relative}',flush=True)
    manifest={'version':'IbBit v8G','source':'IbBit v8 palette entries',
              'colors':{name:PALETTE[i].tolist() for i,name in enumerate(LABELS)},
              'global_thresholds_lab_l':[round(v,4) for v in t],
              'shadow_fraction':shadow_fraction,'light_fraction':light_fraction,
              'min_component_size':min_area,'count':len(summary),'files':summary}
    (output_dir/'IbBit_v8G_raport.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description='IbBit v8G — 3 fixed colors from v8 for dirt textures')
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--size',type=int,default=SIZE)
    parser.add_argument('--shadow-fraction',type=float,default=SHADOW_FRACTION)
    parser.add_argument('--light-fraction',type=float,default=LIGHT_FRACTION)
    parser.add_argument('--min-area',type=int,default=MIN_COMPONENT_SIZE)
    args=parser.parse_args()
    if not 32<=args.size<=2048:parser.error('--size must be in [32,2048]')
    if not 1<=args.min_area<=256:parser.error('--min-area must be in [1,256]')
    run_batch(args.input_dir,args.output_dir,size=args.size,
              shadow_fraction=args.shadow_fraction,light_fraction=args.light_fraction,min_area=args.min_area)


if __name__=='__main__':main()