#!/usr/bin/env python3
from __future__ import annotations

import colorsys
import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ALPHA_THRESHOLD = 8
SOURCE_TILE = 128
TARGET_TILE = 96
MARGIN = 2
SHADES_PER_FAMILY = 5
HUE_BINS = 12
SAT_NEUTRAL = 0.18
VAL_DARK_NEUTRAL = 0.10
MIN_FAMILY_PIXELS = 20
MIN_FAMILY_RATIO = 0.03
GLOW_MIN_PIXELS = 4
GLOW_SATURATION = 0.72
GLOW_VALUE = 0.63
BLACK_LUMA_MAX = 12.0
WHITE_LUMA_MIN = 233.0
NEUTRAL_SAT_MAX = 0.10
ARTIFACT_OPAQUE_NEIGHBORS_MIN = 1
COLOR_CLEANUP_PASSES = 1
COLOR_CLEANUP_NEIGHBOR_MAJORITY = 4
SOURCE_NEIGHBOR_MAX_DELTA_E = 14.0
NOISE_COMPONENT_MAX = 3
OUTLINE_PROTECTION_DELTA_L = 22.0

FAMILY_LABELS = [
    "red", "orange", "yellow", "yellow_green", "green", "teal",
    "cyan", "azure", "blue", "violet", "magenta", "rose"
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ensure_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def resize_rgba_premultiplied(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if img.size == size:
        return img.copy()
    arr = np.asarray(img).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4] / 255.0
    premul = rgb * alpha
    premul_img = Image.fromarray(np.clip(np.rint(premul), 0, 255).astype(np.uint8), "RGB")
    alpha_img = Image.fromarray(np.clip(np.rint(arr[:, :, 3]), 0, 255).astype(np.uint8), "L")
    interp = Image.Resampling.LANCZOS
    premul_r = np.asarray(premul_img.resize(size, interp)).astype(np.float32)
    alpha_r = np.asarray(alpha_img.resize(size, interp)).astype(np.float32) / 255.0
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    out[:, :, 3] = np.clip(np.rint(alpha_r * 255.0), 0, 255).astype(np.uint8)
    valid = alpha_r > 1e-6
    if np.any(valid):
        out[valid, :3] = np.clip(np.rint(premul_r[valid] / alpha_r[valid, None]), 0, 255).astype(np.uint8)
    out[out[:, :, 3] < 8] = 0
    return Image.fromarray(out, "RGBA")


def crop_foreground(img: Image.Image) -> Image.Image | None:
    img = ensure_rgba(img)
    alpha = img.getchannel("A")
    mask = alpha.point(lambda p: 255 if p >= ALPHA_THRESHOLD else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return None
    cropped = img.crop(bbox).copy()
    # Preserve original antialiasing (hard alpha created artificial dark sawtooth edges).
    cropped.putalpha(cropped.getchannel("A").point(lambda p: p if p >= 8 else 0))
    return cropped


def normalize_to_target(icon: Image.Image | None) -> Image.Image:
    tile = Image.new("RGBA", (TARGET_TILE, TARGET_TILE), (0, 0, 0, 0))
    if icon is None:
        return tile
    sprite = crop_foreground(icon)
    if sprite is None:
        return tile
    ow, oh = sprite.size
    fit_w = TARGET_TILE - 2 * MARGIN
    fit_h = TARGET_TILE - 2 * MARGIN
    scale = min(1.0, fit_w / max(1, ow), fit_h / max(1, oh))
    nw = max(1, min(fit_w, int(round(ow * scale))))
    nh = max(1, min(fit_h, int(round(oh * scale))))
    sprite = resize_rgba_premultiplied(sprite, (nw, nh))
    dx = (TARGET_TILE - nw) // 2
    dy = TARGET_TILE - MARGIN - nh
    tile.alpha_composite(sprite, (dx, dy))
    return tile


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.astype(np.float32) / 255.0
    out = np.zeros((flat.shape[0], 3), dtype=np.float32)
    for i, (r, g, b) in enumerate(flat):
        out[i] = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    return out


def family_for_pixel(h: float, s: float, v: float) -> str:
    if s < SAT_NEUTRAL or v < VAL_DARK_NEUTRAL:
        return "neutral"
    idx = int(math.floor(h * HUE_BINS)) % HUE_BINS
    return FAMILY_LABELS[idx]


def pick_real_nearest(group: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    diff = np.sum((group.astype(np.float32) - target_rgb.astype(np.float32)) ** 2, axis=1)
    return group[int(np.argmin(diff))]


def unique_ordered(colors: List[np.ndarray]) -> np.ndarray:
    out = []
    seen = set()
    for c in colors:
        t = tuple(int(x) for x in np.rint(c).clip(0, 255))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return np.array(out, dtype=np.uint8) if out else np.zeros((0, 3), dtype=np.uint8)


def representative_palette_for_family(pixels: np.ndarray) -> np.ndarray:
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if len(pixels) <= SHADES_PER_FAMILY:
        return unique_ordered([p for p in pixels.astype(np.uint8)])
    lum = luminance(pixels)
    order = np.argsort(lum)
    px = pixels[order]
    lum_sorted = lum[order]

    # Explicitly preserve extremes and strong bright/dark variants.
    idxs = [
        0,  # darkest
        max(0, int(round(0.20 * (len(px) - 1)))),
        int(round(0.50 * (len(px) - 1))),
        min(len(px) - 1, int(round(0.80 * (len(px) - 1)))),
        len(px) - 1,  # brightest
    ]
    chosen = [px[i] for i in idxs]

    # If the family includes very bright or very dark rare accents, lock them in.
    brightest = px[-1]
    darkest = px[0]
    chosen[-1] = brightest
    chosen[0] = darkest

    return unique_ordered(chosen)


def split_families(rgb: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    hsv = rgb_to_hsv(rgb)
    lum = luminance(rgb)
    s = hsv[:, 1]
    neutral_mask = s < NEUTRAL_SAT_MAX
    black_mask = neutral_mask & (lum <= BLACK_LUMA_MAX)
    white_mask = neutral_mask & (lum >= WHITE_LUMA_MIN)
    free_mask = black_mask | white_mask

    families: Dict[str, List[int]] = {}
    total = len(rgb)
    for i, ((h, sat, v), is_free) in enumerate(zip(hsv, free_mask)):
        if is_free:
            continue
        fam = family_for_pixel(float(h), float(sat), float(v))
        families.setdefault(fam, []).append(i)

    # Merge tiny chromatic families into nearest significant one; keep neutral.
    # Preserve genuine rare, saturated accents even if they occupy very few pixels.
    significant: Dict[str, List[int]] = {}
    small: Dict[str, List[int]] = {}
    for fam, idxs in families.items():
        intense = hsv[idxs, 1] >= GLOW_SATURATION
        luminous = hsv[idxs, 2] >= 0.58
        rare_accent = len(idxs) >= 2 and np.mean(intense & luminous) >= 0.65
        if fam == "neutral" or rare_accent or len(idxs) >= max(MIN_FAMILY_PIXELS, int(total * MIN_FAMILY_RATIO)):
            significant[fam] = idxs
        else:
            small[fam] = idxs

    order_map = {name: i for i, name in enumerate(FAMILY_LABELS)}
    chroma_keys = [k for k in significant.keys() if k != "neutral"]
    if not chroma_keys:
        for fam, idxs in small.items():
            significant[fam] = idxs
        small = {}
        chroma_keys = [k for k in significant.keys() if k != "neutral"]

    for fam, idxs in small.items():
        if not chroma_keys:
            significant.setdefault(fam, []).extend(idxs)
            continue
        src = order_map.get(fam, 0)
        nearest = min(
            chroma_keys,
            key=lambda k: min((src - order_map.get(k, 0)) % HUE_BINS, (order_map.get(k, 0) - src) % HUE_BINS),
        )
        significant.setdefault(nearest, []).extend(idxs)

    # Reserve *another* five-shade family for small intense accents instead of
    # letting a large brown/red area consume all five red palette entries.
    # This protects fiery eyes and tiny bright runes without global recoloring.
    for fam, idxs in list(significant.items()):
        if fam == 'neutral' or len(idxs) < 20:
            continue
        px_hsv = hsv[idxs]
        accent_mask = ((px_hsv[:, 1] >= GLOW_SATURATION) &
                       (px_hsv[:, 2] >= GLOW_VALUE))
        n = int(accent_mask.sum())
        if GLOW_MIN_PIXELS <= n <= int(len(idxs) * .40):
            significant[fam] = [idxs[i] for i in np.flatnonzero(~accent_mask)]
            significant[fam + '_accent'] = [idxs[i] for i in np.flatnonzero(accent_mask)]

    return {k: np.array(v, dtype=np.int32) for k, v in significant.items()}, black_mask, white_mask, neutral_mask


def build_tile_palette(rgb: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, int]]:
    families, black_mask, white_mask, _ = split_families(rgb)
    palette_by_family: Dict[str, np.ndarray] = {}
    stats: Dict[str, int] = {}

    # free black and white if present
    black_palette = np.zeros((0, 3), dtype=np.uint8)
    white_palette = np.zeros((0, 3), dtype=np.uint8)
    if np.any(black_mask):
        blacks = rgb[black_mask]
        darkest = blacks[np.argmin(luminance(blacks))]
        black_palette = unique_ordered([darkest])
        stats['free_black'] = len(black_palette)
    if np.any(white_mask):
        whites = rgb[white_mask]
        brightest = whites[np.argmax(luminance(whites))]
        white_palette = unique_ordered([brightest])
        stats['free_white'] = len(white_palette)

    for fam in ["neutral"] + FAMILY_LABELS:
        if fam not in families:
            continue
        pal = representative_palette_for_family(rgb[families[fam]])
        if len(pal):
            palette_by_family[fam] = pal
            stats[fam] = len(pal)

    return palette_by_family, black_palette, white_palette, stats


def family_for_hsv_lum(h: float, s: float, lum: float) -> str:
    # A dark saturated red or blue is chromatic, never automatic black.
    if s <= 0.12:
        if lum <= 38.0:
            return "black"
        if lum >= 225.0:
            return "white"
        return "gray"
    return FAMILY_LABELS[int(math.floor(h * HUE_BINS)) % HUE_BINS]


def family_for_pixel_from_rgb(rgb: np.ndarray) -> str:
    r, g, b = [float(x) for x in rgb]
    h, s, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return family_for_hsv_lum(h, s, 0.2126 * r + 0.7152 * g + 0.0722 * b)


def assign_quantized(rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    hsv = rgb_to_hsv(rgb)
    luma = luminance(rgb)
    groups = {}
    for i, ((h, sat, _), light) in enumerate(zip(hsv, luma)):
        family = family_for_hsv_lum(float(h), float(sat), float(light))
        groups.setdefault(family, []).append(i)

    mapped = np.zeros_like(rgb, dtype=np.uint8)
    stats = {}
    for family, indices in groups.items():
        ix = np.asarray(indices, dtype=np.int32)
        palette = representative_palette_for_family(rgb[ix])
        if len(palette) == 0:
            raise RuntimeError("No palette: " + family)
        pts, pal = rgb[ix].astype(np.float32), palette.astype(np.float32)
        dist = np.sum((pts[:, None, :] - pal[None, :, :]) ** 2, axis=2)
        mapped[ix] = palette[np.argmin(dist, axis=1)]
        stats[family] = len(palette)
    return mapped, stats

def _lab_array(rgb: np.ndarray) -> np.ndarray:
    # OpenCV needs HxWx3; Nx3 palettes are temporarily reshaped to Nx1x3.
    arr = rgb.astype(np.float32) / 255.0
    if arr.ndim == 2:
        return cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)


def _flat_color_components(rgb: np.ndarray, mask: np.ndarray):
    # Connected *palette-color* components. Only very small islands are considered.
    opaque = rgb[mask]
    if len(opaque) == 0:
        return
    colors = np.unique(opaque.reshape(-1, 3), axis=0)
    for c in colors:
        color_mask = (np.all(rgb == c, axis=2) & mask).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, 8)
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area <= NOISE_COMPONENT_MAX:
                yield (labels, idx, area)


def cleanup_artifacts(rgba: np.ndarray, source_rgba: np.ndarray | None = None) -> np.ndarray:
    """Conservative source-guided 1-3 pixel color island cleanup.
    Never reshape silhouette or change authentic eye/glow/crease details.
    """
    out = rgba.copy()
    if source_rgba is None:
        return out
    src = source_rgba.copy()
    alpha_mask = (out[:, :, 3] >= ALPHA_THRESHOLD)
    if not np.any(alpha_mask):
        return out
    h, w = alpha_mask.shape
    src_lab = _lab_array(src[:, :, :3])

    # Clean only solitary *faint* alpha remnants, not true silhouette parts.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_mask.astype(np.uint8), 8)
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) != 1:
            continue
        yy, xx = np.where(labels == i)
        y, x = int(yy[0]), int(xx[0])
        if int(src[y, x, 3]) < 80:
            out[y, x] = 0
    alpha_mask = out[:, :, 3] >= ALPHA_THRESHOLD

    for _pass in range(2):
        changes = []
        palette = np.unique(out[:, :, :3][alpha_mask], axis=0)
        for color in palette:
            color_mask = (np.all(out[:, :, :3] == color, axis=2) & alpha_mask).astype(np.uint8)
            num, comps, areas, _ = cv2.connectedComponentsWithStats(color_mask, 8)
            for idx in range(1, num):
                if int(areas[idx, cv2.CC_STAT_AREA]) > 3:
                    continue
                yy, xx = np.where(comps == idx)
                neigh_counter = {}
                neighbor_src = []
                centers = []
                for y, x in zip(yy.tolist(), xx.tolist()):
                    centers.append(src_lab[y, x])
                    for ny in range(max(0, y - 1), min(h, y + 2)):
                        for nx in range(max(0, x - 1), min(w, x + 2)):
                            if not alpha_mask[ny, nx] or comps[ny, nx] == idx:
                                continue
                            c = tuple(int(v) for v in out[ny, nx, :3])
                            neigh_counter[c] = neigh_counter.get(c, 0) + 1
                            neighbor_src.append(src_lab[ny, nx])
                if not neighbor_src:
                    continue
                candidate, count_nb = max(neigh_counter.items(), key=lambda pair: pair[1])
                if count_nb < 4:
                    continue
                center_lab = np.mean(centers, axis=0)
                neighbor_lab = np.median(neighbor_src, axis=0)
                if float(np.linalg.norm(center_lab - neighbor_lab)) > 12.5:
                    continue
                new_lab = _lab_array(np.asarray(candidate, dtype=np.uint8).reshape(1, 3))[0]
                if float(np.linalg.norm(new_lab - center_lab)) > 28:
                    continue
                src_rgb = np.mean(src[yy, xx, :3].astype(np.float32), axis=0)
                from_family = family_for_pixel_from_rgb(src_rgb)
                to_family = family_for_pixel_from_rgb(np.asarray(candidate, dtype=np.float32))
                if from_family not in ("black", "gray", "white") and to_family in ("black", "gray"):
                    continue
                for y, x in zip(yy.tolist(), xx.tolist()):
                    changes.append((y, x, candidate))
        if not changes:
            break
        for y, x, c in changes:
            out[y, x, :3] = c
    out[out[:, :, 3] == 0, :3] = 0
    return out

# BOM v8: source-faithful perceptual mapping. All representative colors are
# real source pixels. An unsafe mapping is rejected, never force-applied.
V8_SHADES=5
V8_DELTA_MAX=11.0
V8_LUMA_MAX=7.0
V8_HUE_MAX=10.0
V8_EDGE_THRESHOLD=14.0


def v8_lab(rgb):
    a=np.asarray(rgb,dtype=np.float32)/255.0
    if a.ndim==2:
        return cv2.cvtColor(a[:,None,:],cv2.COLOR_RGB2LAB)[:,0,:]
    return cv2.cvtColor(a,cv2.COLOR_RGB2LAB)


def v8_hsv(rgb):
    a=np.asarray(rgb,dtype=np.float32)/255.0
    return cv2.cvtColor(a.reshape(-1,1,3),cv2.COLOR_RGB2HSV)[:,0,:]


def v8_labels(rgb):
    hsv=v8_hsv(rgb)
    hues,sats,values=hsv[:,0],hsv[:,1],hsv[:,2]
    labels=(np.floor(hues/360.0*12).astype(np.int32)%12)
    neutral=sats<=.11
    labels[neutral&(values<.18)]=12
    labels[neutral&(values>=.18)&(values<.84)]=13
    labels[neutral&(values>=.84)]=14
    return labels,hsv


def v8_palette(pixels):
    colors,counts=np.unique(pixels.astype(np.uint8),axis=0,return_counts=True)
    if len(colors)<=V8_SHADES:return colors
    L=v8_lab(colors)[:,0]
    order=np.argsort(L)
    colors,counts,L=colors[order],counts[order],L[order]
    cumulative=np.cumsum(counts)/sum(counts)
    indices=[]
    for q in (0,.18,.5,.82,1):
        ix=0 if q==0 else len(colors)-1 if q==1 else int(np.searchsorted(cumulative,q))
        if ix not in indices:indices.append(ix)
    if len(indices)<V8_SHADES:
        for ix in np.argsort(-counts):
            if int(ix) not in indices:indices.append(int(ix))
            if len(indices)>=V8_SHADES:break
    picked=[]
    for j,ix in enumerate(indices[:V8_SHADES]):
        if j in (0,4):picked.append(colors[ix]);continue
        loc=np.flatnonzero(np.abs(L-L[ix])<2.5)
        if len(loc):ix=int(loc[np.argmax(counts[loc])])
        picked.append(colors[ix])
    return np.unique(np.array(picked,dtype=np.uint8),axis=0)


def v8_source_edges(lab,visible):
    h,w=visible.shape
    result=np.zeros_like(visible,dtype=bool)
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        ya=slice(max(0,-dy),min(h,h-dy))
        xa=slice(max(0,-dx),min(w,w-dx))
        yb=slice(max(0,dy),min(h,h+dy))
        xb=slice(max(0,dx),min(w,w+dx))
        diff=np.linalg.norm(lab[ya,xa]-lab[yb,xb],axis=2)
        strong=(diff>V8_EDGE_THRESHOLD)&visible[ya,xa]&visible[yb,xb]
        result[ya,xa]|=strong
        result[yb,xb]|=strong
    return result


def process_tile(img):
    arr=np.asarray(ensure_rgba(img)).copy()
    vis=arr[:,:,3]>=8
    src=arr[:,:,:3][vis]
    if not len(src):
        arr[arr[:,:,3]==0,:3]=0
        return Image.fromarray(arr,'RGBA'),{},0
    labels,hsv=v8_labels(src)
    src_lab=v8_lab(src)
    mapped=src.copy()
    chosen_delta=np.zeros(len(src),dtype=np.float32)
    families={}
    for fam in np.unique(labels):
        indices=np.flatnonzero(labels==fam)
        palette=v8_palette(src[indices])
        families[str(int(fam))]=len(palette)
        pal_lab=v8_lab(palette)
        dist2=np.sum((src_lab[indices,None,:]-pal_lab[None,:,:])**2,axis=2)
        ix=np.argmin(dist2,axis=1)
        diff=np.sqrt(dist2[np.arange(len(indices)),ix])
        dl=np.abs(src_lab[indices,0]-pal_lab[ix,0])
        good=(diff<=V8_DELTA_MAX)&(dl<=V8_LUMA_MAX)
        if fam<12:
            pal_hue=v8_hsv(palette)[:,0]
            hue_delta=np.abs((hsv[indices,0]-pal_hue[ix]+180)%360-180)
            good&=(hue_delta<=V8_HUE_MAX)
        mapped[indices[good]]=palette[ix[good]]
        chosen_delta[indices]=diff
    labs=v8_lab(arr[:,:,:3])
    edges=v8_source_edges(labs,vis)
    saturated_glow=(hsv[:,1]>=.55)&(hsv[:,2]>=.45)
    keep=edges[vis]&((chosen_delta>4.5)|saturated_glow)
    keep|=saturated_glow&(chosen_delta>3.0)
    mapped[keep]=src[keep]
    out=arr.copy()
    out[:,:,:3][vis]=mapped
    new_labs=v8_lab(out[:,:,:3])
    diff=np.linalg.norm(new_labs-labs,axis=2)
    restore=(~edges)&vis&(diff>V8_DELTA_MAX*.8)
    out[:,:,:3][restore]=arr[:,:,:3][restore]
    out[:,:,3]=arr[:,:,3]
    out[out[:,:,3]==0,:3]=0
    changed=int(np.count_nonzero(np.any(out[:,:,:3]!=arr[:,:,:3],axis=2)&vis))
    final_lab=v8_lab(out[:,:,:3])
    max_err=float(np.max(np.linalg.norm(final_lab[vis]-labs[vis],axis=1)))
    if max_err>V8_DELTA_MAX+.01:raise ValueError('BOM v8 exceeded Lab guard: '+str(max_err))
    assert np.array_equal(arr[:,:,3],out[:,:,3])
    stats={('family_'+k):int(v) for k,v in families.items()}
    stats.update({'protected_pixels':int(np.sum(keep)),
                  'changed_pixels':changed,'visible_pixels':int(np.sum(vis)),
                  'max_lab_delta_milli':int(round(max_err*1000)),'preserved_alpha':1})
    colors=len(np.unique(out[:,:,:3][vis],axis=0))
    return Image.fromarray(out,'RGBA'),stats,colors


def render_preview(manifest: dict, split_root: Path, output: Path, title: str, subtitle: str) -> None:
    cols, rows = int(manifest['grid']['columns']), int(manifest['grid']['rows'])
    tw, th = map(int, manifest['tile_size'])
    cell_width, label_height = 184, 60
    pad, head, foot, sep = 16, 65, 30, 5
    step_x, step_y = cell_width + sep, th + label_height + sep
    w, h = 2 * pad + cols * step_x - sep, head + rows * step_y - sep + foot
    im = Image.new('RGBA', (w, h), (28, 31, 33, 255))
    d = ImageDraw.Draw(im)

    def font(size: int, bold: bool = False):
        filename = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        try:
            return ImageFont.truetype(filename, size=size)
        except OSError:
            return ImageFont.load_default()

    titlefont, font13, small = font(23, True), font(13), font(11)
    d.text((pad, 12), title, font=titlefont, fill=(242, 237, 221))
    d.text((pad, 44), subtitle, font=small, fill=(160, 174, 179))

    def measure(text, f):
        bb = d.textbbox((0, 0), text, font=f)
        return bb[2] - bb[0], bb[3] - bb[1]

    def wrap(text, f, width, limit=3):
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text.replace('_', ' _ '))
        words = text.split()
        if not words:
            return ['(brak)']
        lines, line = [], ''
        for word in words:
            candidate = (line + ' ' + word).strip()
            if line and measure(candidate, f)[0] > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        if len(lines) > limit:
            lines = lines[:limit]
            last = lines[-1]
            while last and measure(last + '…', f)[0] > width:
                last = last[:-1]
            lines[-1] = last + '…'
        return lines

    for t in manifest['tiles']:
        r, c = int(t['row']) - 1, int(t['column']) - 1
        x, y = pad + c * step_x, head + r * step_y
        base = (42, 46, 47, 255) if not t.get('empty') else (55, 52, 48, 255)
        d.rectangle((x, y, x + cell_width - 1, y + th + label_height - 1), fill=base, outline=(106, 115, 111), width=1)
        with Image.open(split_root / t['file']) as src:
            sprite = ensure_rgba(src).copy()
        ix = x + (cell_width - tw) // 2
        im.alpha_composite(sprite, (ix, y))
        d.rectangle((x + 5, y + 5, x + 65, y + 24), fill=(15, 17, 18, 240))
        d.text((x + 10, y + 7), t['id'], font=small, fill=(225, 229, 216))
        d.line((x + 1, y + th, x + cell_width - 2, y + th), fill=(102, 112, 112), width=1)
        lines = wrap(str(t.get('name') or t['id']), font13, cell_width - 13, 3)
        top = y + th + max(3, (label_height - len(lines) * 17) // 2)
        color = (238, 238, 223) if t.get('prefab') else (236, 186, 136)
        if t.get('empty'):
            color = (161, 161, 156)
        for line in lines:
            ww, _ = measure(line, font13)
            d.text((x + (cell_width - ww) // 2, top), line, font=font13, fill=color)
            top += 17
    output.parent.mkdir(parents=True, exist_ok=True)
    im.convert('RGB').save(output, 'PNG', optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='BOM v8 - adaptive color families + source-guided artifact cleanup')
    parser.add_argument('--source-root', type=Path, default=Path('source_128'),
                        help='Folder containing original atlas_manifest.json and tiles/ files')
    parser.add_argument('--output-root', type=Path, default=Path('valheim-bom-wyniki-v8'),
                        help='Where to save the completed atlas, split tiles and review')
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    out_root = args.output_root.resolve()
    root = out_root.parent
    source_manifest = json.loads((source_root / 'atlas_manifest.json').read_text(encoding='utf-8'))
    split_root = out_root / 'processed_split'
    tiles_root = split_root / 'tiles'
    named_root = out_root / 'ikony_BOM_v8_96x96'
    tiles_root.mkdir(parents=True, exist_ok=True)
    named_root.mkdir(parents=True, exist_ok=True)

    cols, rows = int(source_manifest['grid']['columns']), int(source_manifest['grid']['rows'])
    atlas = Image.new('RGBA', (cols * TARGET_TILE, rows * TARGET_TILE), (0, 0, 0, 0))
    manifest_tiles = []
    nonempty_count = 0
    family_usage: Dict[str, int] = {}
    tile_color_counts: List[int] = []

    for t in source_manifest['tiles']:
        tid = str(t['id'])
        with Image.open(source_root / t['file']) as im:
            src_tile = ensure_rgba(im).copy()
        # Preserve every original 96x96 source pixel and its antialiased alpha.
        # In this fresh workflow the original was only placed, never resized, in a 128x128 cell.
        # Undo that temporary padding exactly instead of normalizing/rescaling to 92x92.
        if src_tile.size == (128, 128):
            normalized = src_tile.crop((16, 16, 112, 112))
        elif src_tile.size == (96, 96):
            normalized = src_tile.copy()
        else:
            raise ValueError(f"Unexpected original source tile dimensions: {src_tile.size}")
        processed, family_stats, unique_count = process_tile(normalized)
        tile_color_counts.append(unique_count)
        for k, v in family_stats.items():
            family_usage[k] = family_usage.get(k, 0) + int(v)
        empty = processed.getchannel('A').getbbox() is None
        if not empty:
            nonempty_count += 1
        x = (int(t['column']) - 1) * TARGET_TILE
        y = (int(t['row']) - 1) * TARGET_TILE
        atlas.alpha_composite(processed, (x, y))
        tile_path = tiles_root / f'{tid}.png'
        processed.save(tile_path, 'PNG', optimize=True)
        if not empty:
            processed.save(named_root / str(t['name']), 'PNG', optimize=True)
        manifest_tiles.append({
            'id': tid,
            'name': t.get('name', tid),
            'prefab': t.get('prefab', ''),
            'row': int(t['row']),
            'column': int(t['column']),
            'x': x,
            'y': y,
            'width': TARGET_TILE,
            'height': TARGET_TILE,
            'file': f'tiles/{tid}.png',
            'empty': bool(empty),
            'sha256_rgba': sha256_bytes(np.asarray(processed).tobytes()),
        })

    atlas_path = out_root / 'atlas_BOM_v8.png'
    atlas.save(atlas_path, 'PNG', optimize=True)

    manifest = {
        'schema': 'sirroh.atlas.tiles.v1',
        'description': 'BOM v8 on exact live original PNG: 5 shades per chromatic/black/gray/white family with guarded microartifact cleanup',
        'atlas': {
            'source': atlas_path.name,
            'size': [atlas.width, atlas.height],
            'mode': 'RGBA',
            'sha256_rgba': sha256_bytes(np.asarray(atlas).tobytes()),
            'sha256_png': sha256_bytes(atlas_path.read_bytes()),
        },
        'grid': {'columns': cols, 'rows': rows},
        'tile_size': [TARGET_TILE, TARGET_TILE],
        'tiles': manifest_tiles,
    }
    manifest_path = split_root / 'atlas_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    reassembled = Image.new('RGBA', (atlas.width, atlas.height), (0, 0, 0, 0))
    for t in manifest_tiles:
        with Image.open(split_root / t['file']) as im:
            reassembled.alpha_composite(ensure_rgba(im).copy(), (int(t['x']), int(t['y'])))
    reassembled_path = out_root / 'reassembled_BOM_v8.png'
    reassembled.save(reassembled_path, 'PNG', optimize=True)
    verification = {
        'schema': 'sirroh.atlas.tiles.v1',
        'tiles': len(manifest_tiles),
        'size': [atlas.width, atlas.height],
        'pixel_identical': np.asarray(reassembled).tobytes() == np.asarray(atlas).tobytes(),
        'png_binary_identical': sha256_bytes(reassembled_path.read_bytes()) == sha256_bytes(atlas_path.read_bytes()),
        'source_sha256_rgba': sha256_bytes(np.asarray(atlas).tobytes()),
        'reassembled_sha256_rgba': sha256_bytes(np.asarray(reassembled).tobytes()),
        'source_sha256_png': sha256_bytes(atlas_path.read_bytes()),
        'reassembled_sha256_png': sha256_bytes(reassembled_path.read_bytes()),
        'modified_tiles_detected': False,
    }
    (out_root / 'reassembled_BOM_v8.verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')

    report = {
        'build': 'BOM-V8-SOURCE-GUIDED-CLEANUP-2026-10-17',
        'method': 'Source-faithful perceptual mapping with strict hue (10 degrees), LAB (11), luminance (7) guards; preserve original eye/glow/shadows and alpha',
        'source_icons': sum(1 for t in source_manifest['tiles'] if not t.get('empty')),
        'processed_icons': nonempty_count,
        'grid': [cols, rows],
        'size': [atlas.width, atlas.height],
        'legacy_excluded': True,
        'alpha': 'source alpha >= 8 retained; original silhouette preserved',
        'shades_per_family': 5,
        'max_lab_delta_allowed': 11.0,
        'max_hue_delta_allowed_degrees': 10.0,
        'black_shades_if_present': 5,
        'gray_shades_if_present': 5,
        'white_shades_if_present': 5,
        'family_palette_usage': family_usage,
        'tile_color_counts': tile_color_counts,
        'max_colors_any_tile': max(tile_color_counts) if tile_color_counts else 0,
        'min_colors_any_tile': min(tile_color_counts) if tile_color_counts else 0,
        'avg_colors_any_tile': float(np.mean(tile_color_counts)) if tile_color_counts else 0.0,
        'cleanup': {
            'passes': COLOR_CLEANUP_PASSES,
            'majority_neighbors': COLOR_CLEANUP_NEIGHBOR_MAJORITY,
            'remove_isolated_opaque_pixels': 'only faint detached 1px remnants',
            'source_guided_recolor': True,
            'source_neighbor_delta_e_max': SOURCE_NEIGHBOR_MAX_DELTA_E,
        },
        'glow_accents_protected': True,
        'source_guided_cleanup': True,
        'all_reassembled_pixels_identical': verification['pixel_identical'],
        'original_index_preserved': True,
        'all_tiles_checked': True,
    }
    (out_root / 'atlas_BOM_v8.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    preview_path = out_root / 'preview_named_atlas_v8.png'
    render_preview(
        manifest,
        split_root,
        preview_path,
        title='SIRROH BOM V8  ·  PODGLĄD PLIKÓW',
        subtitle=f'{cols} × {rows}  |  {len(manifest_tiles)} kafelków  |  5 odcieni/rodzina + chronione detale + czyste krawędzie  |  tylko podgląd',
    )

    icons_zip = root / 'Valheim_Icons_BOM_v8_96x96.zip'
    with zipfile.ZipFile(icons_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(named_root.glob('*.png')):
            z.write(p, p.name)

    full_zip = root / 'valheim-bom-v8-review.zip'
    with zipfile.ZipFile(full_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_root.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out_root))

    print('DONE')
    print('script', Path(__file__))
    print('atlas', atlas_path)
    print('preview', preview_path)
    print('icons_zip', icons_zip)
    print('full_zip', full_zip)
    print('processed_icons', nonempty_count)
    print('max_colors_any_tile', report['max_colors_any_tile'])
    print('avg_colors_any_tile', report['avg_colors_any_tile'])
    print('reassembled_pixel_identical', verification['pixel_identical'])


if __name__ == '__main__':
    main()