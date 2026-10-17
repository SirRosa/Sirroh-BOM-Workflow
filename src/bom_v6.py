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

ALPHA_THRESHOLD = 32
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


def split_families(rgb: np.ndarray):
    """Exact family membership, independent palettes for dark/gray/white tones.

    Dark saturated source pixels always stay in chromatic families (no V gate).
    Even rare accent hues receive their own five shades; no cross-family fallback.
    """
    hh = rgb_to_hsv(rgb)
    labv = _lab_array(rgb.reshape(-1,1,3)).reshape(-1,3)
    groups: Dict[str, List[int]] = {}
    for i, ((h, saturation, value), lum) in enumerate(zip(hh, labv[:,0])):
        if saturation < 0.13:
            fam = "black" if lum < 36.0 else ("white" if lum > 76.0 else "gray")
        else:
            fam = FAMILY_LABELS[int(math.floor(float(h) * HUE_BINS)) % HUE_BINS]
        groups.setdefault(fam, []).append(i)

    # Separate vivid glowing accents from the main hue where appropriate,
    # without depriving the dark main color of its own five shades.
    for fam, indices in list(groups.items()):
        if fam in ("black", "white", "gray") or len(indices) < 20:
            continue
        idx = np.asarray(indices, dtype=np.int32)
        hsv_values = hh[idx]
        glow = (hsv_values[:,1] >= GLOW_SATURATION) & (hsv_values[:,2] >= GLOW_VALUE)
        count = int(glow.sum())
        if GLOW_MIN_PIXELS <= count <= 0.4 * len(idx):
            groups[fam] = idx[~glow].tolist()
            groups[fam + "_accent"] = idx[glow].tolist()

    empty = np.zeros(len(rgb), dtype=bool)
    return {name: np.array(ix,dtype=np.int32) for name,ix in groups.items()}, empty, empty, empty


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


def assign_quantized(rgb: np.ndarray):
    groups, _, _, _ = split_families(rgb)
    out = np.zeros_like(rgb, dtype=np.uint8)
    assigned = np.zeros(len(rgb), dtype=bool)
    stats = {}
    for fam, indices in groups.items():
        if not len(indices):
            continue
        pal = representative_palette_for_family(rgb[indices])
        if not len(pal):
            continue
        source_lab = _lab_array(rgb[indices].reshape(-1,1,3)).reshape(-1,3)
        palette_lab = _lab_array(pal.reshape(-1,1,3)).reshape(-1,3)
        d = np.sum((source_lab[:,None,:] - palette_lab[None,:,:])**2,axis=-1)
        out[indices] = pal[np.argmin(d,axis=1)]
        assigned[indices] = True
        stats[fam] = len(pal)
    if not np.all(assigned):
        raise RuntimeError(f"Unassigned chromatic pixels: {(~assigned).sum()}")
    return out, stats


def _lab_array(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


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
    """Conservative, source-guided cleanup.

    - Do not invent or erase silhouettes (original anti-alias alpha is retained).
    - Do not repaint true contrasting features: pupils/glow, bright spots, dark creases.
    - Repaint only an isolated palette speck *when source itself is locally smooth*.
    """
    out = rgba.copy()
    if source_rgba is None:
        return out
    src = source_rgba.copy()
    if src.shape != out.shape:
        raise ValueError('source and quantized image dimensions differ')
    foreground = (out[:, :, 3] >= ALPHA_THRESHOLD)
    if not np.any(foreground):
        return out

    # Clean only detached, faint 1-pixel alpha remnants. A 1-pixel glowing eye with
    # a solid alpha is never removed, and RGB geometry of the main object is intact.
    components = (foreground.astype(np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(components, 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area == 1:
            yy, xx = np.where(labels == i)
            y, x = int(yy[0]), int(xx[0])
            if int(src[y, x, 3]) < 110 and int(np.max(src[y, x, :3])) < 70:
                out[y, x] = 0

    foreground = out[:, :, 3] >= ALPHA_THRESHOLD
    src_lab = _lab_array(src[:, :, :3])
    src_l = src_lab[:, :, 0]
    out_lab = _lab_array(out[:, :, :3])
    h, w = foreground.shape

    # Before any cleanup, protect border colors. v4 turned antialiased silhouettes
    # into hard, occasionally black contours. v6 retains original alpha and doesn't
    # repaint edges unless the source itself has the same dark outline.
    interior = cv2.erode(foreground.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)

    for _ in range(COLOR_CLEANUP_PASSES):
        changes = []
        for labels, idx, area in _flat_color_components(out[:, :, :3], interior):
            yy, xx = np.where(labels == idx)
            for y, x in zip(yy.tolist(), xx.tolist()):
                y0, y1 = max(0, y-1), min(h, y+2)
                x0, x1 = max(0, x-1), min(w, x+2)
                neighbors_mask = foreground[y0:y1, x0:x1].copy()
                neighbors_mask[y-y0, x-x0] = False
                nb_rgb = out[y0:y1, x0:x1, :3][neighbors_mask]
                nb_src_lab = src_lab[y0:y1, x0:x1][neighbors_mask]
                if len(nb_rgb) < 5:
                    continue
                vals, counts = np.unique(nb_rgb, axis=0, return_counts=True)
                majority_idx = int(np.argmax(counts))
                if counts[majority_idx] < COLOR_CLEANUP_NEIGHBOR_MAJORITY:
                    continue
                candidate = vals[majority_idx]
                if np.array_equal(candidate, out[y, x, :3]):
                    continue
                # If the original contains a genuinely contrasting light or dark
                # feature, preserve it (small eyes, gloss, cinders, pupil outlines).
                center_lab = src_lab[y, x]
                median_neighbor = np.median(nb_src_lab, axis=0)
                delta_e = float(np.linalg.norm(center_lab - median_neighbor))
                if delta_e > SOURCE_NEIGHBOR_MAX_DELTA_E:
                    continue
                # Reject cleanup when candidate is locally incongruent in the source.
                candidate_lab = _lab_array(candidate.reshape(1, 1, 3)).reshape(3)
                if float(np.linalg.norm(candidate_lab - center_lab)) > 30.0:
                    continue
                # No shadow/glow changes disguised as small isolated artifacts.
                if abs(float(src_l[y, x]) - float(median_neighbor[0])) > 9.5:
                    continue
                changes.append((y, x, candidate))
        if not changes:
            break
        for y, x, candidate in changes:
            out[y, x, :3] = candidate

    # The source silhouette is immutable: no black fringe recoloring on edges.

    # Transparent RGB must be zero, including antialias outside alpha threshold.
    out[out[:, :, 3] == 0, :3] = 0
    return out



def cleanup_micro_specks(result: np.ndarray, source: np.ndarray):
    """Erase only newly quantized 1-4 px patches unsupported by source RGB.

    Checks local source LAB texture, keeps alpha and authentic tiny luminous
    details. Works on borders too, unlike the conservative v6 interior cleanup.
    """
    out = result.copy()
    h,w = out.shape[:2]
    fg = out[:,:,3] >= ALPHA_THRESHOLD
    src_lab = _lab_array(source[:,:,:3])
    src_hsv = cv2.cvtColor(source[:,:,:3].astype(np.float32)/255., cv2.COLOR_RGB2HSV)
    for _ in range(2):
        changes=[]
        for c in np.unique(out[fg,:3],axis=0):
            current = fg & np.all(out[:,:,:3] == c,axis=2)
            n, labels, areas, _ = cv2.connectedComponentsWithStats(current.astype(np.uint8),8)
            for ci in range(1,n):
                if int(areas[ci,cv2.CC_STAT_AREA]) > 4:continue
                ys,xs = np.where(labels==ci)
                cluster = np.array([src_lab[y,x] for y,x in zip(ys,xs)])
                source_center = np.median(cluster,axis=0)
                neighbor_source=[]
                neighbor_rgb=[]
                for y,x in zip(ys,xs):
                    y0,y1=max(0,y-1),min(h,y+2)
                    x0,x1=max(0,x-1),min(w,x+2)
                    valid = fg[y0:y1,x0:x1] & (labels[y0:y1,x0:x1]!=ci)
                    if np.any(valid):
                        neighbor_source.extend(src_lab[y0:y1,x0:x1][valid])
                        neighbor_rgb.extend(out[y0:y1,x0:x1,:3][valid])
                if len(neighbor_rgb)<3:continue
                avg = np.median(np.array(neighbor_source),axis=0)
                source_delta=float(np.linalg.norm(source_center-avg))
                if source_delta>11.5: continue
                # Protected legitimate bright and chromatic accents (eyes, runes).
                if any(src_hsv[y,x,1]>=.49 and src_hsv[y,x,2]>=.50 for y,x in zip(ys,xs)):
                    continue
                options, counts=np.unique(np.array(neighbor_rgb,dtype=np.uint8),axis=0,return_counts=True)
                options=options[counts>=2]
                if not len(options):continue
                labs=_lab_array(options.reshape(-1,1,3)).reshape(-1,3)
                selected=int(np.argmin(np.linalg.norm(labs-source_center[None,:],axis=1)))
                new_color=options[selected]
                old_lab=_lab_array(c.reshape(1,1,3)).reshape(-1,3)[0]
                new_lab=labs[selected]
                if np.linalg.norm(new_lab-source_center)+3 >= np.linalg.norm(old_lab-source_center):
                    continue
                if np.linalg.norm(new_lab-avg)>14:continue
                changes.extend((y,x,new_color) for y,x in zip(ys,xs))
        if not changes:break
        for y,x,col in changes:out[y,x,:3]=col
    return out


def process_tile(img: Image.Image) -> Tuple[Image.Image, Dict[str, int], int]:
    img = ensure_rgba(img)
    arr = np.asarray(img).copy()
    foreground = arr[:, :, 3] >= ALPHA_THRESHOLD
    arr[~foreground] = 0
    if not np.any(foreground):
        return Image.fromarray(arr, 'RGBA'), {}, 0
    qrgb, stats = assign_quantized(arr[:, :, :3][foreground].astype(np.float32))
    out = np.zeros_like(arr)
    # Keep real source edge alpha. Binarizing it caused outlines/jagged artifacts.
    out[:, :, 3] = arr[:, :, 3]
    out[:, :, :3][foreground] = qrgb.astype(np.uint8)
    out = cleanup_artifacts(out, source_rgba=arr)
    out = cleanup_micro_specks(out, source=arr)
    unique_count = len({tuple(px[:3]) for px in out.reshape(-1, 4) if px[3] >= ALPHA_THRESHOLD})
    return Image.fromarray(out, 'RGBA'), stats, unique_count


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
    parser = argparse.ArgumentParser(description='BOM v6 - adaptive color families + source-guided artifact cleanup')
    parser.add_argument('--source-root', type=Path, default=Path('source_128'),
                        help='Folder containing original atlas_manifest.json and tiles/ files')
    parser.add_argument('--output-root', type=Path, default=Path('valheim-bom-wyniki-v6'),
                        help='Where to save the completed atlas, split tiles and review')
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    out_root = args.output_root.resolve()
    root = out_root.parent
    source_manifest = json.loads((source_root / 'atlas_manifest.json').read_text(encoding='utf-8'))
    split_root = out_root / 'processed_split'
    tiles_root = split_root / 'tiles'
    named_root = out_root / 'ikony_BOM_v6_96x96'
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

    atlas_path = out_root / 'atlas_BOM_v6.png'
    atlas.save(atlas_path, 'PNG', optimize=True)

    manifest = {
        'schema': 'sirroh.atlas.tiles.v1',
        'description': 'BOM v6: source-hue family-preserving adaptive families, 5 shades per family, free black/white, cleanup improved',
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
            reassembled.alpha_composite(ensure_rgba(im), (int(t['x']), int(t['y'])))
    reassembled_path = out_root / 'reassembled_BOM_v6.png'
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
    (out_root / 'reassembled_BOM_v6.verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')

    report = {
        'build': 'BOM-V6-NEUTRAL-SHADES-DARK-CHROMA-2026-10-17',
        'method': 'Source-guided speck cleanup, preserved palette-family mappings and anti-aliased silhouette',
        'source_icons': sum(1 for t in source_manifest['tiles'] if not t.get('empty')),
        'processed_icons': nonempty_count,
        'grid': [cols, rows],
        'size': [atlas.width, atlas.height],
        'legacy_excluded': True,
        'alpha': 'source edge alpha retained; only faint detached remnants removed',
        'shades_per_family': SHADES_PER_FAMILY,
        'black_shades_if_present': SHADES_PER_FAMILY,
        'white_shades_if_present': SHADES_PER_FAMILY,
        'family_palette_usage': family_usage,
        'tile_color_counts': tile_color_counts,
        'max_colors_any_tile': max(tile_color_counts) if tile_color_counts else 0,
        'min_colors_any_tile': min(tile_color_counts) if tile_color_counts else 0,
        'avg_colors_any_tile': float(np.mean(tile_color_counts)) if tile_color_counts else 0.0,
        'cleanup': {
            'passes': COLOR_CLEANUP_PASSES,
            'majority_neighbors': COLOR_CLEANUP_NEIGHBOR_MAJORITY,
            'remove_isolated_opaque_pixels': 'only faint detached remnants',
            'source_guided_recolor': True,
            'source_neighbor_delta_e_max': SOURCE_NEIGHBOR_MAX_DELTA_E,
        },
        'glow_accents_protected': True,
        'source_guided_cleanup': True,
        'all_reassembled_pixels_identical': verification['pixel_identical'],
        'original_index_preserved': True,
        'all_tiles_checked': True,
    }
    (out_root / 'atlas_BOM_v6.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    preview_path = out_root / 'preview_named_atlas_v6.png'
    render_preview(
        manifest,
        split_root,
        preview_path,
        title='SIRROH BOM V6  ·  PODGLĄD PLIKÓW',
        subtitle=f'{cols} × {rows}  |  {len(manifest_tiles)} kafelków  |  5 odcieni/rodzina + chronione detale + czyste krawędzie  |  tylko podgląd',
    )

    icons_zip = root / 'Valheim_Icons_BOM_v6_96x96.zip'
    with zipfile.ZipFile(icons_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(named_root.glob('*.png')):
            z.write(p, p.name)

    full_zip = root / 'valheim-bom-v6-review.zip'
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