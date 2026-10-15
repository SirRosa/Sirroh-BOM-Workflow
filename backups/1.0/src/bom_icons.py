#!/usr/bin/env python3
"""B&om Icons: detect source grid, normalize icons per slot, then apply BOM cleanup.

Key difference vs the earlier draft:
- grid is still detected automatically,
- BUT icons are extracted as semantic objects and assigned to source slots,
- then each icon is fit proportionally inside a clean output tile,
- then BOM palette reduction is applied per output tile.

This preserves objects that spill outside their nominal source cells (for example caves)
while still producing a strict aligned output atlas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BUILD_ID = "BOM-ICONS-2026-10-15-G"
TILE_SIZE = 96
LAYOUT_MARGIN = 2
FOREGROUND_ALPHA_THRESHOLD = 32
FOREGROUND_RGB_THRESHOLD = 18
MIN_COMPONENT_AREA = 3
SEMANTIC_COMPONENT_MIN_AREA = 40
FLOATING_ARTIFACT_MAX_AREA = 32
FLOATING_ARTIFACT_MAX_WIDTH = 16
FLOATING_ARTIFACT_MAX_HEIGHT = 16
BILATERAL_PASSES = 3
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 28
BILATERAL_SIGMA_SPACE = 5
BASE_COLORS = 6
CLEANUP_KERNEL = 5
CLEANUP_PASSES = 1
CLEANUP_THRESHOLD = 0.46


def resize_rgba_premultiplied(src: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if src.shape[1] == new_w and src.shape[0] == new_h:
        return src.copy()
    rgb = src[:, :, :3].astype(np.float32)
    alpha = src[:, :, 3].astype(np.float32) / 255.0
    premul = rgb * alpha[:, :, None]
    interp = cv2.INTER_AREA if new_w < src.shape[1] or new_h < src.shape[0] else cv2.INTER_CUBIC
    p = cv2.resize(premul, (new_w, new_h), interpolation=interp)
    a = np.clip(cv2.resize(alpha, (new_w, new_h), interpolation=interp), 0, 1)
    out = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    valid = a > 1e-6
    out[valid, :3] = np.clip(np.rint(p[valid] / a[valid, None]), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    out[~valid, :3] = 0
    out[out[:, :, 3] < FOREGROUND_ALPHA_THRESHOLD] = 0
    return out


def _line_cost(proj: np.ndarray, nominal: float, cell: float) -> tuple[float, int]:
    n = len(proj)
    half_window = max(2, int(round(cell * 0.19)))
    p = int(round(nominal))
    candidates = range(max(1, p - half_window), min(n - 1, p + half_window) + 1)
    mean = max(1.0, float(np.mean(proj)))
    best_val, best_index = float("inf"), p
    for ix in candidates:
        a, b = max(0, ix - 2), min(n, ix + 3)
        opacity = float(np.mean(proj[a:b])) / mean
        displacement = abs(ix - nominal) / cell
        value = opacity + 0.20 * displacement
        if value < best_val:
            best_val, best_index = value, ix
    return best_val, best_index


def _cuts(proj: np.ndarray, count: int) -> tuple[float, list[int]]:
    length = len(proj)
    cell = length / count
    values = [_line_cost(proj, i * cell, cell) for i in range(1, count)]
    if not values:
        return float("inf"), []
    positions = [int(x[1]) for x in values]
    if any(b <= a for a, b in zip(positions, positions[1:])):
        return float("inf"), []
    return float(np.mean([x[0] for x in values])), positions


def detect_grid(src: np.ndarray, cols: int | None = None, rows: int | None = None) -> tuple[int, int, list[int], list[int], dict]:
    h, w = src.shape[:2]
    occupied = src[:, :, 3] >= FOREGROUND_ALPHA_THRESHOLD
    proj_x = np.count_nonzero(occupied, axis=0).astype(np.float32)
    proj_y = np.count_nonzero(occupied, axis=1).astype(np.float32)
    max_cols = min(32, max(2, w // 40))
    max_rows = min(32, max(2, h // 40))
    cs = [cols] if cols else range(2, max_cols + 1)
    rs = [rows] if rows else range(2, max_rows + 1)
    options = []
    for c in cs:
        if c is None or c < 1 or c > w // 16:
            raise ValueError(f"Invalid column count: {c}")
        for r in rs:
            if r is None or r < 1 or r > h // 16:
                raise ValueError(f"Invalid row count: {r}")
            aspect = (w / c) / (h / r)
            if (cols is None or rows is None) and not 0.88 <= aspect <= 1.12:
                continue
            sx, xcut = _cuts(proj_x, c)
            sy, ycut = _cuts(proj_y, r)
            if not np.isfinite(sx + sy):
                continue
            score = (sx + sy) / 2 + 0.15 * abs(np.log(aspect))
            options.append((score, c, r, sx, sy, xcut, ycut))
    if not options:
        raise ValueError("Could not find a square-cell grid; specify --cols and --rows")
    score, c, r, sx, sy, xc, yc = min(options, key=lambda a: (a[0], -a[1] * a[2]))
    if (cols is None or rows is None) and (sx >= 0.58 or sy >= 0.58):
        raise ValueError(f"Grid gutters ambiguous (best {c}x{r}, x={sx:.2f}, y={sy:.2f}). Use --cols/--rows")
    return c, r, [0, *xc, w], [0, *yc, h], {"score": round(score, 4), "x_gutter": round(sx, 4), "y_gutter": round(sy, 4)}


def clean_foreground_mask(rgba: np.ndarray) -> np.ndarray:
    fg = (rgba[:, :, 3] >= FOREGROUND_ALPHA_THRESHOLD) & (rgba[:, :, :3].max(axis=2) > FOREGROUND_RGB_THRESHOLD)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < MIN_COMPONENT_AREA:
            continue
        if area <= FLOATING_ARTIFACT_MAX_AREA and w <= FLOATING_ARTIFACT_MAX_WIDTH and h <= FLOATING_ARTIFACT_MAX_HEIGHT:
            continue
        cleaned[labels == i] = True
    return cleaned


def bilateral_denoise(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[~fg] = 0
    for _ in range(BILATERAL_PASSES):
        out = cv2.bilateralFilter(out, d=BILATERAL_D, sigmaColor=BILATERAL_SIGMA_COLOR, sigmaSpace=BILATERAL_SIGMA_SPACE)
    out[~fg] = 0
    return out


def hue_family(rgb: np.ndarray) -> np.ndarray:
    """Classify noticeable chromatic families without assuming which sprite it is.

    0: neutral, 1: reds/oranges/yellows, 2: greens,
    3: cyan/blues, 4: violet/magenta.
    Values are derived from original RGB, before smoothing/quantization.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    if not rgb.size:
        return np.zeros(rgb.shape[:-1], dtype=np.uint8)
    shape = rgb.shape[:-1]
    flat = rgb.reshape(-1, 1, 3)
    hsv = cv2.cvtColor(flat, cv2.COLOR_RGB2HSV).reshape(*shape, 3)
    h, s, v = [hsv[..., i].astype(np.int16) for i in range(3)]
    visible_chroma = (s >= 58) & (v >= 36)
    cls = np.zeros(shape, dtype=np.uint8)
    cls[visible_chroma & ((h <= 35) | (h >= 170))] = 1
    cls[visible_chroma & (h > 35) & (h < 92)] = 2
    cls[visible_chroma & (h >= 92) & (h < 145)] = 3
    cls[visible_chroma & (h >= 145) & (h < 170)] = 4
    # Yellow highlight/ochre remains warm even if HSV places it on the
    # near-green boundary. This rule relies on source color, not row/column.
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    # Amber highlights stay warm, but olive leaf-edge pixels must not be
    # forced to orange merely because a little red exceeds green.
    warm_strong = (red >= green * 1.45) & (red >= blue * 1.22)
    warm_bright_yellow = (red >= 180) & (red >= green * 1.10) & (red >= blue * 1.30)
    weak_olive = (cls == 1) & ~(warm_strong | warm_bright_yellow)
    cls[weak_olive] = 0
    cls[visible_chroma & (warm_strong | warm_bright_yellow)] = 1
    return cls


def _cluster_palette(sample: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Find actual original colors that best represent one perceptual family."""
    if len(sample) == 0 or k == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    k = min(k, len(np.unique(sample, axis=0)))
    # Budget memory and runtime while giving all distinct material accents a vote.
    if len(sample) > 15000:
        rng = np.random.default_rng(seed)
        sample = sample[rng.choice(len(sample), 15000, replace=False)]
    lab = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.15)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(lab, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    result = np.asarray([sample[int(np.sum((lab - center) ** 2, axis=1).argmin())] for center in centers], dtype=np.uint8)
    return result


def extract_palette(rgb: np.ndarray, mask: np.ndarray, k: int, seed: int, source_rgb: np.ndarray | None = None) -> np.ndarray:
    """Allocate the tile's limited colors among genuine source color families.

    A global k-means may overlook a minority hue (e.g. small red highlights or
    leaves). Fitting each chromatic family separately ensures that warm fruit
    and green leaves have distinct colors in a 6-entry palette.
    """
    pixels = rgb[mask]
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    k = min(k, len(np.unique(pixels, axis=0)))
    if k == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    original = rgb if source_rgb is None else source_rgb
    source_families = hue_family(original[mask])
    counts = np.bincount(source_families, minlength=5)
    # Reserve representatives for materially present colors; do not allocate
    # an entire palette slot to a single anti-aliased edge pixel.
    threshold = max(5, int(np.ceil(len(pixels) * 0.03)))
    present = [i for i in range(5) if counts[i] >= threshold]
    if not present:
        present = [int(np.argmax(counts))]
    if len(present) > k:
        present = sorted(present, key=lambda i: -counts[i])[:k]
    allocations = {g: 1 for g in present}
    # Largest-remainder apportionment by pixel population with moderate bias
    # toward minor accent families: fewer greens than orange berries, but not 0.
    weight = {g: float(counts[g]) for g in present}
    for _ in range(k - len(present)):
        g = max(present, key=lambda x: (weight[x] / (allocations[x] + 1.0), -x))
        allocations[g] += 1
    result = []
    for g in present:
        group_pixels = original[mask][source_families == g]
        if len(group_pixels) < allocations[g]:
            continue
        group_k = allocations[g]
        chosen = []
        fitting_pixels = group_pixels
        if group_k >= 3 and len(group_pixels) >= 80:
            group_lab = cv2.cvtColor(group_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            brightness = group_lab[:, 0].astype(np.int32)
            p50 = float(np.percentile(brightness, 50))
            p95 = float(np.percentile(brightness, 95))
            # Reserve one true bright source color if small bright highlights
            # would otherwise be lost inside a dominant middle-tone cluster.
            if p95 - p50 >= 15:
                bright = group_pixels[brightness >= np.percentile(brightness, 95)]
                bright_lab = cv2.cvtColor(bright.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
                mean = np.mean(bright_lab, axis=0)
                chosen.append(bright[int(np.sum((bright_lab - mean) ** 2, axis=1).argmin())])
                fitting_pixels = group_pixels[brightness < np.percentile(brightness, 95)]
                group_k -= 1
        for entry in _cluster_palette(fitting_pixels, group_k, seed + 97 * g):
            result.append(entry)
        result.extend(chosen)
    palette = np.asarray(result, dtype=np.uint8).reshape(-1, 3)
    lum = palette @ np.array([0.2126, 0.7152, 0.0722])
    return palette[np.argsort(lum)]


def _lab_distance_to_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Squared perceptual distances (int32; no overflow or modulo errors)."""
    shape = rgb.shape[:-1]
    pts = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.int32)
    pal = cv2.cvtColor(palette.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.int32)
    return ((pts[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2).reshape(*shape, len(palette))


def allowed_palette_colors(source_rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Prevent cross-hue substitutions *without* forced extreme lightness shifts.

    Dark brown and ochre can be classified as warm despite being substantially
    closer to a neutral/dark wood color than to a bright orange/beige accent.
    Only force a hue family if it contains a reasonably faithful source color.
    A close neutral alternative is permitted when the only same-family color
    would introduce an obvious glowing or white speckle on dark material.
    """
    distances = _lab_distance_to_palette(source_rgb, palette)
    src_classes = hue_family(source_rgb)
    pal_classes = hue_family(palette)
    allowed = np.ones(distances.shape, dtype=bool)
    neutral = pal_classes == 0
    if np.any(neutral):
        nearest_neutral = np.min(distances[..., neutral], axis=-1)
    else:
        nearest_neutral = np.full(src_classes.shape, np.iinfo(np.int32).max // 8)
    for cls in range(1, 5):
        matching = pal_classes == cls
        members = src_classes == cls
        if not np.any(matching) or not np.any(members):
            continue
        nearest_same = np.min(distances[..., matching], axis=-1)
        # Only prefer a neutral color when it is dramatically more faithful.
        # For vividly colored fruit or foliage, retain strict hue separation.
        neutral_escape = (nearest_neutral * 3 + 700 < nearest_same) & (nearest_neutral < 2800)
        restricted = members & ~neutral_escape
        allowed[restricted, :] &= matching
        escape = members & neutral_escape
        if np.any(escape):
            allowed[escape, :] &= (matching | neutral)
    return allowed


def quantize(rgb: np.ndarray, mask: np.ndarray, palette: np.ndarray, source_rgb: np.ndarray | None = None) -> np.ndarray:
    """Perceptual nearest-neighbor assignment with source-faithful hue guards."""
    idx = np.full(mask.shape, -1, dtype=np.int16)
    if not len(palette) or not np.any(mask):
        return idx
    source = rgb if source_rgb is None else source_rgb
    distances = _lab_distance_to_palette(rgb, palette)
    compatible = allowed_palette_colors(source, palette)
    distances[~compatible] = np.iinfo(np.int32).max // 8
    idx[mask] = np.argmin(distances[mask], axis=1).astype(np.int16)
    return idx


def mode_cleanup(idx: np.ndarray, mask: np.ndarray, colors: int,
                 source_rgb: np.ndarray | None = None, palette: np.ndarray | None = None) -> np.ndarray:
    if not np.any(mask) or colors <= 0:
        return idx
    out = idx.copy()
    density = cv2.blur(mask.astype(np.float32), (CLEANUP_KERNEL, CLEANUP_KERNEL))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if source_rgb is not None and palette is not None:
        permitted = allowed_palette_colors(source_rgb, palette)
        distances = _lab_distance_to_palette(source_rgb, palette)
    else:
        permitted = None
        distances = None
    for _ in range(CLEANUP_PASSES):
        scores = []
        for ci in range(colors):
            amount = cv2.blur(((out == ci) & mask).astype(np.float32), (CLEANUP_KERNEL, CLEANUP_KERNEL))
            scores.append(np.divide(amount, density, out=np.zeros_like(amount), where=density > 1e-6))
        scores = np.stack(scores, axis=-1)
        if permitted is not None:
            scores[~permitted] = -1.0
        best = scores.argmax(axis=2).astype(np.int16)
        confidence = scores.max(axis=2)
        change = interior & (confidence >= CLEANUP_THRESHOLD) & (best != out)
        if distances is not None:
            proposed = np.take_along_axis(distances, best[..., None], axis=2)[..., 0]
            current = np.take_along_axis(distances, np.maximum(out, 0)[..., None], axis=2)[..., 0]
            # A smoothing pass cannot invent a bright speck on a dark material.
            change &= proposed <= (current * 1.5 + 550)
        out[change] = best[change]
    return out


def process_tile(tile: np.ndarray, colors: int, seed: int) -> tuple[np.ndarray, int]:
    tile = tile.copy()
    tile[tile[:, :, 3] < FOREGROUND_ALPHA_THRESHOLD] = 0
    fg = clean_foreground_mask(tile)
    smoothed = bilateral_denoise(tile[:, :, :3].copy(), fg)
    palette = extract_palette(smoothed, fg, colors, seed, source_rgb=tile[:, :, :3])
    idx = mode_cleanup(quantize(smoothed, fg, palette, source_rgb=tile[:, :, :3]), fg,
                       len(palette), source_rgb=tile[:, :, :3], palette=palette)
    out = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    out[fg, 3] = 255
    for pi, color in enumerate(palette):
        out[fg & (idx == pi), :3] = color
    return out, int(len(palette))


def locate_slot(value: float, bounds: list[int]) -> int:
    i = int(np.searchsorted(np.asarray(bounds, dtype=np.int32), value, side="right") - 1)
    return max(0, min(i, len(bounds) - 2))


def extract_objects_by_slot(original: np.ndarray, xbounds: list[int], ybounds: list[int], cols: int, rows: int):
    base = original.copy()
    base[base[:, :, 3] < FOREGROUND_ALPHA_THRESHOLD] = 0
    original_fg = clean_foreground_mask(base)
    semantic = cv2.morphologyEx(original_fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(semantic, 8)

    slot_parts: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {(r, c): [] for r in range(rows) for c in range(cols)}
    assigned = 0
    ignored = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < SEMANTIC_COMPONENT_MIN_AREA:
            ignored += 1
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp_sem = (labels[y:y+h, x:x+w] == i)
        expanded = cv2.dilate(comp_sem.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        local_mask = original_fg[y:y+h, x:x+w] & expanded
        if not np.any(local_mask):
            local_mask = original_fg[y:y+h, x:x+w] & comp_sem
        if not np.any(local_mask):
            ignored += 1
            continue
        cx, cy = centroids[i]
        col = locate_slot(float(cx), xbounds)
        row = locate_slot(float(cy), ybounds)
        slot_parts[(row, col)].append((x, y, local_mask))
        assigned += 1

    # Merge all assigned parts per slot into one cropped RGBA icon.
    slots: list[list[np.ndarray | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    nonempty = 0
    for row in range(rows):
        for col in range(cols):
            parts = slot_parts[(row, col)]
            if not parts:
                continue
            min_x = min(x for x, _, mask in parts for _ in [mask])
            min_y = min(y for _, y, mask in parts for _ in [mask])
            max_x = max(x + mask.shape[1] for x, _, mask in parts)
            max_y = max(y + mask.shape[0] for _, y, mask in parts)
            union = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
            for x, y, local_mask in parts:
                oy = y - min_y
                ox = x - min_x
                union[oy:oy + local_mask.shape[0], ox:ox + local_mask.shape[1]] |= local_mask
            crop = base[min_y:max_y, min_x:max_x].copy()
            crop[~union] = 0
            slots[row][col] = crop
            nonempty += 1

    stats_out = {
        "semantic_components_assigned": assigned,
        "semantic_components_ignored": ignored,
        "nonempty_slots": nonempty,
        "empty_slots": rows * cols - nonempty,
    }
    return slots, stats_out


def normalize_icon_to_tile(icon: np.ndarray | None) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    if icon is None or icon.size == 0:
        return tile
    mask = icon[:, :, 3] >= FOREGROUND_ALPHA_THRESHOLD
    if not np.any(mask):
        return tile
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sprite = icon[y0:y1, x0:x1].copy()
    local_mask = mask[y0:y1, x0:x1]
    sprite[~local_mask] = 0

    obj_h, obj_w = sprite.shape[:2]
    fit_w = TILE_SIZE - 2 * LAYOUT_MARGIN
    fit_h = TILE_SIZE - 2 * LAYOUT_MARGIN
    scale = min(1.0, fit_w / max(1, obj_w), fit_h / max(1, obj_h))
    new_w = max(1, min(fit_w, int(round(obj_w * scale))))
    new_h = max(1, min(fit_h, int(round(obj_h * scale))))

    sprite_r = resize_rgba_premultiplied(sprite, new_w, new_h)
    dx = (TILE_SIZE - new_w) // 2
    dy = TILE_SIZE - LAYOUT_MARGIN - new_h  # bottom anchored, exactly 2 px padding
    tile[dy:dy + new_h, dx:dx + new_w] = sprite_r
    return tile


def process(input_file: Path, output_file: Path, colors: int = BASE_COLORS, cols: int | None = None, rows: int | None = None) -> dict:
    original = np.asarray(Image.open(input_file).convert("RGBA"))
    col_count, row_count, xbounds, ybounds, detection = detect_grid(original, cols, rows)
    slots, slot_stats = extract_objects_by_slot(original, xbounds, ybounds, col_count, row_count)
    result = np.zeros((row_count * TILE_SIZE, col_count * TILE_SIZE, 4), dtype=np.uint8)
    counts = []

    for row in range(row_count):
        for col in range(col_count):
            normalized = normalize_icon_to_tile(slots[row][col])
            tile, count = process_tile(normalized, colors, 100 + row * col_count + col)
            result[row * TILE_SIZE:(row + 1) * TILE_SIZE, col * TILE_SIZE:(col + 1) * TILE_SIZE] = tile
            counts.append(count)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, "RGBA").save(output_file, "PNG", optimize=True)
    if not set(np.unique(result[:, :, 3]).tolist()).issubset({0, 255}):
        raise RuntimeError("Unexpected non-binary alpha")

    report = {
        "build": BUILD_ID,
        "input": str(input_file),
        "output": str(output_file),
        "source_size": [int(original.shape[1]), int(original.shape[0])],
        "grid": [col_count, row_count],
        "tiles": row_count * col_count,
        "tile_size": TILE_SIZE,
        "output_size": [int(result.shape[1]), int(result.shape[0])],
        "max_colors_per_tile": colors,
        "tile_color_counts": counts,
        "cuts_x": xbounds,
        "cuts_y": ybounds,
        "detection": detection,
        "slot_extraction": slot_stats,
        "alpha": [0, 255],
    }
    report_file = output_file.parent / (output_file.stem + ".json")
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"B&om OK: {input_file.name} | {col_count}×{row_count} = {col_count * row_count} tiles | "
        f"{result.shape[1]}×{result.shape[0]} px | {colors} colors/tile | nonempty {slot_stats['nonempty_slots']}"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="B&om: detect icon grid, normalize icons into aligned tiles, and reduce colors per tile")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--input", type=Path, help="Process one image instead of all input PNGs")
    parser.add_argument("--output", type=Path, help="Custom output PNG path (requires --input)")
    parser.add_argument("--cols", type=int, help="Manual column count for unusual atlases")
    parser.add_argument("--rows", type=int, help="Manual row count for unusual atlases")
    parser.add_argument("--colors", type=int, default=BASE_COLORS, help=f"Colors per tile (default: {BASE_COLORS})")
    args = parser.parse_args()

    if args.colors < 1 or args.colors > 16:
        parser.error("--colors must be between 1 and 16")
    if args.output and not args.input:
        parser.error("--output requires --input")

    sources = [args.input] if args.input else sorted(args.input_dir.glob("*.png"))
    if not sources:
        print(f"No input PNGs in {args.input_dir}", file=sys.stderr)
        return 1

    failed = 0
    for src in sources:
        dest = args.output or args.output_dir / (src.stem + "_BOM.png")
        try:
            process(src, dest, args.colors, args.cols, args.rows)
        except Exception as exc:
            failed += 1
            print(f"B&om ERROR {src}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
