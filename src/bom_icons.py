#!/usr/bin/env python3
"""B&om Icons: atlas -> automatically detected NxM grid -> 64px tiles.

Preserves the essential BOM biome engine: premultiplied-alpha resampling,
black/transparent cleanup, 3x bilateralFilter(7,28,5), fixed-seed OpenCV
k-means palette of *actual* source colors, no dithering, mode cleanup,
and strict alpha 0/255. The palette is fitted independently for EACH tile.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BUILD_ID = "BOM-ICONS-2026-10-15-A"
TILE_SIZE = 64
FOREGROUND_ALPHA_THRESHOLD = 32
FOREGROUND_RGB_THRESHOLD = 18
MIN_COMPONENT_AREA = 3
FLOATING_ARTIFACT_MAX_AREA = 32
FLOATING_ARTIFACT_MAX_WIDTH = 16
FLOATING_ARTIFACT_MAX_HEIGHT = 16
BILATERAL_PASSES = 3
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 28
BILATERAL_SIGMA_SPACE = 5
BASE_COLORS = 4
CLEANUP_KERNEL = 5
CLEANUP_PASSES = 1
CLEANUP_THRESHOLD = 0.46


def resize_rgba_premultiplied(src: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """BOM resize: never mix invisible RGB into the visible edges."""
    if src.shape[1] == new_w and src.shape[0] == new_h:
        return src.copy()
    rgb = src[:, :, :3].astype(np.float32)
    alpha = src[:, :, 3].astype(np.float32) / 255.0
    premul = rgb * alpha[:, :, None]
    interp = cv2.INTER_AREA if new_w < src.shape[1] or new_h < src.shape[0] else cv2.INTER_CUBIC
    p = cv2.resize(premul, (new_w, new_h), interpolation=interp)
    a = np.clip(cv2.resize(alpha, (new_w, new_h), interpolation=interp), 0, 1)
    out = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    valid = a >= FOREGROUND_ALPHA_THRESHOLD / 255.0
    out[valid, :3] = np.clip(np.rint(p[valid] / a[valid, None]), 0, 255).astype(np.uint8)
    out[valid, 3] = np.clip(np.rint(a[valid] * 255), 0, 255).astype(np.uint8)
    return out


def _line_cost(proj: np.ndarray, nominal: float, cell: float) -> tuple[float, int]:
    """Find the transparent gutter near a predicted row/column division."""
    n = len(proj)
    half_window = max(2, int(round(cell * 0.19)))
    p = int(round(nominal))
    candidates = range(max(1, p - half_window), min(n - 1, p + half_window) + 1)
    mean = max(1.0, float(np.mean(proj)))
    best_val, best_index = float("inf"), p
    for ix in candidates:
        # Surrounding 5 pixels discourage cutting through anti-aliased objects.
        a, b = max(0, ix - 2), min(n, ix + 3)
        opacity = float(np.mean(proj[a:b])) / mean
        displacement = abs(ix - nominal) / cell
        # Displacement penalty prevents chasing unrelated empty strips.
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
    """Detect a square-cell grid from repeated gutters, never hard-code 8x9.

    Fails explicitly if gutters are ambiguous. Optional manual row/col override
    for irregular atlases, with gutters still detected around those positions.
    """
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


def foreground_mask(rgba: np.ndarray) -> np.ndarray:
    fg = (rgba[:, :, 3] >= FOREGROUND_ALPHA_THRESHOLD) & (rgba[:, :, :3].max(axis=2) > FOREGROUND_RGB_THRESHOLD)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    cleaned = np.zeros_like(fg)
    # Original BOM limits were intended for 128px sprites, so rescale areas to 64px.
    scale = TILE_SIZE / 128.0
    max_area = max(1, int(round(FLOATING_ARTIFACT_MAX_AREA * scale * scale)))
    max_side = max(1, int(round(FLOATING_ARTIFACT_MAX_WIDTH * scale)))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < max(1, round(MIN_COMPONENT_AREA * scale * scale)):
            continue
        if area <= max_area and cw <= max_side and ch <= max_side:
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


def extract_palette(rgb: np.ndarray, mask: np.ndarray, k: int, seed: int) -> np.ndarray:
    pixels = rgb[mask]
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    k = min(k, len(np.unique(pixels, axis=0)))
    sample = pixels
    if len(sample) > 20000:
        rng = np.random.default_rng(seed)
        sample = sample[rng.choice(len(sample), 20000, replace=False)]
    data = sample.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    sample_f = sample.astype(np.float32)
    palette = np.asarray([sample[int(np.sum((sample_f - center) ** 2, axis=1).argmin())] for center in centers], dtype=np.uint8)
    lum = palette @ np.array([0.2126, 0.7152, 0.0722])
    return palette[np.argsort(lum)]


def quantize(rgb: np.ndarray, mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    idx = np.full(mask.shape, -1, dtype=np.int8)
    if not len(palette) or not np.any(mask):
        return idx
    pts = rgb[mask].astype(np.int16)
    pal = palette.astype(np.int16)
    idx[mask] = np.argmin(((pts[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2), axis=1).astype(np.int8)
    return idx


def mode_cleanup(idx: np.ndarray, mask: np.ndarray, colors: int) -> np.ndarray:
    if not np.any(mask) or colors <= 0:
        return idx
    out = idx.copy()
    density = cv2.blur(mask.astype(np.float32), (CLEANUP_KERNEL, CLEANUP_KERNEL))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    for _ in range(CLEANUP_PASSES):
        scores = []
        for ci in range(colors):
            amount = cv2.blur(((out == ci) & mask).astype(np.float32), (CLEANUP_KERNEL, CLEANUP_KERNEL))
            scores.append(np.divide(amount, density, out=np.zeros_like(amount), where=density > 1e-6))
        scores = np.stack(scores)
        best = scores.argmax(axis=0).astype(np.int8)
        confidence = scores.max(axis=0)
        change = interior & (confidence >= CLEANUP_THRESHOLD) & (best != out)
        out[change] = best[change]
    return out


def process_tile(tile: np.ndarray, colors: int, seed: int) -> tuple[np.ndarray, int]:
    tile = tile.copy()
    near_invisible = tile[:, :, 3] < FOREGROUND_ALPHA_THRESHOLD
    tile[near_invisible] = 0
    rgba = resize_rgba_premultiplied(tile, TILE_SIZE, TILE_SIZE)
    fg = foreground_mask(rgba)
    smoothed = bilateral_denoise(rgba[:, :, :3].copy(), fg)
    palette = extract_palette(smoothed, fg, colors, seed)
    idx = mode_cleanup(quantize(smoothed, fg, palette), fg, len(palette))
    out = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    out[fg, 3] = 255
    for pi, color in enumerate(palette):
        out[fg & (idx == pi), :3] = color
    return out, int(len(palette))


def process(input_file: Path, output_file: Path, colors: int = BASE_COLORS, cols: int | None = None, rows: int | None = None) -> dict:
    original = np.asarray(Image.open(input_file).convert("RGBA"))
    col_count, row_count, xbounds, ybounds, detection = detect_grid(original, cols, rows)
    result = np.zeros((row_count * TILE_SIZE, col_count * TILE_SIZE, 4), dtype=np.uint8)
    counts = []
    for row in range(row_count):
        for col in range(col_count):
            x0, x1 = xbounds[col], xbounds[col + 1]
            y0, y1 = ybounds[row], ybounds[row + 1]
            tile, count = process_tile(original[y0:y1, x0:x1], colors, 100 + row * col_count + col)
            result[row * TILE_SIZE:(row + 1) * TILE_SIZE, col * TILE_SIZE:(col + 1) * TILE_SIZE] = tile
            counts.append(count)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, "RGBA").save(output_file, "PNG", optimize=True)
    if not set(np.unique(result[:, :, 3]).tolist()).issubset({0, 255}):
        raise RuntimeError("Unexpected non-binary alpha")
    report = {
        "build": BUILD_ID,
        "input": str(input_file), "output": str(output_file),
        "source_size": [int(original.shape[1]), int(original.shape[0])],
        "grid": [col_count, row_count], "tiles": row_count * col_count,
        "tile_size": TILE_SIZE, "output_size": [int(result.shape[1]), int(result.shape[0])],
        "max_colors_per_tile": colors, "tile_color_counts": counts,
        "cuts_x": xbounds, "cuts_y": ybounds, "detection": detection,
        "alpha": [0, 255],
    }
    report_file = output_file.parent / (output_file.stem + ".json")
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"B&om OK: {input_file.name} | {col_count}×{row_count} = {col_count*row_count} tiles | {result.shape[1]}×{result.shape[0]} px | {colors} colors/tile")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="B&om: automatically find icon grid and convert to 64×64 tiles")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--input", type=Path, help="Process one image instead of all input PNGs")
    parser.add_argument("--output", type=Path, help="Custom output PNG path (requires --input)")
    parser.add_argument("--cols", type=int, help="Manual column count for unusual atlases")
    parser.add_argument("--rows", type=int, help="Manual row count for unusual atlases")
    parser.add_argument("--colors", type=int, default=BASE_COLORS, help="Colors PER tile (default: 4, similar to BOM)")
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
