#!/usr/bin/env python3
"""IbBit v1 — three-tone-per-color-family stylization for transparent game sprites.

Non-generative, deterministic, 256x256-first processor. Requires Pillow, numpy, cv2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 256
FAMILY_NAMES = ("red", "orange", "yellow", "olive", "green", "teal", "cyan",
                "blue", "violet", "magenta", "brown", "neutral")


def fit_256(img: Image.Image, size: int = SIZE) -> Image.Image:
    """Resize BEFORE processing. Preserve complete sprite and premultiplied-alpha edges."""
    src = img.convert("RGBA")
    if src.size == (size, size):
        return src.copy()
    ratio = min(size / src.width, size / src.height)
    wh = tuple(max(1, int(round(dim * ratio))) for dim in src.size)
    a = np.asarray(src, dtype=np.float32)
    alpha = a[:, :, 3:4] / 255.0
    premul = a[:, :, :3] * alpha
    rgb = cv2.resize(premul, wh, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    alp = cv2.resize(a[:, :, 3], wh, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    if rgb.ndim == 2:
        rgb = rgb[:, :, None]
    if alp.ndim == 2:
        alp = alp[:, :, None]
    else:
        alp = alp.reshape((*alp.shape[:2], 1))
    o = np.zeros((wh[1], wh[0], 4), dtype=np.uint8)
    o[:, :, :3] = np.clip(np.rint(rgb / np.maximum(alp / 255.0, 1e-5)), 0, 255).astype(np.uint8)
    o[:, :, 3] = np.clip(np.rint(alp[:, :, 0]), 0, 255).astype(np.uint8)
    o[o[:, :, 3] == 0, :3] = 0
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(Image.fromarray(o, "RGBA"), ((size - wh[0]) // 2, (size - wh[1]) // 2))
    return canvas


def family_labels(rgb: np.ndarray) -> np.ndarray:
    """Separate browns from oranges, while keeping distinct color categories."""
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    # Each interval is half-open; no overlap or arbitrary global K-means clusters.
    labels = np.select(
        [h < 14, h < 34, h < 65, h < 90, h < 165, h < 200,
         h < 230, h < 265, h < 300, h < 345],
        list(range(10)), default=0).astype(np.uint8)
    brown = (h >= 14) & (h < 65) & (s >= 0.17) & (v < 0.64) & (luma < 160)
    labels[brown] = 10
    labels[(s < 0.17) | (v < 0.085)] = 11
    return labels


def _lab_palette(lab_pixels: np.ndarray, family: int, min_gap: float) -> np.ndarray:
    """Exactly three tonal slots (dark, middle, light) for each present family."""
    L = lab_pixels[:, 0]
    q = np.quantile(L, (0.14, 0.5, 0.86)).astype(np.float32)
    middle = float(q[1])
    # Large, visible tone steps: avoid hundreds of near-duplicate shades.
    dark = min(float(q[0]), middle - min_gap)
    light = max(float(q[2]), middle + min_gap)
    levels = np.clip(np.array([dark, middle, light], dtype=np.float32), 2.0, 96.0)
    if family == 11:
        ab = np.array([0, 0], dtype=np.float32)
    else:
        ab = np.median(lab_pixels[:, 1:], axis=0).astype(np.float32)
    triplet = np.column_stack((levels, np.repeat(ab[None, :], 3, axis=0)))
    out = cv2.cvtColor(triplet.reshape(1, 3, 3), cv2.COLOR_LAB2RGB).reshape(3, 3)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def _cleanup_small_islands(families: np.ndarray, tones: np.ndarray, visible: np.ndarray,
                           max_area: int = 8) -> np.ndarray:
    """Remove isolated 1–8 px dither islands only within one color family.

    Bigger structures, material edges and long dark cracks remain intact.
    """
    out = tones.copy()
    h, w = tones.shape
    for family in np.unique(families[visible]):
        fam_mask = visible & (families == family)
        for tone in range(3):
            mask = (fam_mask & (out == tone)).astype(np.uint8)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for idx in range(1, n):
                if int(stats[idx, cv2.CC_STAT_AREA]) > max_area:
                    continue
                x, y, ww, hh, _ = stats[idx]
                x0, y0 = max(0, int(x)-1), max(0, int(y)-1)
                x1, y1 = min(w, int(x+ww)+1), min(h, int(y+hh)+1)
                component = cc[y0:y1, x0:x1] == idx
                surrounding = fam_mask[y0:y1, x0:x1] & ~component
                choices = out[y0:y1, x0:x1][surrounding]
                choices = choices[choices != tone]
                if choices.size < 4:
                    continue
                binc = np.bincount(choices, minlength=3)
                candidate = int(np.argmax(binc))
                if binc[candidate] / choices.size < 0.65:
                    continue
                local = out[y0:y1, x0:x1]
                local[component] = candidate
    return out


def process_image(img: Image.Image, *, size: int = SIZE, min_gap: float = 17.0,
                  smoothing: bool = True) -> tuple[Image.Image, dict]:
    original = np.asarray(fit_256(img, size), dtype=np.uint8)
    alpha = original[:, :, 3]
    visible = alpha >= 8
    output = np.zeros_like(original)
    output[:, :, 3] = alpha  # Exact alpha preservation, including antialiasing.
    if not np.any(visible):
        return Image.fromarray(output, "RGBA"), {"families": {}, "size": [size, size]}

    rgb = original[:, :, :3].copy()
    # Fill only invisible RGB for safe edge-aware filtering (never change alpha).
    missing = (alpha < 8).astype(np.uint8)
    if np.any(missing):
        rgb = cv2.inpaint(rgb, missing, 2, cv2.INPAINT_TELEA)
    if smoothing:
        rgb = cv2.bilateralFilter(rgb, 7, 37, 4)
    groups = family_labels(rgb)
    lab = cv2.cvtColor(rgb.astype(np.float32)/255.0, cv2.COLOR_RGB2LAB)
    smooth_l = lab[:, :, 0]
    if smoothing:
        smooth_l = cv2.bilateralFilter(smooth_l, 5, 9.0, 3)
    tones = np.zeros(groups.shape, np.uint8)
    palettes = {}
    family_report = {}
    for family in np.unique(groups[visible]):
        mask = visible & (groups == family)
        pixels = lab[mask]
        tri = _lab_palette(pixels, int(family), min_gap)
        palettes[int(family)] = tri
        tri_lab = cv2.cvtColor(tri.astype(np.float32).reshape(1, 3, 3) / 255.0,
                               cv2.COLOR_RGB2LAB).reshape(3, 3)
        thresholds = (float((tri_lab[0, 0] + tri_lab[1, 0]) / 2),
                      float((tri_lab[1, 0] + tri_lab[2, 0]) / 2))
        tones[mask] = np.searchsorted(thresholds, smooth_l[mask]).astype(np.uint8)
        family_report[FAMILY_NAMES[int(family)]] = {
            "pixels": int(mask.sum()),
            "tones_rgb": {"dark": tri[0].tolist(), "neutral": tri[1].tolist(),
                          "light": tri[2].tolist()},
        }
    if smoothing:
        tones = _cleanup_small_islands(groups, tones, visible)
    for family, triplet in palettes.items():
        mask = visible & (groups == family)
        output[:, :, :3][mask] = triplet[tones[mask]]
    output[~visible, :3] = 0
    return Image.fromarray(output, "RGBA"), {
        "size": [size, size], "visible_pixels": int(visible.sum()),
        "alpha_identical_after_resize": bool(np.array_equal(output[:,:,3], original[:,:,3])),
        "families": family_report,
    }


def run_batch(input_dir: Path, output_dir: Path, *, size: int = SIZE,
              min_gap: float = 17.0, smoothing: bool = True) -> dict:
    sources = sorted(input_dir.rglob('*.png'))
    if not sources:
        raise ValueError(f'No PNG files found in {input_dir}')
    if output_dir.resolve() == input_dir.resolve() or input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError('Output directory must be separate from the source directory')
    records = {}
    for src in sources:
        relative = src.relative_to(input_dir)
        dest = output_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            processed, report = process_image(im, size=size, min_gap=min_gap, smoothing=smoothing)
        processed.save(dest, optimize=True)
        records[str(relative)] = report
        print(f'IbBit: {relative}', flush=True)
    report = {"version": "IbBit v1", "count": len(records), "size": [size, size], "files": records}
    (output_dir / 'ibbit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='IbBit v1 — 3 tones per color family, size-first, edge-aware cleanup')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--size', type=int, default=SIZE)
    parser.add_argument('--tone-gap', type=float, default=17.0, help='Minimum LAB lightness gap between shades')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable bilateral filter and island cleanup')
    args = parser.parse_args()
    if not 32 <= args.size <= 2048:
        parser.error('size must be between 32 and 2048')
    run_batch(args.input_dir, args.output_dir, size=args.size, min_gap=args.tone_gap,
              smoothing=not args.no_smoothing)


if __name__ == '__main__':
    main()
