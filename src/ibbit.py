#!/usr/bin/env python3
"""IbBit v7 — universal 3-tone grim palette for 2D assets.

There are NO material/object-specific profiles. Every asset follows the same
palette, family classification, brightness mapping and artifact cleanup.

Dependencies: numpy, opencv-python-headless, Pillow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 256
FAMILIES = (
    "red", "orange", "yellow", "olive", "green", "teal", "cyan",
    "blue", "violet", "magenta", "brown", "neutral"
)
NEUTRAL_ID = FAMILIES.index("neutral")
BROWN_ID = FAMILIES.index("brown")

# One fixed muddy/swamp-dark palette, shared across ALL images. Each family has 3 ordered tones.
# The darkest neutral is the user's chosen rotten-gray-black RGB (34, 37, 34).
# Nothing in the other families may be darker in perceived brightness.
PALETTE = {
    "red":     ((56, 38, 36), (88, 59, 56), (122, 84, 78)),
    "orange":  ((60, 46, 35), (90, 69, 50), (126, 96, 70)),
    "yellow":  ((57, 56, 39), (86, 85, 59), (120, 117, 80)),
    "olive":   ((46, 52, 36), (74, 81, 56), (107, 113, 81)),
    "green":   ((42, 51, 42), (68, 80, 63), (101, 111, 86)),
    "teal":    ((42, 53, 49), (69, 82, 74), (101, 114, 101)),
    "cyan":    ((44, 53, 53), (70, 83, 82), (102, 114, 110)),
    "blue":    ((44, 52, 56), (70, 80, 85), (103, 111, 116)),
    "violet":  ((52, 47, 54), (78, 72, 81), (110, 103, 110)),
    "magenta": ((57, 42, 49), (85, 62, 71), (117, 88, 95)),
    "brown":   ((52, 41, 31), (80, 65, 48), (115, 94, 70)),
    "neutral": ((34, 37, 34), (68, 75, 67), (118, 122, 110)),
}


def perceived_brightness(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def validate_palette(palette: dict = PALETTE) -> None:
    if set(palette) != set(FAMILIES):
        raise ValueError("Palette must contain all defined color families")
    black = perceived_brightness(palette["neutral"][0])
    for family, colors in palette.items():
        if len(colors) != 3:
            raise ValueError(f"{family}: exactly 3 tones required")
        luminances = perceived_brightness(colors)
        if not (luminances[0] < luminances[1] < luminances[2]):
            raise ValueError(f"{family}: tone brightness must be strictly increasing")
        if family != "neutral" and np.any(luminances <= black):
            raise ValueError(f"{family}: no tone can be as dark as neutral dark")


validate_palette()
PALETTE_ARRAY = np.asarray([PALETTE[f] for f in FAMILIES], dtype=np.uint8)


def fit_256(img: Image.Image, size: int = SIZE) -> Image.Image:
    """Resize FIRST, with preserved aspect ratio and premultiplied-alpha edges."""
    src = img.convert("RGBA")
    if src.size == (size, size):
        return src.copy()
    ratio = min(size / src.width, size / src.height)
    wh = tuple(max(1, int(round(dim * ratio))) for dim in src.size)
    a = np.asarray(src, dtype=np.float32)
    alpha = a[:, :, 3:4] / 255.0
    premul = a[:, :, :3] * alpha
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    rgb = cv2.resize(premul, wh, interpolation=interp)
    alp = cv2.resize(a[:, :, 3], wh, interpolation=interp)
    if alp.ndim == 2:
        alp = alp[:, :, None]
    o = np.zeros((wh[1], wh[0], 4), dtype=np.uint8)
    o[:, :, :3] = np.clip(np.rint(rgb / np.maximum(alp / 255, 1e-5)), 0, 255).astype(np.uint8)
    o[:, :, 3] = np.clip(np.rint(alp[:, :, 0]), 0, 255).astype(np.uint8)
    o[o[:, :, 3] == 0, :3] = 0
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(Image.fromarray(o, "RGBA"), ((size - wh[0]) // 2, (size - wh[1]) // 2))
    return canvas


def family_labels(rgb: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Conservative, universal HSV classification: gray is neutral, wood-like browns brown.

    Only distinctly chromatic pixels enter reserved accent families.
    Nothing about an image's filename or type is involved.
    """
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    luma = perceived_brightness(rgb)
    labels = np.full(visible.shape, NEUTRAL_ID, dtype=np.uint8)

    clear_color = visible & (s >= 0.30) & (v >= 0.12)
    brown = clear_color & (h >= 12) & (h < 60) & (v < 0.72) & (luma < 170)
    labels[brown] = BROWN_ID
    remaining = clear_color & ~brown
    hue_family = np.select(
        [h < 14, h < 34, h < 65, h < 90, h < 165, h < 200,
         h < 230, h < 265, h < 300, h < 345],
        list(range(10)), default=0,
    ).astype(np.uint8)
    labels[remaining] = hue_family[remaining]

    # Prevent colorful pixel noise from becoming a distinct material: a chromatic
    # family must be both sufficiently present and clearly saturated on average.
    total = int(visible.sum())
    counts = np.bincount(labels[visible], minlength=len(FAMILIES))
    keep_brown = counts[BROWN_ID] >= max(36, round(0.01 * total))
    keep = {NEUTRAL_ID}
    if keep_brown:
        keep.add(BROWN_ID)
    for fam in range(10):
        mask = visible & (labels == fam)
        count = int(counts[fam])
        if count >= max(24, round(0.0035 * total)) and float(s[mask].mean()) >= 0.42:
            keep.add(fam)
    for fam in range(10):
        if fam in keep:
            continue
        mask = visible & (labels == fam)
        if fam in (0, 1, 2) and keep_brown:
            labels[mask] = BROWN_ID
        else:
            labels[mask] = NEUTRAL_ID
    if not keep_brown:
        labels[visible & (labels == BROWN_ID)] = NEUTRAL_ID

    # Absorb globally insignificant colored specks into the dominant family.
    counts = np.bincount(labels[visible], minlength=len(FAMILIES))
    dominant = int(np.argmax(counts))
    dominance = counts[dominant] / max(total, 1)
    if dominant in (NEUTRAL_ID, BROWN_ID) and dominance >= 0.73:
        for fam in np.flatnonzero(counts):
            if fam == dominant:
                continue
            if fam not in (NEUTRAL_ID, BROWN_ID) and counts[fam] <= max(48, int(0.01 * total)):
                labels[visible & (labels == fam)] = dominant
    return labels


def luminance_to_tones(lightness: np.ndarray, visible_family: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Three ranked tones: shadows / mids / highlights, no absolute dark override.

    Adaptive thresholds ensure contrast is retained even when source pixels are
    globally dark (as in the supplied stone walls). There is no forced-dark rule
    that can erase the mid-tone. Truly flat input is kept flat.
    """
    levels = np.zeros(lightness.shape, dtype=np.uint8)
    vals = lightness[visible_family]
    if vals.size == 0:
        return levels, (0.0, 0.0)
    q1, q2 = np.quantile(vals, (1 / 3, 2 / 3))
    if q2 - q1 < 1.0:
        middle = float(np.median(vals))
        q1, q2 = middle - 0.5, middle + 0.5
    # Universal brightness floor: if a family occurs only in almost-black pixels,
    # its relative upper tercile must NOT become a bright highlight.  These are
    # absolute L* floors applied to ALL families, not material/object profiles.
    q1 = max(float(q1), 12.0)
    q2 = max(float(q2), 21.0, q1 + 3.0)
    levels[visible_family] = np.searchsorted([q1, q2], vals, side="right").astype(np.uint8)
    return levels, (float(q1), float(q2))


def _component_counts(indexed: np.ndarray, visible: np.ndarray, below: int = 16) -> dict:
    count = 0
    pixels = 0
    for color_id in np.unique(indexed[visible]):
        mask = ((indexed == color_id) & visible).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        count += int(np.count_nonzero(areas < below))
        pixels += int(areas[areas < below].sum())
    return {"components": count, "pixels": pixels}


def cleanup_small_color_blobs(indexed: np.ndarray, visible: np.ndarray, min_area: int = 16,
                              passes: int = 6) -> tuple[np.ndarray, dict]:
    """Replace all possible 8-connected single-color blobs <min_area pixels.

    Choose the touching color with the greatest EIGHT-neighbor boundary contact.
    Never turn transparent pixels opaque; tie-break deterministically. Each pass
    re-labels against the updated result, so cleanup is repeatable and stable.
    """
    out = indexed.copy().astype(np.int32)
    h, w = out.shape
    before = _component_counts(out, visible, min_area)
    changed_pixels = 0
    for _ in range(passes):
        changed = False
        # Stable scanning order; don't let mutation leak across a component mask.
        for color_id in np.unique(out[visible]).tolist():
            color_mask = (visible & (out == color_id)).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
            for idx in range(1, n):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area >= min_area:
                    continue
                x, y, ww, hh, _ = (int(z) for z in stats[idx])
                component = labels[y:y + hh, x:x + ww] == idx
                ys, xs = np.where(component)
                ys, xs = ys + y, xs + x
                contacts: dict[int, int] = {}
                for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    ny, nx = ys + dy, xs + dx
                    good = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                    ny, nx = ny[good], nx[good]
                    nonself = visible[ny, nx] & (labels[ny, nx] != idx)
                    ny, nx = ny[nonself], nx[nonself]
                    if not len(ny):
                        continue
                    colors, freqs = np.unique(out[ny, nx], return_counts=True)
                    for new_color, freq in zip(colors.tolist(), freqs.tolist()):
                        if new_color != color_id:
                            contacts[new_color] = contacts.get(new_color, 0) + freq
                if not contacts:
                    continue  # detached whole object / no other opaque neighbors
                # Favor maximal edge contact; then similar perceptual brightness,
                # then stable numeric ID. Do not add random new colors.
                old_luma = float(perceived_brightness(PALETTE_ARRAY[color_id // 3, color_id % 3]))
                target = min(contacts, key=lambda v: (
                    -contacts[v],
                    abs(float(perceived_brightness(PALETTE_ARRAY[v // 3, v % 3])) - old_luma),
                    v,
                ))
                # Avoid using stale per-color CC labels to overwrite pixels already
                # changed by another component in the same pass.
                still_same = out[ys, xs] == color_id
                if np.any(still_same):
                    out[ys[still_same], xs[still_same]] = target
                    changed_pixels += int(still_same.sum())
                    changed = True
        if not changed:
            break
    after = _component_counts(out, visible, min_area)
    return out, {"min_area": min_area, "removed_pixels": changed_pixels,
                 "components_before": before["components"],
                 "components_after": after["components"]}


def process_image(img: Image.Image, *, size: int = SIZE, smoothing: bool = True,
                  min_area: int = 16) -> tuple[Image.Image, dict]:
    original = np.asarray(fit_256(img, size), dtype=np.uint8)
    alpha = original[..., 3]
    visible = alpha >= 8
    output = np.zeros_like(original)
    output[..., 3] = alpha
    if not np.any(visible):
        return Image.fromarray(output, "RGBA"), {"size": [size, size], "families": {}, "alpha_preserved": True}

    rgb = original[..., :3].copy()
    unseen = (alpha < 8).astype(np.uint8)
    if np.any(unseen):
        rgb = cv2.inpaint(rgb, unseen, 2, cv2.INPAINT_TELEA)
    groups = family_labels(original[..., :3], visible)
    if smoothing:
        rgb = cv2.bilateralFilter(rgb, 7, 30, 4)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    if smoothing:
        lightness = cv2.bilateralFilter(lightness, 5, 8, 3)

    tones = np.zeros(visible.shape, dtype=np.uint8)
    family_report = {}
    for family in np.unique(groups[visible]):
        mask = visible & (groups == family)
        band, thresholds = luminance_to_tones(lightness, mask)
        tones[mask] = band[mask]
        family_report[FAMILIES[int(family)]] = {
            "input_pixels": int(mask.sum()),
            "quantile_thresholds": list(thresholds),
            "raw_tone_pixels": [int((band[mask] == k).sum()) for k in range(3)],
        }

    color_index = groups.astype(np.int32) * 3 + tones.astype(np.int32)
    cleanup = {"min_area": min_area, "removed_pixels": 0}
    if smoothing:
        color_index, cleanup = cleanup_small_color_blobs(color_index, visible, min_area=min_area)
    out_rgb = PALETTE_ARRAY[color_index // 3, color_index % 3]
    output[..., :3][visible] = out_rgb[visible]
    output[~visible, :3] = 0
    for idx, fam in enumerate(FAMILIES):
        if fam in family_report:
            family_report[fam]["final_tone_pixels"] = [int(((color_index == idx * 3 + k) & visible).sum()) for k in range(3)]

    return Image.fromarray(output, "RGBA"), {
        "version": "IbBit v7",
        "size": [size, size],
        "visible_pixels": int(visible.sum()),
        "alpha_preserved": bool(np.array_equal(output[..., 3], original[..., 3])),
        "universal_palette": True,
        "families": family_report,
        "cleanup": cleanup,
    }


def run_batch(input_dir: Path, output_dir: Path, *, size: int = SIZE,
              min_area: int = 16, smoothing: bool = True) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be outside the input directory")
    sources = sorted(input_dir.rglob("*.png"))
    if not sources:
        raise ValueError(f"No PNG files found in {input_dir}")
    records = {}
    for source in sources:
        relative = source.relative_to(input_dir)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            processed, report = process_image(image, size=size, min_area=min_area, smoothing=smoothing)
        processed.save(target, optimize=True)
        records[str(relative)] = report
        print("IbBit v7:", relative, flush=True)
    full_report = {"version": "IbBit v7", "size": [size, size],
                   "processed": len(records), "palette": PALETTE,
                   "families": FAMILIES, "files": records}
    (output_dir / "ibbit_report.json").write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return full_report


def main() -> None:
    ap = argparse.ArgumentParser(description="IbBit v7: shared 3-tone grim palette, 256px first, small-island cleanup")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--no-smoothing", action="store_true")
    args = ap.parse_args()
    if not (32 <= args.size <= 2048):
        ap.error("--size must be in 32..2048")
    if not (2 <= args.min_area <= 128):
        ap.error("--min-area must be in 2..128")
    run_batch(args.input_dir, args.output_dir, size=args.size,
              min_area=args.min_area, smoothing=not args.no_smoothing)


if __name__ == "__main__":
    main()