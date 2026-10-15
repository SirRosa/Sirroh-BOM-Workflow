#!/usr/bin/env python3
"""Rebuild an RGBA atlas by immutable coordinates in split_atlas.py's manifest.

Never sorts by name: edited prefab names and file ordering cannot change
sprite positions. Conflicting, missing, overlapping or altered tiles fail hard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

SCHEMA = "sirroh.atlas.tiles.v1"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def assemble(manifest_path: Path, output: Path, verify_original: Path | None = None, allow_modified: bool = False) -> dict:
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unknown manifest schema: {data.get('schema')}")
    if data.get("atlas", {}).get("mode") != "RGBA":
        raise ValueError("RGBA manifest required")
    width, height = data["atlas"]["size"]
    tw, th = data["tile_size"]
    cols, rows = data["grid"]["columns"], data["grid"]["rows"]
    if any(not isinstance(v, int) or v <= 0 for v in (width, height, tw, th, cols, rows)):
        raise ValueError("Invalid atlas dimensions")
    if width != tw * cols or height != th * rows:
        raise ValueError("Manifest grid dimensions do not match atlas dimensions")
    tiles = data.get("tiles", [])
    if len(tiles) != cols * rows:
        raise ValueError("Incorrect tile count: missing/duplicate cells")
    atlas = Image.new("RGBA", (width, height))
    occupied = set()
    unchanged = True
    base = manifest_path.parent.resolve()
    for item in tiles:
        r, c = item["row"], item["column"]
        if not isinstance(r, int) or not isinstance(c, int) or not (1 <= r <= rows and 1 <= c <= cols):
            raise ValueError("Invalid row/column")
        pos = (r, c)
        if pos in occupied:
            raise ValueError(f"Duplicate position: {pos}")
        occupied.add(pos)
        x, y = (c - 1) * tw, (r - 1) * th
        tile_id = f"r{r:02d}_c{c:02d}"
        if item["id"] != tile_id or (item["x"], item["y"], item["width"], item["height"]) != (x, y, tw, th):
            raise ValueError(f"Position mismatch at {pos} / {item['id']}")
        # Prevent path traversal from manually edited or hostile manifests.
        filename = item["file"]
        path = (base / filename).resolve()
        if not path.is_relative_to(base) or path != (base / "tiles" / (tile_id + ".png")):
            raise ValueError(f"Unexpected tile file path: {filename}")
        with Image.open(path) as input_tile:
            if input_tile.mode != "RGBA" or input_tile.size != (tw, th):
                raise ValueError(f"Wrong mode or size: {path}")
            tile = input_tile.copy()
        if sha256(tile.tobytes()) != item["sha256_rgba"]:
            unchanged = False
            if not allow_modified:
                raise ValueError(f"Tile altered since splitting: {tile_id}; use --allow-modified intentionally")
        atlas.paste(tile, (x, y))
    if len(occupied) != cols * rows:
        raise ValueError("Not all atlas cells are present")

    output = Path(output)
    if verify_original is not None and output.resolve() == Path(verify_original).resolve():
        raise ValueError("Output must not overwrite the original atlas being checked")
    output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output, "PNG", optimize=True)
    expected_pixels = data["atlas"]["sha256_rgba"]
    actual_pixels = sha256(atlas.tobytes())
    pixels_match_original = actual_pixels == expected_pixels
    bytes_match_original = None
    if verify_original is not None:
        original = Path(verify_original)
        with Image.open(original) as original_image:
            if original_image.mode != "RGBA" or original_image.size != (width, height):
                raise AssertionError("Reassembled atlas differs in mode or size")
            if original_image.tobytes() != atlas.tobytes():
                raise AssertionError("Reassembled atlas differs from original RGBA pixels")
        bytes_match_original = sha256(original.read_bytes()) == sha256(output.read_bytes())
    if not allow_modified and not pixels_match_original:
        raise AssertionError("Reassembled atlas RGBA pixels differ from original manifest")
    result = {
        "schema": SCHEMA, "tiles": len(tiles), "size": [width, height],
        "pixel_identical": bool(pixels_match_original), "png_binary_identical": bytes_match_original,
        "source_sha256_rgba": expected_pixels, "reassembled_sha256_rgba": actual_pixels,
        "source_sha256_png": data["atlas"]["sha256_png"],
        "reassembled_sha256_png": sha256(output.read_bytes()),
        "modified_tiles_detected": not unchanged,
    }
    report = output.with_suffix(".verification.json")
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"ASSEMBLE OK: {output} | pixel-identical: {pixels_match_original} | PNG byte-identical: {bytes_match_original}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild sprite atlas from editable placement manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-original", type=Path)
    p.add_argument("--allow-modified", action="store_true", help="Allow intentionally edited tile PNGs")
    a = p.parse_args()
    assemble(a.manifest, a.output, a.verify_original, a.allow_modified)


if __name__ == "__main__":
    main()
