#!/usr/bin/env python3
"""Losslessly split a regularly gridded RGBA PNG atlas into named PNG tiles.

The JSON manifest is the contract with assemble_atlas.py. The immutable `id`
(r01_c01) and stable `file` identify tile placement; `name` is explicitly
editable later to the game's prefab name without moving or renaming files.
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


def split_atlas(source: Path, folder: Path, tile_width: int = 96, tile_height: int = 96) -> dict:
    source = Path(source)
    folder = Path(folder)
    if tile_width < 1 or tile_height < 1:
        raise ValueError("Tile dimensions must be positive")
    with Image.open(source) as img:
        # Exact-channel round-trip: require original RGBA, no implicit conversion.
        if img.mode != "RGBA":
            raise ValueError(f"RGBA required for pixel-exact splitting; got {img.mode}")
        atlas = img.copy()
    w, h = atlas.size
    if w % tile_width or h % tile_height:
        raise ValueError(f"Atlas {w}x{h} cannot be divided evenly into {tile_width}x{tile_height} tiles")
    cols, rows = w // tile_width, h // tile_height
    manifest_path = folder / "atlas_manifest.json"
    existing = {}
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA or data.get("grid") != {"columns": cols, "rows": rows} or data.get("atlas", {}).get("size") != [w, h]:
            raise ValueError("Existing manifest belongs to a different layout; refusing to overwrite names")
        existing = {t["id"]: t for t in data.get("tiles", [])}

    tiles_dir = folder / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    records = []
    nonempty = 0
    atlas_bytes = atlas.tobytes()
    for r in range(rows):
        for c in range(cols):
            tile_id = f"r{r + 1:02d}_c{c + 1:02d}"
            relative_file = f"tiles/{tile_id}.png"
            x, y = c * tile_width, r * tile_height
            tile = atlas.crop((x, y, x + tile_width, y + tile_height))
            tile.save(folder / relative_file, format="PNG", optimize=True)
            old = existing.get(tile_id, {})
            is_empty = tile.getchannel("A").getbbox() is None
            nonempty += not is_empty
            records.append({
                "id": tile_id,
                "name": old.get("name") if isinstance(old.get("name"), str) else tile_id,
                "prefab": old.get("prefab") if isinstance(old.get("prefab"), str) else "",
                "row": r + 1, "column": c + 1,
                "x": x, "y": y, "width": tile_width, "height": tile_height,
                "file": relative_file,
                "empty": bool(is_empty),
                "sha256_rgba": sha256(tile.tobytes()),
            })
    manifest = {
        "schema": SCHEMA,
        "description": "Edit tiles[].name and/or tiles[].prefab. Keep id, file and positions unchanged.",
        "atlas": {
            "source": source.name,
            "size": [w, h],
            "mode": "RGBA",
            "sha256_rgba": sha256(atlas_bytes),
            "sha256_png": sha256(source.read_bytes()),
        },
        "grid": {"columns": cols, "rows": rows},
        "tile_size": [tile_width, tile_height],
        "tiles": records,
    }
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SPLIT OK: {cols}x{rows}, {len(records)} tiles ({nonempty} nonempty); manifest {manifest_path}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Split an RGBA atlas losslessly with editable JSON manifest")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--tile-width", type=int, default=96)
    p.add_argument("--tile-height", type=int, default=96)
    a = p.parse_args()
    split_atlas(a.input, a.dest, a.tile_width, a.tile_height)


if __name__ == "__main__":
    main()
