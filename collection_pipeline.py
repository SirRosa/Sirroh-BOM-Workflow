#!/usr/bin/env python3
"""Rebuild and BOM-process all non-creature, non-legacy icons currently in this repository.

Stages: atlas assembly -> BOM v8 -> atlas split -> named review preview.
The source PNGs are read-only. New results are written to collection_results/.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from collection_categories import classify, order_key, organize_collection

ROOT = Path(__file__).resolve().parent
TILES = ROOT / "atlas_work/Valheim_Compass_Atlas_BOM/tiles"
OLD_MANIFEST = ROOT / "atlas_work/Valheim_Compass_Atlas_BOM/atlas_manifest.json"
WORK = ROOT / "collection_run"
SOURCE = WORK / "source"
PROCESSED = WORK / "processed"
FINAL_SPLIT = WORK / "final_split"
RESULT = ROOT / "collection_results"
TILE = 96
COLUMNS = 10
CREATURES = {"deer", "boar", "wolf", "troll", "serpent"}
SKIP_DIRS = {"legacy", "creatures", "creature", "mobs", "monsters"}
GRID_FILE = re.compile(r"r\d{2}_c\d{2}\.png$", re.I)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(*args: object) -> None:
    command = [sys.executable, *map(str, args)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def source_selection() -> tuple[list[tuple[str, Path]], list[str]]:
    base = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    found: dict[str, tuple[str, Path]] = {}
    exclusions = []
    for t in base["tiles"]:
        name = str(t.get("name") or "").strip()
        src = OLD_MANIFEST.parent / t["file"]
        if t.get("empty") or not name:
            continue
        if name.lower() in CREATURES:
            exclusions.append(str(src.relative_to(ROOT)))
            continue
        if not src.is_file():
            raise FileNotFoundError(src)
        found[name.casefold()] = (name, src)

    # Named files are more recent than the original rRR_cCC extraction.
    for src in sorted(TILES.rglob("*.png"), key=lambda p: str(p).casefold()):
        parts = [p.lower() for p in src.relative_to(TILES).parts[:-1]]
        if any(part in SKIP_DIRS for part in parts) or GRID_FILE.fullmatch(src.name):
            continue
        name = src.stem
        if name.casefold() in CREATURES:
            exclusions.append(str(src.relative_to(ROOT)))
            continue
        found[name.casefold()] = (name, src)

    values = sorted(found.values(), key=lambda value: (order_key(value[0]), str(value[1])))
    if not values:
        raise RuntimeError("No non-creature icon files found")
    if len({x[0].casefold() for x in values}) != len(values):
        raise RuntimeError("Duplicate names in selected collection")
    return values, exclusions


def prepare_source(selected: list[tuple[str, Path]]) -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    if RESULT.exists():
        shutil.rmtree(RESULT)
    (SOURCE / "tiles").mkdir(parents=True)
    RESULT.mkdir(parents=True)

    rows = math.ceil(len(selected) / COLUMNS)
    atlas = Image.new("RGBA", (TILE * COLUMNS, TILE * rows))
    records = []
    sources = []
    for index in range(rows * COLUMNS):
        r, c = divmod(index, COLUMNS)
        name, path = selected[index] if index < len(selected) else (f"empty_{index}", None)
        sprite = Image.new("RGBA", (TILE, TILE))
        if path is not None:
            with Image.open(path) as im:
                original = im.convert("RGBA")
            if original.size == (TILE, TILE):
                sprite = original.copy()
            else:
                # Keep proportions when a source icon has a nonstandard size.
                scaled = original.copy()
                scaled.thumbnail((TILE, TILE), Image.Resampling.LANCZOS)
                sprite.alpha_composite(scaled, ((TILE-scaled.width)//2, (TILE-scaled.height)//2))
            sources.append({
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "source_size": list(original.size),
                "source_sha256": digest(path.read_bytes()),
                "category": classify(name),
            })
        ident = f"r{r+1:02d}_c{c+1:02d}"
        file = f"tiles/{ident}.png"
        sprite.save(SOURCE / file, "PNG", optimize=True)
        atlas.paste(sprite, (c*TILE, r*TILE))
        records.append({
            "id": ident, "name": name+".png" if path else ident+".png",
            "prefab": name if path else "",
            "category": classify(name) if path else None,
            "row": r+1, "column": c+1, "x": c*TILE, "y": r*TILE,
            "width": TILE, "height": TILE, "file": file,
            "empty": path is None, "sha256_rgba": digest(sprite.tobytes()),
        })
    atlas_path = SOURCE / "assembled_source.png"
    atlas.save(atlas_path, "PNG", optimize=True)
    manifest = {
        "schema": "sirroh.atlas.tiles.v1",
        "description": "Non-creature non-legacy collection, named icons override original grid tiles",
        "atlas": {
            "source": atlas_path.name, "size": list(atlas.size), "mode": "RGBA",
            "sha256_rgba": digest(atlas.tobytes()),
            "sha256_png": digest(atlas_path.read_bytes())
        },
        "grid": {"columns": COLUMNS, "rows": rows},
        "tile_size": [TILE, TILE], "tiles": records,
    }
    (SOURCE / "atlas_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (RESULT / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def main() -> None:
    selected, exclusions = source_selection()
    print(f"SELECTED: {len(selected)} icons; excluded creature icons: {len(exclusions)}", flush=True)
    prepare_source(selected)

    # 1. Assemble source atlas with repo's assembler and verify against source pixels.
    run("tools/assemble_atlas.py", "--manifest", SOURCE / "atlas_manifest.json",
        "--output", SOURCE / "assembled_verified.png",
        "--verify-original", SOURCE / "assembled_source.png")

    # 2. Process each original sprite via current source-guided BOM v8.
    run("src/bom_v8.py", "--source-root", SOURCE, "--output-root", PROCESSED)

    # 3. Split processed atlas with the existing lossless splitter.
    run("tools/split_atlas.py", "--input", PROCESSED / "atlas_BOM_v8.png",
        "--dest", FINAL_SPLIT)
    final_manifest_path = FINAL_SPLIT / "atlas_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    for dst, src in zip(final_manifest["tiles"],
                        json.loads((SOURCE / "atlas_manifest.json").read_text(encoding="utf-8"))["tiles"]):
        if dst["id"] != src["id"]:
            raise AssertionError(f"Grid ID mismatch: {dst['id']} != {src['id']}")
        dst["name"], dst["prefab"] = src["prefab"] or dst["id"], src["prefab"]
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    # Lossless verification of the new split, then 4. the labeled preview.
    run("tools/assemble_atlas.py", "--manifest", final_manifest_path,
        "--output", WORK / "verified_reassembled.png",
        "--verify-original", PROCESSED / "atlas_BOM_v8.png")
    run("tools/preview_atlas_labels.py", "--manifest", final_manifest_path,
        "--output", RESULT / "preview_named_atlas.png", "--label-mode", "name")

    # Publish inspectable outputs plus individual final 96px icons.
    shutil.copy2(PROCESSED / "atlas_BOM_v8.png", RESULT / "atlas_BOM_v8.png")
    shutil.copy2(PROCESSED / "atlas_BOM_v8.json", RESULT / "bom_report.json")
    shutil.copy2(final_manifest_path, RESULT / "atlas_manifest.json")
    shutil.copy2(WORK / "verified_reassembled.verification.json", RESULT / "verification.json")
    icon_dir = RESULT / "icons"
    icon_dir.mkdir()
    for name, _ in selected:
        filename = f"{name}.png"
        src = PROCESSED / "ikony_BOM_v8_96x96" / filename
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, icon_dir / filename)
        with Image.open(icon_dir / filename) as im:
            if im.size != (96, 96) or im.mode != "RGBA":
                raise AssertionError(f"Invalid processed icon: {filename}")
    report = json.loads((RESULT / "bom_report.json").read_text(encoding="utf-8"))
    verification = json.loads((RESULT / "verification.json").read_text(encoding="utf-8"))
    if not verification["pixel_identical"] or report["processed_icons"] != len(selected):
        raise AssertionError("BOM count or lossless roundtrip verification failed")
    summary = {
        "source_icons": len(selected), "processed_icons": report["processed_icons"],
        "excluded_creature_files": exclusions,
        "legacy_excluded": True, "originals_unchanged": True,
        "source_priority": "named files over rRR_cCC tiles",
        "grid": final_manifest["grid"],
        "atlas_size": final_manifest["atlas"]["size"],
        "palette": "BOM v8, up to 5 shades per color family",
        "split_roundtrip_pixel_identical": verification["pixel_identical"],
        "file_names": [name + ".png" for name, _ in selected],
    }
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    # Sort finished PNGs into stable category folders and render grouped reviews.
    organize_collection(RESULT)
    print("SUCCESS", json.dumps({k: v for k, v in summary.items() if k != "file_names"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
