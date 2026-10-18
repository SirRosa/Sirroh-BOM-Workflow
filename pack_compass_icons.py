#!/usr/bin/env python3
"""Package original IconsForCompass PNGs into thematic ZIP files.

Only copies original bytes; does not resize, re-encode or alter any images.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import struct

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "IconsForCompass"
OUT = ROOT / "IconPacks" / "Zips"
MANIFEST = ROOT / "IconPacks" / "manifest.json"


def main() -> None:
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    packs = spec["packs"]
    input_files = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*.png")}
    chosen = [name for entries in packs.values() for name in entries]

    if Counter(chosen) != Counter(input_files):
        missing = sorted(input_files - set(chosen))
        unexpected = sorted(set(chosen) - input_files)
        repeated = sorted(name for name, count in Counter(chosen).items() if count > 1)
        raise ValueError(f"Manifest/source mismatch; missing={missing}, unexpected={unexpected}, repeated={repeated}")

    if len(chosen) != spec["total_icons"]:
        raise ValueError(f"Expected {spec['total_icons']} icons, got {len(chosen)}")
    if not all(1 <= len(entries) <= 10 for entries in packs.values()):
        raise ValueError("A ZIP pack must contain between 1 and 10 PNGs")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.zip"):
        old.unlink()

    lines = [
        "# Compass icon packs",
        "",
        f"Source: `IconsForCompass/` — **{len(chosen)} original PNGs**.",
        "Archives preserve the source PNG bytes and original file names.",
        "Each icon appears exactly once; each ZIP contains at most 10 icons.",
        "",
        "| ZIP | Files | Contents |",
        "| --- | ---: | --- |",
    ]
    for pack, entries in packs.items():
        names = [Path(entry).name for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate file names after flattening in {pack}")

        destination = OUT / f"{pack}.zip"
        with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for entry, filename in zip(entries, names):
                data = (SOURCE / entry).read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"Not a valid PNG signature: {entry}")
                width, height = struct.unpack(">II", data[16:24])
                if width < 1 or height < 1:
                    raise ValueError(f"Invalid PNG dimensions: {entry}")
                archive.writestr(filename, data)
        with ZipFile(destination) as verification:
            if verification.testzip() is not None or verification.namelist() != names:
                raise ValueError(f"ZIP verification failed: {destination}")
        lines.append(f"| [{destination.name}](Zips/{destination.name}) | {len(entries)} | " +
                     ", ".join(f"`{name}`" for name in names) + " |")
        print(f"{destination.name}: {len(entries)} original icons")

    (ROOT / "IconPacks" / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"PACKAGING OK — {len(chosen)} icons in {len(packs)} archives")


if __name__ == "__main__":
    main()
