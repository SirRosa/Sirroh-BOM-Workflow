#!/usr/bin/env python3
"""Stable categories for the non-creature BOM collection.

Keep original icon file names. Place processed 96x96 PNGs into category
directories, persist a machine-readable index, and render grouped previews.
Unknown future names go into To_Review instead of being silently mislabelled.
"""
from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CATEGORY_ORDER = (
    "Locations", "Fish", "Plants", "Objects_POI", "Items", "Custom", "To_Review"
)
CATEGORY_LABELS = {
    "Locations": "Lokacje / wejścia / punkty świata",
    "Fish": "Ryby",
    "Plants": "Rośliny / grzyby / owoce / uprawy",
    "Objects_POI": "POI obiektów / konstrukcje / skrzynie / ule",
    "Items": "Przedmioty / surowce / trofea",
    "Custom": "Ikony customowe",
    "To_Review": "Do ręcznego przypisania (nowe ikony)",
}
CATEGORY_ITEMS = {
    "Locations": """
        BearCave CharredFortress DG_Cave DG_ForestCrypt DraugrVillage
        ForgeOfPotential FulingVillage Hildir_cave Hildir_crypt
        Hildir_plainsfortress Mistlands_DvergrBossEntrance1
        Mistlands_DvergrTownEntrance1 MistlandsExcavation
        MistlandsGuardTower MistlandsHarbour MistlandsLighthouse
        MorgenHole1 MorkBorg MountainCave02 PlaceofMystery1
        PlaceofMystery2 PlaceofMystery3 SacrificialStones
        SunkenCrypt4 TheHole01 TrollCave
    """.split(),
    "Fish": """
        Fish1 Fish2 Fish3 Fish4_cave Fish5 Fish6 Fish7 Fish8
        Fish9 Fish10 Fish11 Fish12
    """.split(),
    "Plants": """
        Barley Beech1 Beech_Sapling Beech_Stub Blueberries Carrot
        Cloudberry Dandelion Fiddleheadfern FirTree Flax Kale
        Lingonberry Mushroom MushroomJotunPuffs MushroomMagecap
        MushroomSmokePuff MushroomYellow Oat Onion Poteitr
        Pukeberries Raspberry Thistle Turnip Vineberry
    """.split(),
    "Objects_POI": """
        dvergrprops_crate_long fire_pit forge_ext4 piece_beehive
        piece_cauldron piece_chest_wood Player_tombstone rock4_copper
        stake_wall stone_arch stone_pile VikingShip wood_door
    """.split(),
    "Items": """
        ChickenEgg Crystal Honey ShieldWood Stone Torch
        TrophyDeer TrophySkeleton
    """.split(),
}
NAME_CATEGORY = {}
for cat, names in CATEGORY_ITEMS.items():
    for name in names:
        key = name.casefold()
        if key in NAME_CATEGORY:
            raise AssertionError(f"Duplicated mapping: {name}")
        NAME_CATEGORY[key] = cat


def classify(name: str) -> str:
    """Classification takes the prefab/file stem, not a random input location."""
    if name.lower().startswith("custom_"):
        return "Custom"
    return NAME_CATEGORY.get(name.casefold(), "To_Review")


def order_key(name: str):
    cat = classify(name)
    return (CATEGORY_ORDER.index(cat), name.casefold(), name)


def _font(size: int, bold: bool = False):
    files = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
    )
    for filename in files:
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _lines(draw, text: str, font, width: int, limit: int = 3):
    text = text.removesuffix(".png")
    # Prefer natural breaks around underscores; also split very long words.
    words = []
    for component in text.split("_"):
        if words and component:
            words[-1] += "_" + component
        else:
            words.append(component)
        if draw.textbbox((0, 0), words[-1], font=font)[2] > width:
            words[-1] = words[-1].removesuffix("_")
            words.append(component)
    result = []
    for word in words:
        if not word:
            continue
        if draw.textbbox((0, 0), word, font=font)[2] <= width:
            result.append(word)
        else:
            while word:
                size = len(word)
                while size > 1 and draw.textbbox((0, 0), word[:size], font=font)[2] > width:
                    size -= 1
                result.append(word[:size])
                word = word[size:]
    if len(result) > limit:
        result = result[:limit]
        while result[-1] and draw.textbbox((0, 0), result[-1]+"…", font=font)[2] > width:
            result[-1] = result[-1][:-1]
        result[-1] += "…"
    return result


def _panel(result: Path, category: str, filenames: list[str]) -> Image.Image:
    cols, cell_w, cell_h, gap, margin = 8, 178, 158, 7, 24
    width = 2*margin + cols*cell_w + (cols-1)*gap
    nrows = math.ceil(len(filenames) / cols)
    height = 82 + nrows*(cell_h+gap) + 18
    canvas = Image.new("RGB", (width, height), (26, 30, 33))
    draw = ImageDraw.Draw(canvas)
    title = f"{CATEGORY_LABELS[category]}  —  {len(filenames)}"
    draw.text((margin, 14), title, font=_font(23, True), fill=(240, 238, 219))
    draw.line((margin, 53, width-margin, 53), fill=(91, 107, 107), width=2)
    titlefont = _font(13)
    for i, filename in enumerate(filenames):
        row, column = divmod(i, cols)
        x, y = margin+column*(cell_w+gap), 70+row*(cell_h+gap)
        draw.rounded_rectangle((x,y,x+cell_w-1,y+cell_h-1),radius=8,
                               fill=(42,48,52),outline=(83,97,100),width=1)
        with Image.open(result/"icons"/category/filename) as img:
            icon = img.convert("RGBA")
            if icon.size != (96,96):
                raise ValueError(f"Wrong dimensions for {filename}: {icon.size}")
            canvas.paste(icon,(x+(cell_w-96)//2,y+4),icon)
        lines = _lines(draw, filename, titlefont, cell_w-12, 3)
        for n,line in enumerate(lines):
            bb = draw.textbbox((0,0),line,font=titlefont)
            draw.text((x+(cell_w-(bb[2]-bb[0]))//2, y+105+n*16),
                      line, font=titlefont, fill=(222,227,228))
    return canvas


def organize_collection(result: Path) -> dict:
    result = Path(result)
    icon_dir = result/"icons"
    summary_path = result/"summary.json"
    manifest_path = result/"atlas_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = summary["file_names"]
    if len(expected) != len(set(expected)):
        raise AssertionError("Duplicate file names in collection")
    groups = {cat: [] for cat in CATEGORY_ORDER}
    for filename in expected:
        category = classify(Path(filename).stem)
        groups[category].append(filename)
    for files in groups.values():
        files.sort(key=lambda name: (name.casefold(), name))

    # File relocation within the generated output. No source originals are moved.
    for cat, filenames in groups.items():
        if not filenames:
            continue
        folder = icon_dir/cat
        folder.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            old = icon_dir/filename
            new = folder/filename
            if not old.is_file():
                raise FileNotFoundError(old)
            if new.exists():
                raise FileExistsError(new)
            shutil.move(old, new)
    stray = [p for p in icon_dir.rglob("*.png")
             if p.parent.name not in CATEGORY_ORDER]
    if stray:
        raise AssertionError(f"Unsafely uncategorized files: {stray}")

    for tile in manifest["tiles"]:
        tile["category"] = classify(tile["prefab"]) if tile["prefab"] else None
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    index = {
        "schema": "sirroh.bom.categories.v1",
        "total_icons": len(expected),
        "legacy_excluded": True,
        "creatures_excluded": True,
        "categories": [
            {"id": cat, "label": CATEGORY_LABELS[cat],
             "count": len(files),
             "files": [f"icons/{cat}/{name}" for name in files]}
            for cat, files in groups.items() if files
        ],
    }
    (result/"categories.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    previews = result/"previews"
    previews.mkdir(exist_ok=True)
    panels = []
    for category, filenames in groups.items():
        if not filenames:
            continue
        panel = _panel(result, category, filenames)
        panel.save(previews/f"{category}.png", "PNG", optimize=True)
        panels.append(panel)
    if not panels:
        raise AssertionError("No preview panels")
    overview = Image.new("RGB", (max(im.width for im in panels),
                                 62+sum(im.height for im in panels)), (20,24,26))
    d = ImageDraw.Draw(overview)
    d.text((24, 13), "SIRROH BOM V8  |  KATALOG IKON", font=_font(26,True),
           fill=(243,236,216))
    d.text((24, 43), f"{len(expected)} ikon  |  96 x 96 px  |  bez legacy i stworzen",
           font=_font(12),fill=(178,197,200))
    pos = 62
    for panel in panels:
        overview.paste(panel, (0, pos))
        pos += panel.height
    overview.save(result/"preview_grouped.png", "PNG", optimize=True)

    source_cats = [classify(Path(n).stem) for n in expected]
    if source_cats != sorted(source_cats, key=CATEGORY_ORDER.index):
        raise AssertionError("Atlas source order was not categorized")
    summary["category_counts"] = {k:len(v) for k,v in groups.items() if v}
    summary["category_paths"] = [f"icons/{classify(Path(n).stem)}/{n}" for n in expected]
    summary["sorted_atlas"] = True
    summary["categories_index"] = "categories.json"
    summary["grouped_preview"] = "preview_grouped.png"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")

    readme = [
        "# BOM v8 — posortowana kolekcja ikon",
        "",
        f"{len(expected)} ikon 96×96 PNG po BOM v8. Zachowano nazwy prefabów i plików.",
        "Nie uwzględniono `legacy` ani samych stworzeń (trofea są przedmiotami).",
        "Oryginalne ikony wejściowe pozostają bez zmian.",
        "",
        "## Kategorie",
        "",
        "| Katalog | Liczba | Zawartość |",
        "|---|---:|---|",
    ]
    for cat, items in groups.items():
        if items:
            readme.append(f"| `icons/{cat}/` | {len(items)} | {CATEGORY_LABELS[cat]} |")
    readme.extend([
        "",
        "## Podgląd i indeks",
        "",
        "- `preview_grouped.png`: wszystkie ikony w blokach kategorii z podpisami.",
        "- `previews/`: osobne podglądy każdej kategorii.",
        "- `atlas_BOM_v8.png`: atlas gry; układ według kategorii.",
        "- `atlas_manifest.json`: pozycja każdego kafelka i pole `category`.",
        "- `categories.json`: maszynowy indeks prefabów i ścieżek.",
        "- `verification.json`: potwierdzenie identycznego składania kafelków.",
        "",
        "## Wykaz plików",
    ])
    for cat, filenames in groups.items():
        if filenames:
            readme.extend(["",f"### {CATEGORY_LABELS[cat]} (`{cat}`)", ""])
            readme.extend(f"- `{filename}`" for filename in filenames)
    (result/"README.md").write_text("\n".join(readme)+"\n",encoding="utf-8")
    if sum(len(files) for files in groups.values()) != len(expected):
        raise AssertionError("Some icon was lost during categorization")
    print("CATEGORIES OK:", json.dumps(summary["category_counts"],ensure_ascii=False),
          "| grouped preview:", result/"preview_grouped.png", flush=True)
    return index


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, default=Path("collection_results"))
    args=p.parse_args()
    organize_collection(args.result_dir)
