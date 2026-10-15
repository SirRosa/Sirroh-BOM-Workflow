"""Regression suite for lossless split/assemble, naming and manifest safety."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_atlas import split_atlas
from assemble_atlas import assemble

SOURCE = Path(__file__).resolve().parents[1] / "output" / "Valheim_Compass_Atlas_BOM.png"


class RoundTripTests(unittest.TestCase):
    def test_complete_real_atlas_roundtrip(self):
        self.assertTrue(SOURCE.exists(), "Verified BOM 1.0/G atlas is required")
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "work"
            rebuilt = work / "reassembled.png"
            m = split_atlas(SOURCE, work)
            self.assertEqual(m["grid"], {"columns": 8, "rows": 9})
            self.assertEqual(len(m["tiles"]), 72)
            self.assertEqual(sum(not t["empty"] for t in m["tiles"]), 71)
            self.assertTrue(m["tiles"][-1]["empty"])
            self.assertEqual(m["tiles"][-1]["id"], "r09_c08")
            result = assemble(work / "atlas_manifest.json", rebuilt, SOURCE)
            self.assertTrue(result["pixel_identical"])
            # Both PNGs are saved by Pillow optimize=True; on this runtime even
            # the encoded bytes should be identical, not only RGBA pixel values.
            with Image.open(SOURCE) as a, Image.open(rebuilt) as b:
                self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
            self.assertEqual(m["atlas"]["sha256_rgba"], result["reassembled_sha256_rgba"])
            self.assertTrue((work / "tiles" / "r07_c02.png").exists())

    def test_names_and_prefabs_editable_without_affecting_positions(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "synthetic.png"
            atlas = Image.new("RGBA", (192, 96), (0, 0, 0, 0))
            atlas.paste((255, 30, 10, 255), (0, 0, 96, 96))
            atlas.paste((2, 40, 80, 255), (96, 0, 192, 96))
            atlas.save(source, optimize=True)
            root = Path(td) / "split"
            m = split_atlas(source, root)
            m["tiles"][0]["name"] = "Czaszka jelenia"
            m["tiles"][0]["prefab"] = "DeerTrophy"
            # JSON entry ordering is irrelevant: coordinates determine location.
            m["tiles"].reverse()
            (root / "atlas_manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            result = assemble(root / "atlas_manifest.json", root / "rebuilt.png", source)
            self.assertTrue(result["pixel_identical"])
            # Re-running split keeps user-assigned names and prefab values.
            again = split_atlas(source, root)
            self.assertEqual(again["tiles"][0]["name"], "Czaszka jelenia")
            self.assertEqual(again["tiles"][0]["prefab"], "DeerTrophy")
            self.assertEqual(again["tiles"][0]["id"], "r01_c01")

    def test_rejects_unexpected_cells_and_missing_or_altered_tiles(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "simple.png"
            Image.new("RGBA", (192, 96), (34, 56, 78, 255)).save(source)
            root = Path(td) / "split"
            manifest = split_atlas(source, root)
            path = root / "atlas_manifest.json"
            manifest["tiles"][1]["row"] = manifest["tiles"][0]["row"]
            manifest["tiles"][1]["column"] = manifest["tiles"][0]["column"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                assemble(path, root / "rebuilt.png")
            manifest = split_atlas(source, root)
            tile_path = root / "tiles" / "r01_c01.png"
            with Image.open(tile_path) as im:
                altered = im.copy()
            altered.putpixel((0, 0), (12, 12, 12, 255))
            altered.save(tile_path)
            with self.assertRaisesRegex(ValueError, "Tile altered"):
                assemble(path, root / "rebuilt.png")
            out = assemble(path, root / "modified.png", allow_modified=True)
            self.assertFalse(out["pixel_identical"])
            self.assertTrue(out["modified_tiles_detected"])
            tile_path.unlink()
            with self.assertRaises(FileNotFoundError):
                assemble(path, root / "rebuilt.png")

    def test_fails_if_atlas_size_is_not_multiple_of_tile_size(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "bad.png"
            Image.new("RGBA", (97, 96)).save(src)
            with self.assertRaisesRegex(ValueError, "divided evenly"):
                split_atlas(src, Path(td) / "tiles")


if __name__ == "__main__":
    unittest.main()
