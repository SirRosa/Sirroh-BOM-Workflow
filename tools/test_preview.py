"""Regression tests for read-only labeled atlas preview and wiki-ID mapping."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from preview_atlas_labels import render_preview

ROOT = Path(__file__).resolve().parents[1] / 'atlas_work' / 'Valheim_Compass_Atlas_BOM'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PreviewTests(unittest.TestCase):
    def test_preview_is_read_only_and_has_grid(self):
        manifest = ROOT / 'atlas_manifest.json'
        original = ROOT / 'tiles' / 'r09_c01.png'
        before = (digest(manifest), digest(original))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'preview.png'
            render_preview(manifest, dest)
            with Image.open(dest) as img:
                self.assertEqual(img.size, (1539, 1539))
                self.assertEqual(img.mode, 'RGB')
                self.assertNotEqual(img.getpixel((5, 5)), (0, 0, 0))
        self.assertEqual(before, (digest(manifest), digest(original)))

    def test_wiki_dungeon_id_and_integrity(self):
        manifest = json.loads((ROOT / 'atlas_manifest.json').read_text())
        items = {x['id']: x for x in manifest['tiles']}
        self.assertEqual(len(items), 72)
        self.assertEqual(items['r09_c01']['name'], 'DG_ForestCrypt')
        self.assertEqual(items['r09_c01']['prefab'], 'DG_ForestCrypt')
        self.assertEqual(items['r09_c03']['name'], 'DG_Cave')
        self.assertEqual(items['r09_c02']['name'], 'SunkenCrypt4')
        self.assertEqual(items['r09_c04']['name'], 'MountainCave02')
        self.assertEqual(items['r09_c08']['empty'], True)
        self.assertEqual(items['r07_c05']['name'], 'piece_cauldron')
        self.assertEqual(items['r06_c04']['prefab'], '')
        self.assertTrue(all(t['file'].endswith(t['id'] + '.png') for t in items.values()))
        self.assertTrue(all(t['x'] == (t['column'] - 1) * 96 and t['y'] == (t['row'] - 1) * 96 for t in items.values()))


if __name__ == '__main__': unittest.main()
