import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bom_icons import detect_grid, process_tile, process, normalize_icon_to_tile, quantize, TILE_SIZE, LAYOUT_MARGIN


class B0MTests(unittest.TestCase):
    def test_detect_grid_uneven_padding(self):
        arr = np.zeros((360, 320, 4), dtype=np.uint8)
        for row in range(9):
            for col in range(8):
                y = row * 40 + 6
                x = col * 40 + 7
                arr[y:y+26, x:x+24] = [120, 60, 50, 255]
        c, r, xs, ys, _ = detect_grid(arr)
        self.assertEqual((c, r), (8, 9))
        self.assertEqual((len(xs), len(ys)), (9, 10))

    def test_tile_alpha_colors(self):
        t = np.zeros((110, 110, 4), dtype=np.uint8)
        t[14:90, 12:94] = [142, 58, 31, 255]
        out, n = process_tile(normalize_icon_to_tile(t), colors=6, seed=7)
        self.assertEqual(out.shape, (96, 96, 4))
        self.assertLessEqual(n, 6)
        self.assertTrue(set(np.unique(out[:,:,3])).issubset({0,255}))

    def test_bottom_anchor_and_horizontal_center(self):
        for w, h in [(27, 53), (53, 27), (120, 220), (220, 120)]:
            src = np.zeros((h + 12, w + 16, 4), dtype=np.uint8)
            src[6:6+h, 8:8+w] = [140, 90, 60, 255]
            out = normalize_icon_to_tile(src)
            ys, xs = np.where(out[:, :, 3] >= 32)
            self.assertGreater(len(xs), 0)
            self.assertEqual(TILE_SIZE - 1 - int(ys.max()), LAYOUT_MARGIN)
            self.assertLessEqual(abs((int(xs.min()) + int(xs.max())) / 2 - (TILE_SIZE - 1) / 2), 0.5)
            self.assertGreaterEqual(int(xs.min()), LAYOUT_MARGIN)
            self.assertLessEqual(int(xs.max()), TILE_SIZE - 1 - LAYOUT_MARGIN)


    def test_perceptual_quantization_no_integer_overflow(self):
        # Previous int16 squared RGB distances overflowed, assigning bright
        # orange highlight pixels to a dark green palette entry.
        rgb = np.array([[[251, 166, 44], [238, 110, 24]]], dtype=np.uint8)
        palette = np.array([[66, 90, 28], [238, 112, 24], [248, 151, 42]], dtype=np.uint8)
        idx = quantize(rgb, np.ones((1, 2), dtype=bool), palette)
        self.assertEqual(int(idx[0, 0]), 2)
        self.assertNotEqual(int(idx[0, 1]), 0)

    def test_cloudberry_warm_highlights_not_green(self):
        # Regression fixture: source sprite at column 8, row 4. In BOM-D,
        # gold/orange berries developed dark green marks after quantization.
        atlas_file = Path(__file__).resolve().parents[1] / 'input' / 'Valheim_Compass_Atlas.png'
        if not atlas_file.exists():
            self.skipTest('source atlas fixture is not available')
        from bom_icons import extract_objects_by_slot
        original = np.asarray(Image.open(atlas_file).convert('RGBA'))
        cols, rows, xs, ys, _ = detect_grid(original)
        self.assertEqual((cols, rows), (8, 9))
        slots, _ = extract_objects_by_slot(original, xs, ys, cols, rows)
        before = normalize_icon_to_tile(slots[3][7])
        after, used = process_tile(before, colors=6, seed=100 + 3 * 8 + 7)
        warm = (before[:, :, 3] >= 32) & (before[:, :, 0] >= 190) & (before[:, :, 0] >= 1.4 * before[:, :, 1])
        highlights = warm & (before[:, :, 1] >= 125)
        self.assertGreater(int(np.count_nonzero(highlights)), 20)
        self.assertEqual(used, 6)
        # Every warm berry pixel must remain reddish/orange, not leaf-green.
        warm_rgb = after[:, :, :3][warm].astype(int)
        self.assertTrue(np.all(warm_rgb[:, 0] >= 1.5 * warm_rgb[:, 1]))
        self.assertTrue(np.all(warm_rgb[:, 0] > warm_rgb[:, 2]))
        # Bright original highlights must retain a visibly lighter shade.
        highlight_rgb = after[:, :, :3][highlights].astype(int)
        self.assertGreater(float(np.mean(highlight_rgb[:, 1])), 100.0)

    def test_grid_output_alignment_and_color_limit(self):
        source_file = Path(__file__).resolve().parents[1] / 'input' / 'Valheim_Compass_Atlas.png'
        if not source_file.exists():
            self.skipTest('source atlas fixture is not available')
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / 'atlas_bom.png'
            report = process(source_file, result)
            self.assertEqual(report['grid'], [8, 9])
            self.assertEqual(report['slot_extraction']['nonempty_slots'], 71)
            data = np.asarray(Image.open(result).convert('RGBA'))
            self.assertEqual(data.shape, (864, 768, 4))
            for row in range(9):
                for col in range(8):
                    cell = data[row*96:(row+1)*96, col*96:(col+1)*96]
                    opaque = cell[:,:,3] > 0
                    self.assertLessEqual(len(np.unique(cell[opaque,:3],axis=0)), 6)
                    self.assertTrue(set(np.unique(cell[:,:,3])).issubset({0,255}))
                    if np.any(opaque):
                        ys, xs = np.where(opaque)
                        self.assertEqual(95 - int(ys.max()), 2)
                        self.assertGreaterEqual(int(xs.min()), 2)
                        self.assertLessEqual(int(xs.max()), 93)
            self.assertEqual(int(np.count_nonzero(data[8*96:9*96, 7*96:8*96, 3])), 0)


if __name__ == '__main__':
    unittest.main()
