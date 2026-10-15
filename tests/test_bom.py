import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bom_icons import detect_grid, process_tile, process, normalize_icon_to_tile, TILE_SIZE, LAYOUT_MARGIN


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



if __name__ == '__main__':
    unittest.main()
