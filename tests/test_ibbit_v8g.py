import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ibbit_v8g import (PALETTE, fit_rgba, compute_thresholds, quantize,
                       cleanup_color_islands, prepared_lightness, render, run_batch)


class IbBitV8GTests(unittest.TestCase):
    def test_exact_palette_from_v8(self):
        np.testing.assert_array_equal(PALETTE, [[34, 37, 34], [58, 41, 26], [72, 82, 70]])

    def test_three_ordered_tones(self):
        luma = PALETTE.astype(float) @ np.array([.2126, .7152, .0722])
        self.assertTrue(luma[0] < luma[1] < luma[2])

    def test_fit_256_preprocessing(self):
        image = Image.new('RGBA', (500, 100), (120, 85, 30, 255))
        output = fit_rgba(image)
        self.assertEqual(output.size, (256, 256))
        self.assertEqual(output.getpixel((0, 0))[3], 0)
        self.assertEqual(output.getpixel((128, 128))[3], 255)

    def test_alpha_preservation(self):
        arr = np.zeros((35, 45, 4), dtype=np.uint8)
        arr[..., 0:3] = [87, 64, 44]
        arr[..., 3] = np.arange(45)[None] * 5
        image = Image.fromarray(arr, 'RGBA')
        light, visible, alpha = prepared_lightness(image)
        output, meta = render(light, visible, alpha, (10., 50.))
        self.assertTrue(meta['alpha_preserved'])
        np.testing.assert_array_equal(np.asarray(output)[..., 3], alpha)

    def test_only_three_opaque_colors(self):
        rng = np.random.default_rng(123)
        a = rng.integers(0, 256, size=(256, 256, 4), dtype=np.uint8)
        a[..., 3] = 255
        light, vis, alpha = prepared_lightness(Image.fromarray(a, 'RGBA'))
        output, _ = render(light,vis,alpha,(25.,70.))
        unique = np.unique(np.asarray(output)[..., :3].reshape(-1, 3), axis=0)
        self.assertTrue(all(any(np.array_equal(c, p) for p in PALETTE) for c in unique))

    def test_majority_brown_quantiles(self):
        lum = np.arange(100, dtype=np.float32).reshape(10, 10)
        vis = np.ones((10, 10), bool)
        low, high = compute_thresholds([lum], [vis], .12, .12)
        levels = quantize(lum, vis, (low, high))
        self.assertEqual(int((levels == 0).sum()), 12)
        self.assertEqual(int((levels == 1).sum()), 76)
        self.assertEqual(int((levels == 2).sum()), 12)

    def test_small_island_below_16_removed(self):
        levels = np.ones((40, 40), np.uint8)
        levels[12:15,12:15] = 0
        vis = np.ones((40,40), bool)
        clean, meta = cleanup_color_islands(levels, vis)
        self.assertTrue(np.all(clean == 1))
        self.assertGreater(meta['pixels_reassigned'], 0)

    def test_island_equal_16_preserved(self):
        levels = np.ones((40,40), np.uint8)
        levels[12:16,12:16] = 0
        clean, _ = cleanup_color_islands(levels, np.ones((40,40), bool))
        self.assertEqual(int((clean == 0).sum()), 16)

    def test_more_shared_boundary_wins(self):
        levels = np.zeros((35,35), dtype=np.uint8)
        levels[:,15:] = 2
        levels[10:13, 12:15] = 1  # 9 px island touches mainly color 0
        result,_ = cleanup_color_islands(levels,np.ones_like(levels,dtype=bool))
        self.assertTrue(np.all(result[10:13,12:15] == 0))

    def test_batch_shared_palette(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); inp=root/'in'; out=root/'out';inp.mkdir()
            im=np.zeros((256,256,4),np.uint8)
            for i in range(2):
                im[:,:,:3] = 30+i*20
                im[:,:,3]=255
                Image.fromarray(im,'RGBA').save(inp/f'{i}.png')
            report=run_batch(inp,out)
            self.assertEqual(report['count'],2)
            self.assertEqual(len(report['global_thresholds_lab_l']),2)
            self.assertEqual(len(list(out.glob('*.png'))),2)


if __name__ == '__main__':
    unittest.main()