import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ibbit import process_image, fit_256, family_labels


class IbBitTests(unittest.TestCase):
    def test_resize_before_processing(self):
        im = Image.new('RGBA', (600, 120), (145, 90, 40, 255))
        out, report = process_image(im)
        self.assertEqual(out.size, (256, 256))
        self.assertEqual(report['size'], [256, 256])
        self.assertEqual(out.getpixel((0, 0))[3], 0)
        self.assertGreater(out.getpixel((128, 128))[3], 0)

    def test_preserve_alpha_exactly_after_resize(self):
        rng = np.random.default_rng(13)
        a = rng.integers(0, 256, size=(256, 256, 4), dtype=np.uint8)
        original = Image.fromarray(a, 'RGBA')
        output, report = process_image(original)
        np.testing.assert_array_equal(np.asarray(output)[:, :, 3], a[:, :, 3])
        self.assertTrue(report['alpha_identical_after_resize'])

    def test_three_tones_for_each_family(self):
        yy, xx = np.mgrid[0:256, 0:256]
        a = np.zeros((256, 256, 4), dtype=np.uint8)
        a[:, :128, :3] = np.stack((100+xx[:, :128]//8, np.full((256,128),70),
                                  np.full((256,128),45)), axis=-1).astype('uint8')
        a[:, 128:, :3] = np.stack((np.full((256,128),225), np.full((256,128),200),
                                  (30+yy[:, 128:]//8)), axis=-1).astype('uint8')
        a[:, :, 3] = 255
        out, report = process_image(Image.fromarray(a, 'RGBA'))
        self.assertGreaterEqual(len(report['families']), 2)
        for info in report['families'].values():
            self.assertEqual(set(info['tones_rgb']), {'dark', 'neutral', 'light'})

    def test_transparent_sprite_stays_transparent(self):
        a = Image.new('RGBA', (256,256), (0,0,0,0))
        o, r = process_image(a)
        self.assertEqual(np.asarray(o).sum(), 0)
        self.assertEqual(r['families'], {})

    def test_deterministic(self):
        rng = np.random.default_rng(8)
        img = Image.fromarray(rng.integers(0,256,size=(64,32,4),dtype=np.uint8), 'RGBA')
        a,_=process_image(img)
        b,_=process_image(img)
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


if __name__ == '__main__':
    unittest.main()
