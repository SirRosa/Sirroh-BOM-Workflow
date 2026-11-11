from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ibbit import (PALETTE, FAMILIES, NEUTRAL_ID, BROWN_ID,
                      perceived_brightness, fit_256, family_labels,
                      process_image, cleanup_small_color_blobs, run_batch,
                      validate_palette)


class IbBitV7Tests(unittest.TestCase):
    def test_palette_strictly_ordered_and_black_is_global_darkest(self):
        validate_palette()
        black = perceived_brightness(PALETTE['neutral'][0])
        self.assertEqual(PALETTE['neutral'][0], (34,37,34))
        for family, triplet in PALETTE.items():
            self.assertEqual(len(triplet), 3)
            lum = perceived_brightness(triplet)
            self.assertLess(lum[0], lum[1])
            self.assertLess(lum[1], lum[2])
            if family != 'neutral':
                self.assertGreater(lum.min(), black)

    def test_grayscale_three_tones_are_preserved(self):
        yy, xx = np.mgrid[0:256, 0:256]
        gray = np.zeros((256,256,4), np.uint8)
        # 3 broad bands; darkest is very dark, all are completely unsaturated.
        gray[:,:,3] = 255
        gray[:85,:,:3] = 15
        gray[85:170,:,:3] = 49
        gray[170:,:,:3] = 113
        out, report = process_image(Image.fromarray(gray, 'RGBA'))
        self.assertEqual(set(report['families']), {'neutral'})
        used = np.unique(np.asarray(out)[:,:,:3].reshape(-1,3), axis=0)
        self.assertEqual(len(used), 3)
        for c in PALETTE['neutral']:
            self.assertTrue(any(np.array_equal(c, k) for k in used))
        self.assertTrue(all(c > 100 for c in report['families']['neutral']['final_tone_pixels']))

    def test_brown_three_tones_also_preserved(self):
        pixels = np.zeros((256,256,4), np.uint8)
        pixels[:,:,3] = 255
        pixels[:85,:,:3] = (52,33,20)
        pixels[85:170,:,:3] = (115,76,44)
        pixels[170:,:,:3] = (170,113,71)
        out, report = process_image(Image.fromarray(pixels,'RGBA'))
        self.assertIn('brown', report['families'])
        self.assertTrue(all(c > 100 for c in report['families']['brown']['final_tone_pixels']))
        self.assertTrue(report['alpha_preserved'])

    def test_nearly_black_source_does_not_turn_bright(self):
        pixels = np.zeros((256, 256, 4), np.uint8)
        pixels[:, :, 3] = 255
        pixels[:85, :, :3] = 2
        pixels[85:170, :, :3] = 5
        pixels[170:, :, :3] = 11
        out, _ = process_image(Image.fromarray(pixels, 'RGBA'))
        colors = set(map(tuple, np.unique(np.asarray(out)[:, :, :3].reshape(-1, 3), axis=0)))
        self.assertNotIn(PALETTE['neutral'][2], colors)
        self.assertEqual(colors, {PALETTE['neutral'][0]})

    def test_neutral_gray_has_no_rainbow(self):
        a = np.full((256,256,4), 255, np.uint8)
        for y in range(256):
            a[y,:,0:3] = (y//3,y//3,y//3)
        out, report = process_image(Image.fromarray(a,'RGBA'))
        self.assertEqual(set(report['families']), {'neutral'})
        actual = set(map(tuple,np.unique(np.asarray(out)[:,:,:3].reshape(-1,3),axis=0)))
        self.assertLessEqual(len(actual), 3)
        self.assertTrue(actual.issubset(set(PALETTE['neutral'])))

    def test_artifact_15px_merges_and_16px_survives(self):
        ids = np.full((40,40), 11*3+1, np.int32)
        vis = np.ones((40,40), bool)
        ids[4:7,4:9] = BROWN_ID*3+0  # 15px
        ids[20:24,20:24] = BROWN_ID*3+0  # 16px
        result, stats=cleanup_small_color_blobs(ids,vis,min_area=16)
        self.assertTrue(np.all(result[4:7,4:9] == 11*3+1))
        self.assertTrue(np.all(result[20:24,20:24] == BROWN_ID*3+0))
        self.assertGreaterEqual(stats['removed_pixels'], 15)

    def test_tie_break_deterministic(self):
        ids = np.full((30,30),11*3+0,np.int32)
        ids[:,15:]=BROWN_ID*3+0
        ids[10:12,14:16]=11*3+2
        a,_=cleanup_small_color_blobs(ids,np.ones((30,30),bool))
        b,_=cleanup_small_color_blobs(ids,np.ones((30,30),bool))
        np.testing.assert_array_equal(a,b)

    def test_alpha_preserved_exactly(self):
        rng = np.random.default_rng(4)
        img = Image.fromarray(rng.integers(0,256,size=(256,256,4),dtype=np.uint8),'RGBA')
        result, report=process_image(img)
        np.testing.assert_array_equal(np.asarray(img)[:,:,3],np.asarray(result)[:,:,3])
        self.assertTrue(report['alpha_preserved'])

    def test_resize_before_processing_and_aspect(self):
        src = Image.new('RGBA',(600,100),(100,100,100,255))
        output, report=process_image(src)
        self.assertEqual(output.size,(256,256))
        self.assertEqual(output.getpixel((0,0))[3],0)
        self.assertEqual(output.getpixel((128,128))[3],255)
        self.assertEqual(sum(output.getchannel('A').getextrema()),255)

    def test_empty_image(self):
        output,report=process_image(Image.new('RGBA',(256,256)))
        self.assertEqual(report['families'],{})
        self.assertEqual(np.asarray(output).sum(),0)

    def test_batch_preserves_relative_path(self):
        with TemporaryDirectory() as td:
            inp=Path(td)/'input';out=Path(td)/'output'; (inp/'nested').mkdir(parents=True)
            Image.new('RGBA',(100,64),(140,140,140,255)).save(inp/'nested'/'test.png')
            report=run_batch(inp,out)
            self.assertEqual(report['processed'],1)
            with Image.open(out/'nested'/'test.png') as result:
                self.assertEqual(result.size,(256,256))
            self.assertTrue((out/'ibbit_report.json').exists())


    def test_v7_palette_darker_and_less_saturated_than_v6(self):
        import cv2
        old = {
            'red': ((75,34,34),(114,67,61),(159,108,99)),
            'orange': ((69,45,28),(113,78,48),(166,125,80)),
            'yellow': ((63,62,37),(118,110,56),(168,158,95)),
            'olive': ((54,62,38),(91,103,61),(135,142,91)),
            'green': ((47,63,46),(77,98,70),(118,137,105)),
            'teal': ((42,63,58),(70,99,91),(111,137,125)),
            'cyan': ((46,60,64),(78,102,106),(116,140,143)),
            'blue': ((47,57,68),(81,96,108),(122,132,145)),
            'violet': ((58,50,64),(96,84,106),(140,126,146)),
            'magenta': ((71,42,53),(111,77,88),(156,115,127)),
            'brown': ((58,41,26),(96,74,49),(148,118,82)),
            'neutral': ((34,37,34),(91,99,90),(155,166,153)),
        }
        old_rgb = np.asarray([old[k] for k in FAMILIES], np.uint8)
        new_rgb = np.asarray([PALETTE[k] for k in FAMILIES], np.uint8)
        self.assertTrue(np.all(perceived_brightness(new_rgb) <= perceived_brightness(old_rgb) + 0.1))
        self.assertGreater(float(np.mean(perceived_brightness(old_rgb) - perceived_brightness(new_rgb))), 10.0)
        old_sat = cv2.cvtColor(old_rgb.reshape(-1,1,3).astype(np.float32)/255.0, cv2.COLOR_RGB2HSV)[...,1].mean()
        new_sat = cv2.cvtColor(new_rgb.reshape(-1,1,3).astype(np.float32)/255.0, cv2.COLOR_RGB2HSV)[...,1].mean()
        self.assertLess(float(new_sat), float(old_sat))

    def test_full_collection_30_has_neutral_all_three_and_no_small_islands(self):
        import os
        inp=Path(os.environ.get('IBBIT_TEST_ASSET_DIR', '/nonexistent/ibbit_sample_assets'))
        if not inp.is_dir():
            self.skipTest('demo collection not present')
        paths=list(inp.rglob('*.png'))
        self.assertEqual(len(paths),30)
        for path in paths:
            with self.subTest(path=str(path)):
                out,rep=process_image(Image.open(path))
                self.assertTrue(rep['alpha_preserved'])
                palette_set={tuple(c) for family in PALETTE.values() for c in family}
                out_np=np.asarray(out)
                unique = np.unique(out_np[:,:,:3][out_np[:,:,3]>=8], axis=0)
                self.assertTrue(set(map(tuple,unique)).issubset(palette_set))
                self.assertLessEqual(rep['cleanup']['components_after'],1)  # only detached alpha islands can remain
        first=Image.open(inp/'Sciany_Szybu_GrimDark_256x256'/'01_Czysta_Sciana.png')
        result, rep=process_image(first)
        self.assertTrue(all(n > 100 for n in rep['families']['neutral']['final_tone_pixels']))

if __name__=='__main__': unittest.main()