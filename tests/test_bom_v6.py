"""Focused regressions for v6 families, glow preservation and cleanup."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location("bom_v6",Path(__file__).resolve().parents[1]/"src"/"bom_v6.py")
bom=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bom)

class BOMV6Regression(unittest.TestCase):
    def test_five_dark_and_white_shades(self):
        # Every neutral family gets a distinct range, instead of one free black.
        rgb=[]
        for v in (6,15,30,45,70): rgb.append([v,v,v])
        for v in (205,216,229,243,255): rgb.append([v,v,v])
        for v in (14,30,60,85,110): rgb.append([v,1,2]) # saturated dark reds
        x=np.array(rgb,dtype=np.float32)
        groups,black,white,_=bom.split_families(x)
        self.assertIn("black",groups)
        self.assertIn("white",groups)
        self.assertIn("red",groups)
        for name in ("black","white","red"):
            self.assertGreaterEqual(len(bom.representative_palette_for_family(x[groups[name]])),3)
        out,stats=bom.assign_quantized(x)
        self.assertEqual(out.shape,x.shape)
        self.assertGreaterEqual(stats["black"],3)
        self.assertGreaterEqual(stats["white"],3)

    def test_glowing_red_eye_and_original_alpha(self):
        tile=np.zeros((96,96,4),dtype=np.uint8)
        tile[15:81,15:81,:3]=(25,15,28)
        tile[15:81,15:81,3]=255
        tile[40:43,35:38,:3]=(251,23,6)
        tile[40:43,35:38,3]=255
        tile[14,20,:3]=(74,60,67)
        tile[14,20,3]=96
        result,_=bom.process_tile(Image.fromarray(tile,"RGBA"))
        pixels=np.asarray(result)
        self.assertTrue(np.all(pixels[40:43,35:38,0]>=200))
        self.assertEqual(int(pixels[14,20,3]),96)
        self.assertEqual(int(pixels[0,0,3]),0)
        self.assertFalse(np.any(pixels[0,:,:3]))

    def test_synthetic_hood_speck_corrected(self):
        src=np.zeros((96,96,4),dtype=np.uint8)
        src[10:86,10:86,:3]=(74,49,38)
        src[10:86,10:86,3]=255
        result=src.copy()
        result[22,50,:3]=(3,3,3) # invented black pixel on hood
        cleaned=bom.cleanup_micro_specks(result,src)
        self.assertTrue(int(cleaned[22,50,0])>20)
        self.assertTrue(np.array_equal(cleaned[:,:,3],src[:,:,3]))

if __name__=="__main__":
    unittest.main()
