import sys
from pathlib import Path
import unittest
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bom_icons import detect_grid, process_tile, process, normalize_icon_to_tile, quantize, hue_family, TILE_SIZE, LAYOUT_MARGIN


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
        # Test ALL saturated warm pixels, not just warm highlights. BOM-E
        # previously passed the highlight-only check while leaving 28 warm
        # input pixels mapped to green leaf colors (and the user could see it).
        source_families = hue_family(before[:, :, :3])
        output_families = hue_family(after[:, :, :3])
        occupied = before[:, :, 3] >= 32
        all_warm = (source_families == 1) & occupied
        all_green = (source_families == 2) & occupied
        self.assertGreater(int(np.count_nonzero(all_warm)), 2500)
        self.assertGreater(int(np.count_nonzero(all_green)), 1000)
        self.assertEqual(int(np.count_nonzero(all_warm & (output_families == 2))), 0,
                         'Source warm fruit pixels were recolored green')
        self.assertEqual(int(np.count_nonzero(all_green & (output_families == 1))), 0,
                         'Source leaf-green pixels were recolored orange')

        # Form the fruit body from source orange. Its bright berry region may
        # NEVER have green pixels, including any dark areas between highlights.
        berry_seed = (before[:, :, 0].astype(int) > 145) & (
            before[:, :, 0].astype(int) > before[:, :, 1].astype(int) * 1.45
        ) & occupied
        berry_closed = cv2.morphologyEx(berry_seed.astype(np.uint8),
                                        cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(berry_closed, 8)
        main = max(range(1, num), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
        body = labels == main
        self.assertGreater(int(np.count_nonzero(body)), 2500)
        self.assertEqual(int(np.count_nonzero(body & occupied & (output_families == 2))), 0,
                         'Green marks remain inside orange cloudberry body')

        # Highlight must remain visibly lighter than base orange and leaves.
        brightest = occupied & (before[:, :, 0] >= 240) & (before[:, :, 1] >= 158)
        self.assertGreater(int(np.count_nonzero(brightest)), 15)
        self.assertGreater(float(np.mean(after[:, :, 1][brightest])), 135)
        self.assertEqual(used, 6)

    def test_dark_wood_cannot_turn_into_bright_bone(self):
        # Regression: a low-light warm-brown antler pixel must choose an actual
        # brown palette entry, not the pale bone entry forced by hue-family labels.
        source = np.array([[[64, 44, 31], [75, 52, 35], [201, 164, 124]]], dtype=np.uint8)
        palette = np.array([[70, 52, 34], [92, 67, 45], [188, 155, 121]], dtype=np.uint8)
        result = quantize(source, np.ones((1, 3), dtype=bool), palette, source_rgb=source)
        self.assertEqual(result[0].tolist(), [0, 0, 2])

    def test_deer_antlers_no_bone_speckles(self):
        # The deer skull is column 2, row 7 (1-indexed).
        atlas = Path(__file__).resolve().parents[1] / 'input' / 'Valheim_Compass_Atlas.png'
        if not atlas.exists():
            self.skipTest('source atlas fixture is not available')
        from bom_icons import extract_objects_by_slot
        original = np.asarray(Image.open(atlas).convert('RGBA'))
        cols, rows, xb, yb, _ = detect_grid(original)
        slots, _ = extract_objects_by_slot(original, xb, yb, cols, rows)
        source = normalize_icon_to_tile(slots[6][1])
        after, used = process_tile(source, colors=6, seed=100+6*8+1)
        antler = (np.indices(source.shape[:2])[0] < 42) & (source[..., 3] > 0)
        # Dark wood: source luminance never warrants bright-beige output.
        dark = antler & (source[..., 0] < 115)
        bright_mismatch = dark & (after[..., 0] > 145) & (after[..., 1] > 95)
        self.assertGreater(int(dark.sum()), 400)
        self.assertEqual(int(bright_mismatch.sum()), 0, 'Pale bone speckles on dark antlers')
        # Real pale skull must still exist and retain bright pixels.
        bone = (source[..., 3] > 0) & (source[..., 0] > 180) & (np.indices(source.shape[:2])[0] >= 42)
        self.assertGreater(int(bone.sum()), 250)
        self.assertGreater(float(after[..., 0][bone].mean()), 155.0)
        self.assertLessEqual(used, 6)

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
