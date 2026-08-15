import unittest

import numpy as np

import panorama_stitcher as stitcher


class PanoramaMemoryTests(unittest.TestCase):
    def test_single_band_pyramids_reuse_input_buffers(self):
        image = np.zeros((16, 16, 3), dtype=np.float32)
        weight = np.ones((16, 16), dtype=np.float32)

        laplacian = stitcher.build_laplacian_pyramid(image, 1)
        gaussian = stitcher.build_gaussian_pyramid(weight, 1)

        self.assertTrue(np.shares_memory(laplacian[0], image))
        self.assertTrue(np.shares_memory(gaussian[0], weight))

    def test_weight_normalization_is_in_place_and_sums_to_one(self):
        first = np.ones((8, 8), dtype=np.float32)
        second = np.full((8, 8), 3.0, dtype=np.float32)
        weights = [first, second]

        normalized = stitcher.normalize_weight_maps_in_place(weights)

        self.assertIs(normalized, weights)
        self.assertIs(normalized[0], first)
        self.assertIs(normalized[1], second)
        np.testing.assert_allclose(normalized[0] + normalized[1], 1.0)

    def test_multiband_laplacian_reuses_scratch_and_reconstructs_input(self):
        rng = np.random.default_rng(7)
        original = rng.random((32, 32, 3), dtype=np.float32)
        scratch = original.copy()

        pyramid = stitcher.build_laplacian_pyramid(scratch, 4)
        reconstructed = stitcher.reconstruct_from_pyramid(pyramid)

        self.assertTrue(np.shares_memory(pyramid[0], scratch))
        np.testing.assert_allclose(reconstructed, original, atol=1e-5)

    def test_weighted_pyramid_accumulation_reuses_laplacian_buffers(self):
        first_lap = [np.ones((8, 8, 3), dtype=np.float32)]
        first_gau = [np.full((8, 8), 0.25, dtype=np.float32)]

        blended = stitcher.accumulate_weighted_pyramid(None, first_lap, first_gau)

        self.assertIs(blended, first_lap)
        self.assertIs(blended[0], first_lap[0])
        np.testing.assert_allclose(blended[0], 0.25)

        second_lap = [np.full((8, 8, 3), 2.0, dtype=np.float32)]
        second_gau = [np.full((8, 8), 0.5, dtype=np.float32)]
        same_blended = stitcher.accumulate_weighted_pyramid(blended, second_lap, second_gau)

        self.assertIs(same_blended, blended)
        np.testing.assert_allclose(same_blended[0], 1.25)

    def test_canvas_budget_accepts_working_candidate_and_rejects_pathological_one(self):
        self.assertTrue(stitcher.is_canvas_within_memory_budget((1924, 4349)))
        self.assertFalse(stitcher.is_canvas_within_memory_budget((3829, 7574)))


if __name__ == "__main__":
    unittest.main()
