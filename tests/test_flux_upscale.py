"""Tests for Flux MultiDiffusion upscaler."""

import math
import unittest

import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_tiling import (
    LatentTileSpec,
    make_cosine_tile_weight,
    plan_latent_tiles,
    validate_tile_params,
)
from denoise import (
    _pack_latents,
    _unpack_latents,
    _prepare_latent_image_ids,
    _prepare_tile_image_ids,
    _compute_auto_strength,
    _calculate_shift,
)


class TestTileValidation(unittest.TestCase):
    def test_zero_tile_size_raises(self):
        with self.assertRaises(ValueError):
            validate_tile_params(0, 8)

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError):
            validate_tile_params(64, -1)

    def test_overlap_equals_tile_raises(self):
        with self.assertRaises(ValueError):
            validate_tile_params(64, 64)

    def test_valid_params(self):
        validate_tile_params(64, 16)


class TestTilePlanning(unittest.TestCase):
    def test_single_tile(self):
        tiles = plan_latent_tiles(32, 32, tile_size=64, overlap=8)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].h, 32)
        self.assertEqual(tiles[0].w, 32)

    def test_multiple_tiles(self):
        tiles = plan_latent_tiles(128, 128, tile_size=64, overlap=16)
        self.assertGreater(len(tiles), 1)

    def test_tiles_cover_full_area(self):
        h, w = 96, 128
        tiles = plan_latent_tiles(h, w, tile_size=64, overlap=8)
        covered = set()
        for tile in tiles:
            for y in range(tile.y, tile.y + tile.h):
                for x in range(tile.x, tile.x + tile.w):
                    covered.add((y, x))
        for y in range(h):
            for x in range(w):
                self.assertIn((y, x), covered, f"Pixel ({y},{x}) not covered")

    def test_no_tile_exceeds_bounds(self):
        h, w = 100, 120
        tiles = plan_latent_tiles(h, w, tile_size=64, overlap=16)
        for tile in tiles:
            self.assertGreaterEqual(tile.y, 0)
            self.assertGreaterEqual(tile.x, 0)
            self.assertLessEqual(tile.y + tile.h, h)
            self.assertLessEqual(tile.x + tile.w, w)


class TestCosineWeight(unittest.TestCase):
    def test_shape(self):
        w = make_cosine_tile_weight(64, 64, 8, torch.device("cpu"), torch.float32)
        self.assertEqual(w.shape, (1, 1, 64, 64))

    def test_boundary_aware_top_left(self):
        w = make_cosine_tile_weight(
            64, 64, 16, torch.device("cpu"), torch.float32,
            is_top=True, is_left=True,
        )
        self.assertEqual(w[0, 0, 0, 0].item(), 1.0)

    def test_interior_has_ramp(self):
        w = make_cosine_tile_weight(
            64, 64, 16, torch.device("cpu"), torch.float32,
            is_top=False, is_left=False,
        )
        self.assertLess(w[0, 0, 0, 0].item(), 1.0)

    def test_center_is_one(self):
        w = make_cosine_tile_weight(64, 64, 8, torch.device("cpu"), torch.float32)
        self.assertAlmostEqual(w[0, 0, 32, 32].item(), 1.0, places=3)


class TestPackUnpack(unittest.TestCase):
    def test_roundtrip(self):
        B, C, H, W = 1, 16, 32, 32
        x = torch.randn(B, C, H, W)
        packed = _pack_latents(x, B, C, H, W)
        self.assertEqual(packed.shape, (B, (H // 2) * (W // 2), C * 4))
        unpacked = _unpack_latents(packed, H * 8, W * 8, 8)
        self.assertEqual(unpacked.shape, (B, C, H, W))
        self.assertTrue(torch.allclose(x, unpacked, atol=1e-6))

    def test_packed_seq_len(self):
        B, C, H, W = 2, 16, 64, 48
        x = torch.randn(B, C, H, W)
        packed = _pack_latents(x, B, C, H, W)
        self.assertEqual(packed.shape[1], (H // 2) * (W // 2))


class TestRoPEIDs(unittest.TestCase):
    def test_full_image_ids_shape(self):
        ids = _prepare_latent_image_ids(16, 16, torch.device("cpu"), torch.float32)
        self.assertEqual(ids.shape, (16 * 16, 3))

    def test_tile_ids_offset(self):
        ids = _prepare_tile_image_ids(8, 8, 4, 6, torch.device("cpu"), torch.float32)
        self.assertEqual(ids.shape, (8 * 8, 3))
        # First pixel should have global offset
        self.assertEqual(ids[0, 1].item(), 4.0)  # H offset
        self.assertEqual(ids[0, 2].item(), 6.0)  # W offset

    def test_tile_ids_local_ordering(self):
        ids = _prepare_tile_image_ids(4, 4, 0, 0, torch.device("cpu"), torch.float32)
        # Second row, first col
        self.assertEqual(ids[4, 1].item(), 1.0)
        self.assertEqual(ids[4, 2].item(), 0.0)


class TestAutoStrength(unittest.TestCase):
    def test_single_pass_2x(self):
        s = _compute_auto_strength(2.0, 0, 1)
        self.assertEqual(s, 0.35)

    def test_progressive_first_pass(self):
        s = _compute_auto_strength(4.0, 0, 2)
        self.assertEqual(s, 0.35)

    def test_progressive_second_pass(self):
        s = _compute_auto_strength(4.0, 1, 2)
        self.assertEqual(s, 0.25)


class TestCalculateShift(unittest.TestCase):
    def test_returns_float(self):
        mu = _calculate_shift(256)
        self.assertIsInstance(mu, float)

    def test_larger_seq_len_larger_shift(self):
        mu1 = _calculate_shift(256)
        mu2 = _calculate_shift(4096)
        self.assertGreater(mu2, mu1)


if __name__ == "__main__":
    unittest.main()
