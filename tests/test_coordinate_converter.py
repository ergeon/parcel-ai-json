#!/usr/bin/env python3
"""Tests for coordinate conversion utilities."""

import unittest
from unittest.mock import patch
from pathlib import Path

from parcel_ai_json.coordinate_converter import (
    get_image_dimensions,
    ImageCoordinateConverter,
)


class TestGetImageDimensions(unittest.TestCase):
    """Test get_image_dimensions utility function."""

    def test_get_dimensions_from_metadata(self):
        """Test getting dimensions from metadata."""
        satellite_image = {
            "width_px": 640,
            "height_px": 480,
        }

        width, height = get_image_dimensions(satellite_image)

        self.assertEqual(width, 640)
        self.assertEqual(height, 480)

    def test_get_dimensions_missing_metadata_no_path(self):
        """Test error when dimensions missing and no image_path."""
        satellite_image = {}

        with self.assertRaises(ValueError) as cm:
            get_image_dimensions(satellite_image)

        self.assertIn("Image dimensions not in metadata", str(cm.exception))

    def test_get_dimensions_from_file(self):
        """Test reading dimensions from file."""
        # Create a temporary test image
        import tempfile
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a small test image
            img = Image.new("RGB", (1024, 768))
            img.save(tmp_path)

            satellite_image = {}

            width, height = get_image_dimensions(
                satellite_image, image_path=tmp_path
            )

            self.assertEqual(width, 1024)
            self.assertEqual(height, 768)
        finally:
            Path(tmp_path).unlink()

    def test_get_dimensions_pil_not_installed(self):
        """Test error when PIL not installed."""
        satellite_image = {}

        # Mock ImportError when importing PIL
        with patch("builtins.__import__", side_effect=ImportError):
            with self.assertRaises(ImportError) as cm:
                get_image_dimensions(satellite_image, image_path="/tmp/test.jpg")

            self.assertIn("PIL (Pillow) is required", str(cm.exception))


class TestImageCoordinateConverter(unittest.TestCase):
    """Test ImageCoordinateConverter class."""

    def test_initialization(self):
        """Test converter initialization."""
        converter = ImageCoordinateConverter(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=480,
            zoom_level=20,
        )

        self.assertEqual(converter.center_lat, 37.7749)
        self.assertEqual(converter.center_lon, -122.4194)
        self.assertEqual(converter.image_width_px, 640)
        self.assertEqual(converter.image_height_px, 480)
        self.assertEqual(converter.zoom_level, 20)

    def test_from_satellite_image_with_metadata(self):
        """Test factory method with dimensions in metadata."""
        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "width_px": 640,
            "height_px": 480,
            "zoom_level": 19,
        }

        converter = ImageCoordinateConverter.from_satellite_image(satellite_image)

        self.assertEqual(converter.center_lat, 37.7749)
        self.assertEqual(converter.center_lon, -122.4194)
        self.assertEqual(converter.image_width_px, 640)
        self.assertEqual(converter.image_height_px, 480)
        self.assertEqual(converter.zoom_level, 19)

    def test_pixel_to_geo_center(self):
        """Test converting center pixel to geo coordinates."""
        converter = ImageCoordinateConverter(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
            zoom_level=20,
        )

        # Center pixel should return center coordinates
        lon, lat = converter.pixel_to_geo(320, 320)

        self.assertAlmostEqual(lon, -122.4194, places=4)
        self.assertAlmostEqual(lat, 37.7749, places=4)

    def test_geo_to_pixel_center(self):
        """Test converting center geo coordinates to pixel."""
        converter = ImageCoordinateConverter(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
            zoom_level=20,
        )

        # Center coordinates should return center pixel
        x, y = converter.geo_to_pixel(-122.4194, 37.7749)

        self.assertAlmostEqual(x, 320, places=0)
        self.assertAlmostEqual(y, 320, places=0)

    def test_pixel_to_geo_roundtrip(self):
        """Test roundtrip conversion pixel -> geo -> pixel."""
        converter = ImageCoordinateConverter(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
            zoom_level=20,
        )

        # Test corner
        original_x, original_y = 100, 200

        # Convert to geo and back
        lon, lat = converter.pixel_to_geo(original_x, original_y)
        x, y = converter.geo_to_pixel(lon, lat)

        self.assertAlmostEqual(x, original_x, places=0)
        self.assertAlmostEqual(y, original_y, places=0)

    def test_get_image_bounds(self):
        """Test getting geographic bounds of image."""
        converter = ImageCoordinateConverter(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
            zoom_level=20,
        )

        bounds = converter.get_image_bounds()

        self.assertIn("north", bounds)
        self.assertIn("south", bounds)
        self.assertIn("east", bounds)
        self.assertIn("west", bounds)

        # North should be greater than south
        self.assertGreater(bounds["north"], bounds["south"])

        # East should be greater than west (in Western hemisphere)
        self.assertGreater(bounds["east"], bounds["west"])


if __name__ == "__main__":
    unittest.main()
