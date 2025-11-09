"""Tests for tree detection service."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import tempfile
from pathlib import Path

from parcel_ai_json.tree_detector import (
    TreeDetection,
    TreePolygon,
    TreeDetectionService,
)


class TestTreePolygon(unittest.TestCase):
    """Test TreePolygon dataclass."""

    def test_tree_polygon_creation(self):
        """Test creating a TreePolygon object."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]
        pixel_polygon = [(100, 100), (150, 100), (150, 150), (100, 150), (100, 100)]

        polygon = TreePolygon(
            geo_polygon=geo_polygon,
            pixel_polygon=pixel_polygon,
            area_sqm=50.5,
            area_pixels=2500,
        )

        self.assertEqual(polygon.geo_polygon, geo_polygon)
        self.assertEqual(polygon.pixel_polygon, pixel_polygon)
        self.assertEqual(polygon.area_sqm, 50.5)
        self.assertEqual(polygon.area_pixels, 2500)

    def test_tree_polygon_to_dict(self):
        """Test converting TreePolygon to dictionary."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]
        pixel_polygon = [(100, 100), (150, 100), (150, 150), (100, 150), (100, 100)]

        polygon = TreePolygon(
            geo_polygon=geo_polygon,
            pixel_polygon=pixel_polygon,
            area_sqm=50.5,
            area_pixels=2500,
        )

        result = polygon.to_dict()

        self.assertEqual(result["geo_polygon"], geo_polygon)
        self.assertEqual(result["pixel_polygon"], pixel_polygon)
        self.assertEqual(result["area_sqm"], 50.5)
        self.assertEqual(result["area_pixels"], 2500)

    def test_tree_polygon_to_geojson_feature(self):
        """Test converting TreePolygon to GeoJSON feature."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]
        pixel_polygon = [(100, 100), (150, 100), (150, 150), (100, 150), (100, 100)]

        polygon = TreePolygon(
            geo_polygon=geo_polygon,
            pixel_polygon=pixel_polygon,
            area_sqm=50.5,
            area_pixels=2500,
        )

        geojson = polygon.to_geojson_feature()

        self.assertEqual(geojson["type"], "Feature")
        self.assertEqual(geojson["geometry"]["type"], "Polygon")
        self.assertEqual(geojson["geometry"]["coordinates"], [geo_polygon])
        self.assertEqual(geojson["properties"]["feature_type"], "tree_cluster")
        self.assertEqual(geojson["properties"]["area_sqm"], 50.5)
        self.assertEqual(geojson["properties"]["area_pixels"], 2500)


class TestTreeDetection(unittest.TestCase):
    """Test TreeDetection dataclass."""

    def test_tree_detection_creation(self):
        """Test creating a TreeDetection object."""
        tree_polygons = [
            TreePolygon(
                geo_polygon=[(-122.4194, 37.7749), (-122.4193, 37.7749)],
                pixel_polygon=[(100, 100), (150, 100)],
                area_sqm=50.5,
                area_pixels=2500,
            )
        ]

        detection = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
            tree_mask_path="/tmp/mask.png",
            tree_polygons=tree_polygons,
        )

        self.assertEqual(detection.tree_pixel_count, 5000)
        self.assertEqual(detection.total_pixels, 262144)
        self.assertEqual(detection.tree_coverage_percent, 1.91)
        self.assertEqual(detection.width, 512)
        self.assertEqual(detection.height, 512)
        self.assertEqual(detection.tree_mask_path, "/tmp/mask.png")
        self.assertEqual(len(detection.tree_polygons), 1)

    def test_tree_detection_to_dict_with_polygons(self):
        """Test converting TreeDetection to dict with polygons."""
        tree_polygons = [
            TreePolygon(
                geo_polygon=[(-122.4194, 37.7749), (-122.4193, 37.7749)],
                pixel_polygon=[(100, 100), (150, 100)],
                area_sqm=50.5,
                area_pixels=2500,
            )
        ]

        detection = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
            tree_polygons=tree_polygons,
        )

        result = detection.to_dict()

        self.assertEqual(result["tree_pixel_count"], 5000)
        self.assertEqual(result["tree_coverage_percent"], 1.91)
        self.assertEqual(len(result["tree_polygons"]), 1)
        self.assertEqual(result["tree_polygons"][0]["area_sqm"], 50.5)

    def test_tree_detection_to_dict_without_polygons(self):
        """Test converting TreeDetection to dict without polygons."""
        detection = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
        )

        result = detection.to_dict()

        self.assertNotIn("tree_polygons", result)


class TestTreeDetectionService(unittest.TestCase):
    """Test TreeDetectionService."""

    def test_extract_tree_polygons_from_mask(self):
        """Test extracting tree polygons from binary mask."""
        service = TreeDetectionService(use_docker=False)

        # Create a simple binary mask with two tree regions
        mask = np.zeros((100, 100), dtype=np.uint8)
        # First tree cluster (top-left)
        mask[10:30, 10:30] = 1
        # Second tree cluster (bottom-right)
        mask[70:90, 70:90] = 1

        # Mock satellite image data
        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        polygons = service._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Should find 2 tree clusters
        self.assertEqual(len(polygons), 2)

        # Check that polygons have required fields
        for polygon in polygons:
            self.assertIsInstance(polygon, TreePolygon)
            self.assertGreater(len(polygon.geo_polygon), 0)
            self.assertGreater(len(polygon.pixel_polygon), 0)
            self.assertGreater(polygon.area_pixels, 0)
            self.assertGreater(polygon.area_sqm, 0)

    def test_extract_tree_polygons_filters_small_areas(self):
        """Test that small noise regions are filtered out."""
        service = TreeDetectionService(
            use_docker=False, min_tree_area_pixels=100
        )

        # Create mask with one large region and one small noise region
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Large tree cluster
        mask[10:40, 10:40] = 1  # 900 pixels
        # Small noise region (should be filtered)
        mask[50:55, 50:55] = 1  # 25 pixels

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        polygons = service._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Should only find the large cluster
        self.assertEqual(len(polygons), 1)
        self.assertGreater(polygons[0].area_pixels, 100)

    def test_extract_tree_polygons_empty_mask(self):
        """Test extracting polygons from empty mask."""
        service = TreeDetectionService(use_docker=False)

        # Empty mask
        mask = np.zeros((100, 100), dtype=np.uint8)

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        polygons = service._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Should find no polygons
        self.assertEqual(len(polygons), 0)

    @patch("detectree.Classifier")
    def test_detect_trees_with_polygons_native(self, mock_classifier_class):
        """Test tree detection with polygon extraction in native mode."""
        service = TreeDetectionService(
            use_docker=False, extract_polygons=True, min_tree_area_pixels=50
        )

        # Mock detectree classifier
        mock_clf = Mock()
        mock_classifier_class.return_value = mock_clf

        # Create mock prediction with two tree regions
        y_pred = np.zeros((100, 100), dtype=np.uint8)
        y_pred[10:30, 10:30] = 1  # First tree cluster
        y_pred[70:90, 70:90] = 1  # Second tree cluster

        mock_clf.predict_img.return_value = y_pred

        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Mock PIL Image
            with patch("PIL.Image") as mock_image:
                mock_img = Mock()
                mock_img.convert.return_value = mock_img
                mock_img.size = (100, 100)
                mock_image.open.return_value = mock_img

                satellite_image = {
                    "path": str(tmp_path),
                    "center_lat": 37.7749,
                    "center_lon": -122.4194,
                    "zoom_level": 20,
                }

                detection = service.detect_trees(satellite_image)

                # Verify results
                self.assertEqual(detection.tree_pixel_count, 800)  # 400 + 400
                self.assertIsNotNone(detection.tree_polygons)
                self.assertEqual(len(detection.tree_polygons), 2)

                # Check polygon properties
                for polygon in detection.tree_polygons:
                    self.assertGreater(polygon.area_pixels, 50)
                    self.assertGreater(polygon.area_sqm, 0)
                    self.assertGreater(len(polygon.geo_polygon), 0)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_detect_trees_without_polygons(self):
        """Test tree detection without polygon extraction."""
        service = TreeDetectionService(use_docker=False, extract_polygons=False)

        with patch("detectree.Classifier") as mock_classifier_class:
            mock_clf = Mock()
            mock_classifier_class.return_value = mock_clf

            y_pred = np.zeros((100, 100), dtype=np.uint8)
            y_pred[10:30, 10:30] = 1
            mock_clf.predict_img.return_value = y_pred

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            try:
                with patch("PIL.Image") as mock_image:
                    mock_img = Mock()
                    mock_img.convert.return_value = mock_img
                    mock_img.size = (100, 100)
                    mock_image.open.return_value = mock_img

                    satellite_image = {
                        "path": str(tmp_path),
                        "center_lat": 37.7749,
                        "center_lon": -122.4194,
                        "zoom_level": 20,
                    }

                    detection = service.detect_trees(satellite_image)

                    # Should not have polygons
                    self.assertIsNone(detection.tree_polygons)

            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    def test_polygon_simplification(self):
        """Test polygon simplification reduces vertex count."""
        service = TreeDetectionService(
            use_docker=False, simplify_tolerance_meters=1.0
        )

        # Create a complex polygon with many vertices
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create irregular shape
        mask[20:80, 20:80] = 1

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        polygons = service._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Should have simplified polygon
        self.assertEqual(len(polygons), 1)
        # Simplified polygon should have fewer vertices than original
        # (exact number depends on simplification algorithm)
        self.assertLess(len(polygons[0].geo_polygon), 100)

    def test_polygon_simplification_disabled(self):
        """Test that simplification can be disabled."""
        service = TreeDetectionService(
            use_docker=False, simplify_tolerance_meters=0.0
        )

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        polygons_no_simplify = service._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Now with simplification
        service_simplified = TreeDetectionService(
            use_docker=False, simplify_tolerance_meters=1.0
        )
        polygons_simplified = service_simplified._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Simplified should have fewer or equal vertices
        self.assertLessEqual(
            len(polygons_simplified[0].geo_polygon),
            len(polygons_no_simplify[0].geo_polygon),
        )

    def test_polygon_simplification_preserves_topology(self):
        """Test that simplification preserves polygon validity."""
        service = TreeDetectionService(
            use_docker=False, simplify_tolerance_meters=2.0
        )

        # Create two separate tree regions
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[70:90, 70:90] = 1

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        polygons = service._extract_tree_polygons(
            mask, satellite_image, image_width=100, image_height=100
        )

        # Should still have 2 polygons after simplification
        self.assertEqual(len(polygons), 2)

        # All polygons should be valid (closed)
        for polygon in polygons:
            # First and last point should be the same (closed polygon)
            self.assertEqual(
                polygon.geo_polygon[0], polygon.geo_polygon[-1]
            )


if __name__ == "__main__":
    unittest.main()
