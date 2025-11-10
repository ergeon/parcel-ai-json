"""Tests for unified property detection service."""

import unittest
from unittest.mock import Mock, patch

from parcel_ai_json.property_detector import (
    PropertyDetections,
    PropertyDetectionService,
)
from parcel_ai_json.vehicle_detector import VehicleDetection
from parcel_ai_json.swimming_pool_detector import SwimmingPoolDetection
from parcel_ai_json.amenity_detector import AmenityDetection
from parcel_ai_json.tree_detector import TreeDetection


class TestPropertyDetections(unittest.TestCase):
    """Test PropertyDetections dataclass."""

    def test_to_geojson(self):
        """Test converting PropertyDetections to GeoJSON."""
        # Create sample detections
        vehicle = VehicleDetection(
            pixel_bbox=(100, 200, 150, 250),
            geo_polygon=[
                (-122.4194, 37.7749),
                (-122.4193, 37.7749),
                (-122.4193, 37.7748),
                (-122.4194, 37.7748),
                (-122.4194, 37.7749),
            ],
            confidence=0.9,
            class_name="car",
        )

        pool = SwimmingPoolDetection(
            pixel_bbox=(300, 400, 350, 450),
            geo_polygon=[
                (-122.4195, 37.7750),
                (-122.4194, 37.7750),
                (-122.4194, 37.7749),
                (-122.4195, 37.7749),
                (-122.4195, 37.7750),
            ],
            confidence=0.85,
            area_sqm=25.0,
        )

        amenity = AmenityDetection(
            amenity_type="tennis-court",
            pixel_bbox=(500, 600, 550, 650),
            geo_polygon=[
                (-122.4196, 37.7751),
                (-122.4195, 37.7751),
                (-122.4195, 37.7750),
                (-122.4196, 37.7750),
                (-122.4196, 37.7751),
            ],
            confidence=0.8,
            area_sqm=200.0,
        )

        trees = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
            tree_mask_path="/tmp/trees.png",
        )

        detections = PropertyDetections(
            vehicles=[vehicle],
            swimming_pools=[pool],
            amenities=[amenity],
            trees=trees,
        )

        geojson = detections.to_geojson()

        # Check structure
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 3)  # vehicle + pool + amenity

        # Check tree coverage metadata (only essential fields)
        self.assertIn("tree_coverage", geojson)
        self.assertEqual(geojson["tree_coverage"]["tree_coverage_percent"], 1.91)
        # Internal fields should not be exposed
        self.assertNotIn("tree_pixel_count", geojson["tree_coverage"])
        self.assertNotIn("tree_mask_path", geojson["tree_coverage"])
        self.assertNotIn("total_pixels", geojson["tree_coverage"])
        self.assertNotIn("image_width", geojson["tree_coverage"])
        self.assertNotIn("image_height", geojson["tree_coverage"])

    def test_to_geojson_no_tree_mask(self):
        """Test GeoJSON output when tree mask is not available."""
        trees = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
            tree_mask_path=None,
        )

        detections = PropertyDetections(
            vehicles=[],
            swimming_pools=[],
            amenities=[],
            trees=trees,
        )

        geojson = detections.to_geojson()

        # Internal fields should not be in response
        self.assertNotIn("tree_mask_path", geojson["tree_coverage"])
        self.assertNotIn("tree_pixel_count", geojson["tree_coverage"])
        self.assertNotIn("total_pixels", geojson["tree_coverage"])
        # Only essential field should be present
        self.assertIn("tree_coverage_percent", geojson["tree_coverage"])

    def test_summary(self):
        """Test summary statistics."""
        vehicle = VehicleDetection(
            pixel_bbox=(100, 200, 150, 250),
            geo_polygon=[
                (-122.4194, 37.7749),
                (-122.4193, 37.7749),
                (-122.4193, 37.7748),
                (-122.4194, 37.7748),
                (-122.4194, 37.7749),
            ],
            confidence=0.9,
            class_name="car",
        )

        pool = SwimmingPoolDetection(
            pixel_bbox=(300, 400, 350, 450),
            geo_polygon=[
                (-122.4195, 37.7750),
                (-122.4194, 37.7750),
                (-122.4194, 37.7749),
                (-122.4195, 37.7749),
                (-122.4195, 37.7750),
            ],
            confidence=0.85,
            area_sqm=25.0,
        )

        amenity1 = AmenityDetection(
            amenity_type="tennis-court",
            pixel_bbox=(500, 600, 550, 650),
            geo_polygon=[
                (-122.4196, 37.7751),
                (-122.4195, 37.7751),
                (-122.4195, 37.7750),
                (-122.4196, 37.7750),
                (-122.4196, 37.7751),
            ],
            confidence=0.8,
            area_sqm=200.0,
        )

        amenity2 = AmenityDetection(
            amenity_type="tennis-court",
            pixel_bbox=(700, 800, 750, 850),
            geo_polygon=[
                (-122.4197, 37.7752),
                (-122.4196, 37.7752),
                (-122.4196, 37.7751),
                (-122.4197, 37.7751),
                (-122.4197, 37.7752),
            ],
            confidence=0.75,
            area_sqm=210.0,
        )

        trees = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
        )

        detections = PropertyDetections(
            vehicles=[vehicle],
            swimming_pools=[pool],
            amenities=[amenity1, amenity2],
            trees=trees,
        )

        summary = detections.summary()

        self.assertEqual(summary["vehicles"], 1)
        self.assertEqual(summary["swimming_pools"], 1)
        self.assertEqual(summary["total_amenities"], 2)
        self.assertEqual(summary["amenities"]["tennis-court"], 2)
        self.assertEqual(summary["tree_coverage_percent"], 1.91)


class TestPropertyDetectionService(unittest.TestCase):
    """Test PropertyDetectionService."""

    def test_initialization(self):
        """Test service initialization."""
        service = PropertyDetectionService(
            model_path="custom.pt",
            vehicle_confidence=0.2,
            pool_confidence=0.25,
            amenity_confidence=0.35,
            device="cuda",
            tree_use_docker=False,
        )

        self.assertEqual(service.vehicle_detector.model_path, "custom.pt")
        self.assertEqual(service.vehicle_detector.confidence_threshold, 0.2)
        self.assertEqual(service.pool_detector.confidence_threshold, 0.25)
        self.assertEqual(service.amenity_detector.confidence_threshold, 0.35)
        self.assertEqual(service.tree_detector.use_docker, False)

    @patch("parcel_ai_json.property_detector.TreeDetectionService")
    @patch("parcel_ai_json.property_detector.AmenityDetectionService")
    @patch("parcel_ai_json.property_detector.SwimmingPoolDetectionService")
    @patch("parcel_ai_json.property_detector.VehicleDetectionService")
    def test_detect_all(
        self,
        mock_vehicle_service,
        mock_pool_service,
        mock_amenity_service,
        mock_tree_service,
    ):
        """Test detecting all property features."""
        # Mock detectors
        mock_vehicle_detector = Mock()
        mock_vehicle_detector.detect_vehicles.return_value = [
            VehicleDetection(
                pixel_bbox=(100, 200, 150, 250),
                geo_polygon=[
                    (-122.4194, 37.7749),
                    (-122.4193, 37.7749),
                    (-122.4193, 37.7748),
                    (-122.4194, 37.7748),
                    (-122.4194, 37.7749),
                ],
                confidence=0.9,
                class_name="car",
            )
        ]
        mock_vehicle_service.return_value = mock_vehicle_detector

        mock_pool_detector = Mock()
        mock_pool_detector.detect_swimming_pools.return_value = [
            SwimmingPoolDetection(
                pixel_bbox=(300, 400, 350, 450),
                geo_polygon=[
                    (-122.4195, 37.7750),
                    (-122.4194, 37.7750),
                    (-122.4194, 37.7749),
                    (-122.4195, 37.7749),
                    (-122.4195, 37.7750),
                ],
                confidence=0.85,
                area_sqm=25.0,
            )
        ]
        mock_pool_service.return_value = mock_pool_detector

        mock_amenity_detector = Mock()
        mock_amenity_detector.detect_amenities.return_value = []
        mock_amenity_service.return_value = mock_amenity_detector

        mock_tree_detector = Mock()
        mock_tree_detector.detect_trees.return_value = TreeDetection(
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            width=512,
            height=512,
        )
        mock_tree_service.return_value = mock_tree_detector

        # Create service
        service = PropertyDetectionService()

        satellite_image = {
            "path": "/tmp/test.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        detections = service.detect_all(satellite_image)

        # Verify all detectors were called
        mock_vehicle_detector.detect_vehicles.assert_called_once_with(satellite_image)
        mock_pool_detector.detect_swimming_pools.assert_called_once_with(satellite_image)
        mock_amenity_detector.detect_amenities.assert_called_once_with(satellite_image)
        mock_tree_detector.detect_trees.assert_called_once_with(satellite_image)

        # Verify results
        self.assertEqual(len(detections.vehicles), 1)
        self.assertEqual(len(detections.swimming_pools), 1)
        self.assertEqual(len(detections.amenities), 0)
        self.assertEqual(detections.trees.tree_coverage_percent, 1.91)

    @patch("parcel_ai_json.property_detector.TreeDetectionService")
    @patch("parcel_ai_json.property_detector.AmenityDetectionService")
    @patch("parcel_ai_json.property_detector.SwimmingPoolDetectionService")
    @patch("parcel_ai_json.property_detector.VehicleDetectionService")
    def test_detect_all_geojson(
        self,
        mock_vehicle_service,
        mock_pool_service,
        mock_amenity_service,
        mock_tree_service,
    ):
        """Test detecting all features and returning GeoJSON."""
        # Mock detectors with empty results
        mock_vehicle_detector = Mock()
        mock_vehicle_detector.detect_vehicles.return_value = []
        mock_vehicle_service.return_value = mock_vehicle_detector

        mock_pool_detector = Mock()
        mock_pool_detector.detect_swimming_pools.return_value = []
        mock_pool_service.return_value = mock_pool_detector

        mock_amenity_detector = Mock()
        mock_amenity_detector.detect_amenities.return_value = []
        mock_amenity_service.return_value = mock_amenity_detector

        mock_tree_detector = Mock()
        mock_tree_detector.detect_trees.return_value = TreeDetection(
            tree_pixel_count=0,
            total_pixels=262144,
            tree_coverage_percent=0.0,
            width=512,
            height=512,
        )
        mock_tree_service.return_value = mock_tree_detector

        service = PropertyDetectionService()

        satellite_image = {
            "path": "/tmp/test.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
        }

        geojson = service.detect_all_geojson(satellite_image)

        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 0)
        self.assertIn("tree_coverage", geojson)


if __name__ == "__main__":
    unittest.main()
