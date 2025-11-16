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
from parcel_ai_json.tree_detector import TreeDetection, TreeBoundingBox


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

        tree = TreeBoundingBox(
            pixel_bbox=(50.0, 60.0, 100.0, 110.0),
            geo_bbox=(-122.4197, 37.7752, -122.4196, 37.7753),
            confidence=0.75,
        )

        trees = TreeDetection(
            trees=[tree],
            tree_count=1,
            average_confidence=0.75,
            average_crown_area_sqm=15.5,
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
        self.assertEqual(len(geojson["features"]), 4)  # vehicle + pool + amenity + tree

        # Check tree metadata
        self.assertIn("trees", geojson)
        self.assertEqual(geojson["trees"]["tree_count"], 1)
        self.assertEqual(geojson["trees"]["average_confidence"], 0.75)
        self.assertEqual(geojson["trees"]["average_crown_area_sqm"], 15.5)

    def test_to_geojson_no_trees(self):
        """Test GeoJSON output when no trees detected."""
        trees = TreeDetection(trees=[], tree_count=0)

        detections = PropertyDetections(
            vehicles=[], swimming_pools=[], amenities=[], trees=trees
        )

        geojson = detections.to_geojson()

        # Check structure
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 0)

        # Check tree metadata
        self.assertIn("trees", geojson)
        self.assertEqual(geojson["trees"]["tree_count"], 0)
        self.assertNotIn("average_confidence", geojson["trees"])
        self.assertNotIn("average_crown_area_sqm", geojson["trees"])

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
            trees=[],
            tree_count=5,
            average_confidence=0.65,
            average_crown_area_sqm=12.5,
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
        self.assertEqual(summary["tree_count"], 5)
        self.assertEqual(summary["average_tree_confidence"], 0.65)
        self.assertEqual(summary["average_crown_area_sqm"], 12.5)


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
            tree_confidence=0.15,
            tree_model_name="custom/tree-model",
        )

        self.assertEqual(service.vehicle_detector.model_path, "custom.pt")
        self.assertEqual(service.vehicle_detector.confidence_threshold, 0.2)
        self.assertEqual(service.pool_detector.confidence_threshold, 0.25)
        self.assertEqual(service.amenity_detector.confidence_threshold, 0.35)
        # Check tree detector (deepforest and detectree sub-services)
        self.assertEqual(service.tree_detector.deepforest.confidence_threshold, 0.15)
        self.assertEqual(
            service.tree_detector.deepforest.model_name, "custom/tree-model"
        )

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
            trees=[],
            tree_count=3,
            average_confidence=0.6,
            average_crown_area_sqm=10.0,
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
        mock_pool_detector.detect_swimming_pools.assert_called_once_with(
            satellite_image
        )
        mock_amenity_detector.detect_amenities.assert_called_once_with(satellite_image)
        mock_tree_detector.detect_trees.assert_called_once_with(satellite_image)

        # Verify results
        self.assertEqual(len(detections.vehicles), 1)
        self.assertEqual(len(detections.swimming_pools), 1)
        self.assertEqual(len(detections.amenities), 0)
        self.assertEqual(detections.trees.tree_count, 3)
        self.assertEqual(detections.trees.average_confidence, 0.6)

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
            trees=[], tree_count=0
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
        self.assertIn("trees", geojson)
        self.assertEqual(geojson["trees"]["tree_count"], 0)

    def test_to_geojson_with_tree_polygons(self):
        """Test GeoJSON output with tree polygons (detectree)."""
        from parcel_ai_json.tree_detector import TreePolygon

        tree_polygons = [
            TreePolygon(
                geo_polygon=[
                    (-122.4194, 37.7749),
                    (-122.4193, 37.7749),
                    (-122.4193, 37.7748),
                    (-122.4194, 37.7748),
                    (-122.4194, 37.7749),
                ],
                pixel_polygon=[(100, 100), (150, 100), (150, 150), (100, 150)],
                area_sqm=50.5,
                area_pixels=2500,
            )
        ]

        trees = TreeDetection(
            trees=[],
            tree_count=0,
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            tree_polygons=tree_polygons,
        )

        detections = PropertyDetections(
            vehicles=[],
            swimming_pools=[],
            amenities=[],
            trees=trees,
        )

        geojson = detections.to_geojson()

        # Should include tree polygon features
        self.assertEqual(len(geojson["features"]), 1)
        self.assertEqual(
            geojson["features"][0]["properties"]["feature_type"], "tree_cluster"
        )
        self.assertEqual(geojson["trees"]["tree_coverage_percent"], 1.91)

    def test_to_geojson_with_fences(self):
        """Test GeoJSON output with fence detections."""
        from parcel_ai_json.fence_detector import FenceDetection
        import numpy as np

        fence_detection = FenceDetection(
            probability_mask=np.zeros((512, 512), dtype=np.float32),
            binary_mask=np.zeros((512, 512), dtype=np.uint8),
            geo_polygons=[
                [
                    (-122.4194, 37.7749),
                    (-122.4193, 37.7749),
                    (-122.4193, 37.7748),
                    (-122.4194, 37.7748),
                    (-122.4194, 37.7749),
                ]
            ],
            max_probability=0.75,
            mean_probability=0.08,
            fence_pixel_count=300,
            threshold=0.1,
        )

        trees = TreeDetection(trees=[], tree_count=0)

        detections = PropertyDetections(
            vehicles=[],
            swimming_pools=[],
            amenities=[],
            trees=trees,
            fences=fence_detection,
        )

        geojson = detections.to_geojson()

        # Should include fence features
        self.assertGreater(len(geojson["features"]), 0)
        fence_features = [
            f for f in geojson["features"] if f["properties"]["feature_type"] == "fence"
        ]
        self.assertEqual(len(fence_features), 1)

    def test_summary_with_tree_polygons_and_fences(self):
        """Test summary with tree polygons and fences."""
        from parcel_ai_json.tree_detector import TreePolygon
        from parcel_ai_json.fence_detector import FenceDetection
        import numpy as np

        tree_polygons = [
            TreePolygon(
                geo_polygon=[(-122.4194, 37.7749), (-122.4193, 37.7749)],
                pixel_polygon=[(100, 100), (150, 100)],
                area_sqm=50.5,
                area_pixels=2500,
            ),
            TreePolygon(
                geo_polygon=[(-122.4195, 37.7750), (-122.4194, 37.7750)],
                pixel_polygon=[(200, 200), (250, 200)],
                area_sqm=45.0,
                area_pixels=2000,
            ),
        ]

        trees = TreeDetection(
            trees=[],
            tree_count=0,
            tree_pixel_count=5000,
            total_pixels=262144,
            tree_coverage_percent=1.91,
            tree_polygons=tree_polygons,
        )

        fence_detection = FenceDetection(
            probability_mask=np.zeros((512, 512), dtype=np.float32),
            binary_mask=np.zeros((512, 512), dtype=np.uint8),
            geo_polygons=[
                [(-122.4194, 37.7749), (-122.4193, 37.7749)],
                [(-122.4195, 37.7750), (-122.4194, 37.7750)],
            ],
            max_probability=0.75,
            mean_probability=0.08,
            fence_pixel_count=300,
            threshold=0.1,
        )

        detections = PropertyDetections(
            vehicles=[],
            swimming_pools=[],
            amenities=[],
            trees=trees,
            fences=fence_detection,
        )

        summary = detections.summary()

        self.assertEqual(summary["tree_coverage_percent"], 1.91)
        self.assertEqual(summary["tree_polygon_count"], 2)
        self.assertEqual(summary["fence_pixel_count"], 300)
        self.assertEqual(summary["fence_segment_count"], 2)


if __name__ == "__main__":
    unittest.main()
