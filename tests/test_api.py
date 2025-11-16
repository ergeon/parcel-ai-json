"""Tests for FastAPI REST service using Clean Architecture with dependency injection."""

import unittest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from PIL import Image
import io

from parcel_ai_json.api import app, get_detector, get_sam_service
from parcel_ai_json.property_detector import PropertyDetections
from parcel_ai_json.vehicle_detector import VehicleDetection
from parcel_ai_json.tree_detector import TreeDetection


class TestAPI(unittest.TestCase):
    """Test FastAPI endpoints using dependency injection."""

    def setUp(self):
        """Set up test client and clear dependency overrides."""
        self.client = TestClient(app)
        app.dependency_overrides = {}

    def tearDown(self):
        """Clean up dependency overrides after tests."""
        app.dependency_overrides = {}

    def _create_test_image(self) -> bytes:
        """Create a valid test image (100x100 white pixel image)."""
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes.read()

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "running")

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("detector_loaded", data)

    def test_detect_endpoint_summary(self):
        """Test /detect endpoint with summary format."""
        # Mock detector using dependency injection
        mock_detector = Mock()
        mock_detections = PropertyDetections(
            vehicles=[
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
            ],
            swimming_pools=[],
            amenities=[],
            trees=TreeDetection(
                trees=[],  # DeepForest individual trees
                tree_count=0,
                average_confidence=None,
                average_crown_area_sqm=None,
                tree_pixel_count=5000,  # detectree coverage
                total_pixels=262144,
                tree_coverage_percent=1.91,
                tree_polygons=None,
                tree_mask_path=None,
            ),
        )
        mock_detector.detect_all.return_value = mock_detections

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        # Create test image
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "format": "summary",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["vehicles"], 1)
        self.assertEqual(data["swimming_pools"], 0)
        self.assertEqual(data["tree_coverage_percent"], 1.91)

    def test_detect_endpoint_geojson(self):
        """Test /detect endpoint with GeoJSON format."""
        # Mock detector
        mock_detector = Mock()
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [],
            "tree_coverage": {
                "tree_pixel_count": 0,
                "total_pixels": 262144,
                "tree_coverage_percent": 0.0,
                "image_width": 512,
                "image_height": 512,
            },
        }
        # Mock the detect_all() method to return a mock detections object
        mock_detections = Mock()
        mock_detections.to_geojson.return_value = mock_geojson
        mock_detector.detect_all.return_value = mock_detections

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        # Create test image
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "format": "geojson",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIn("features", data)
        self.assertIn("tree_coverage", data)

    def test_detect_endpoint_invalid_zoom(self):
        """Test /detect endpoint with invalid zoom level."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "25",  # Invalid: > 22
                "format": "summary",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Zoom level must be between", response.json()["detail"])

    def test_detect_endpoint_invalid_format(self):
        """Test /detect endpoint with invalid format."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "format": "invalid",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Format must be", response.json()["detail"])

    def test_detect_vehicles_endpoint(self):
        """Test /detect/vehicles endpoint."""
        mock_detector = Mock()
        mock_detector.vehicle_detector.detect_vehicles.return_value = [
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

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        image_data = self._create_test_image()

        response = self.client.post(
            "/detect/vehicles",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 1)
        self.assertEqual(data["features"][0]["properties"]["vehicle_class"], "car")

    def test_detect_pools_endpoint(self):
        """Test /detect/pools endpoint."""
        mock_detector = Mock()
        mock_detector.pool_detector.detect_swimming_pools.return_value = []

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        image_data = self._create_test_image()

        response = self.client.post(
            "/detect/pools",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 0)

    def test_detect_endpoint_with_sam(self):
        """Test /detect endpoint with SAM segmentation enabled."""
        # Mock detector
        mock_detector = Mock()
        mock_detections = Mock()
        mock_detections.vehicles = []
        mock_detections.swimming_pools = []
        mock_detections.amenities = []
        mock_detections.trees = None
        mock_detections.to_geojson.return_value = {
            "type": "FeatureCollection",
            "features": [],
        }
        mock_detector.detect_all.return_value = mock_detections

        # Mock SAM service
        mock_sam = Mock()
        mock_sam.points_per_side = 32
        mock_sam.segment_image.return_value = []

        # Override dependencies
        app.dependency_overrides[get_detector] = lambda: mock_detector
        app.dependency_overrides[get_sam_service] = lambda: mock_sam

        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "include_sam": "true",
                "sam_points_per_side": "16",
            },
        )

        self.assertEqual(response.status_code, 200)
        # Verify SAM was called
        mock_sam.segment_image.assert_called_once()
        # Verify points_per_side was updated
        self.assertEqual(mock_sam.points_per_side, 16)

    def test_detect_amenities_endpoint(self):
        """Test /detect/amenities endpoint."""
        from parcel_ai_json.amenity_detector import AmenityDetection

        mock_detector = Mock()
        mock_detector.amenity_detector.detect_amenities.return_value = [
            AmenityDetection(
                amenity_type="tennis_court",
                pixel_bbox=(50, 100, 200, 250),
                geo_polygon=[
                    (-122.4195, 37.7750),
                    (-122.4194, 37.7750),
                    (-122.4194, 37.7749),
                    (-122.4195, 37.7749),
                    (-122.4195, 37.7750),
                ],
                confidence=0.85,
            )
        ]

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        image_data = self._create_test_image()

        response = self.client.post(
            "/detect/amenities",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 1)
        self.assertEqual(
            data["features"][0]["properties"]["amenity_type"], "tennis_court"
        )

    def test_detect_trees_endpoint(self):
        """Test /detect/trees endpoint."""
        mock_detector = Mock()
        mock_tree_detection = TreeDetection(
            trees=[],
            tree_count=0,
            average_confidence=None,
            average_crown_area_sqm=None,
            tree_pixel_count=3000,
            total_pixels=100000,
            tree_coverage_percent=3.0,
            tree_polygons=None,
            tree_mask_path=None,
        )
        mock_detector.tree_detector.detect_trees.return_value = mock_tree_detection

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        image_data = self._create_test_image()

        response = self.client.post(
            "/detect/trees",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["tree_coverage_percent"], 3.0)
        self.assertEqual(data["tree_count"], 0)

    def test_detect_fences_endpoint(self):
        """Test /detect/fences endpoint."""
        mock_detector = Mock()

        # Create a mock fence detection object with required attributes
        mock_fence_detection = Mock()
        mock_fence_detection.geo_polygons = []
        mock_fence_detection.to_geojson_features.return_value = []
        mock_fence_detection.to_dict.return_value = {
            "num_segments": 0,
            "total_length_m": 0,
        }

        mock_detector.fence_detector.detect_fences.return_value = mock_fence_detection

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        image_data = self._create_test_image()

        response = self.client.post(
            "/detect/fences",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIn("features", data)
        self.assertIn("metadata", data)

    def test_segment_sam_endpoint(self):
        """Test /segment/sam endpoint."""
        mock_sam = Mock()
        mock_sam.points_per_side = 32
        mock_sam.segment_image.return_value = []

        # Override dependency
        app.dependency_overrides[get_sam_service] = lambda: mock_sam

        image_data = self._create_test_image()

        response = self.client.post(
            "/segment/sam",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "points_per_side": "32",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIn("features", data)
        mock_sam.segment_image.assert_called_once()

    def test_detect_endpoint_missing_params(self):
        """Test /detect endpoint with missing required parameters."""
        image_data = self._create_test_image()

        # Missing center_lat and center_lon
        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_detect_endpoint_no_image(self):
        """Test /detect endpoint without image file."""
        response = self.client.post(
            "/detect",
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
            },
        )

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_validate_image_content_type_non_image(self):
        """Test validation rejects non-image content type."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.txt", image_data, "text/plain")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("File must be an image", response.json()["detail"])

    def test_validate_coordinates_invalid_lat(self):
        """Test validation rejects invalid latitude."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "95.0",  # Invalid: > 90
                "center_lon": "-122.4194",
                "zoom_level": "20",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Latitude must be between", response.json()["detail"])

    def test_validate_coordinates_invalid_lon(self):
        """Test validation rejects invalid longitude."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-185.0",  # Invalid: < -180
                "zoom_level": "20",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Longitude must be between", response.json()["detail"])

    def test_validate_sam_points_per_side_invalid(self):
        """Test validation rejects invalid SAM points_per_side."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "include_sam": "true",
                "sam_points_per_side": "100",  # Invalid: > 64
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("points_per_side must be between", response.json()["detail"])

    def test_segment_sam_invalid_coordinates(self):
        """Test /segment/sam with invalid coordinates."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/segment/sam",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "91.0",  # Invalid
                "center_lon": "-122.4194",
                "zoom_level": "20",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Latitude must be between", response.json()["detail"])

    def test_segment_sam_invalid_zoom(self):
        """Test /segment/sam with invalid zoom level."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/segment/sam",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "0",  # Invalid: < 1
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Zoom level must be between", response.json()["detail"])

    def test_segment_sam_invalid_points_per_side(self):
        """Test /segment/sam with invalid points_per_side."""
        image_data = self._create_test_image()

        response = self.client.post(
            "/segment/sam",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "points_per_side": "5",  # Invalid: < 8
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("points_per_side must be between", response.json()["detail"])

    def test_parse_regrid_parcel_polygon_geojson_feature(self):
        """Test parsing Regrid parcel polygon from GeoJSON Feature."""
        import json
        from parcel_ai_json.api import parse_regrid_parcel_polygon

        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.0, 37.0], [-121.9, 37.0], [-121.9, 36.9]]],
            },
        }

        result = parse_regrid_parcel_polygon(json.dumps(geojson_feature))

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "Polygon")
        self.assertIn("coordinates", result)

    def test_parse_regrid_parcel_polygon_geojson_geometry(self):
        """Test parsing Regrid parcel polygon from GeoJSON Geometry."""
        import json
        from parcel_ai_json.api import parse_regrid_parcel_polygon

        geojson_geometry = {
            "type": "Polygon",
            "coordinates": [[[-122.0, 37.0], [-121.9, 37.0], [-121.9, 36.9]]],
        }

        result = parse_regrid_parcel_polygon(json.dumps(geojson_geometry))

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "Polygon")

    def test_parse_regrid_parcel_polygon_coordinate_list(self):
        """Test parsing Regrid parcel polygon from coordinate list."""
        import json
        from parcel_ai_json.api import parse_regrid_parcel_polygon

        coord_list = [[-122.0, 37.0], [-121.9, 37.0], [-121.9, 36.9]]

        result = parse_regrid_parcel_polygon(json.dumps(coord_list))

        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_parse_regrid_parcel_polygon_invalid_json(self):
        """Test parsing invalid JSON raises HTTPException."""
        from parcel_ai_json.api import parse_regrid_parcel_polygon
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as cm:
            parse_regrid_parcel_polygon("{invalid json")

        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("Invalid JSON", cm.exception.detail)

    def test_parse_regrid_parcel_polygon_none(self):
        """Test parsing None returns None."""
        from parcel_ai_json.api import parse_regrid_parcel_polygon

        result = parse_regrid_parcel_polygon(None)
        self.assertIsNone(result)

    def test_add_regrid_parcel_to_geojson_dict_geometry(self):
        """Test adding Regrid parcel from dict geometry."""
        from parcel_ai_json.api import add_regrid_parcel_to_geojson

        geojson = {"type": "FeatureCollection", "features": []}
        parcel_polygon = {
            "type": "Polygon",
            "coordinates": [[[-122.0, 37.0], [-121.9, 37.0]]],
        }

        add_regrid_parcel_to_geojson(geojson, parcel_polygon)

        self.assertEqual(len(geojson["features"]), 1)
        self.assertEqual(geojson["features"][0]["type"], "Feature")
        self.assertEqual(
            geojson["features"][0]["properties"]["feature_type"], "regrid_parcel"
        )

    def test_add_regrid_parcel_to_geojson_coord_list(self):
        """Test adding Regrid parcel from coordinate list."""
        from parcel_ai_json.api import add_regrid_parcel_to_geojson

        geojson = {"type": "FeatureCollection", "features": []}
        parcel_polygon = [[-122.0, 37.0], [-121.9, 37.0], [-121.9, 36.9]]

        add_regrid_parcel_to_geojson(geojson, parcel_polygon)

        self.assertEqual(len(geojson["features"]), 1)
        self.assertEqual(geojson["features"][0]["geometry"]["type"], "Polygon")

    def test_create_satellite_image_metadata(self):
        """Test creating satellite image metadata."""
        from parcel_ai_json.api import create_satellite_image_metadata
        from pathlib import Path

        image_path = Path("/tmp/test.jpg")
        metadata = create_satellite_image_metadata(
            image_path, 37.7749, -122.4194, 20
        )

        self.assertEqual(metadata["path"], "/tmp/test.jpg")
        self.assertEqual(metadata["center_lat"], 37.7749)
        self.assertEqual(metadata["center_lon"], -122.4194)
        self.assertEqual(metadata["zoom_level"], 20)

    def test_detect_endpoint_with_regrid_parcel(self):
        """Test /detect endpoint with Regrid parcel polygon."""
        import json

        # Mock detector
        mock_detector = Mock()
        mock_detections = Mock()
        mock_detections.to_geojson.return_value = {
            "type": "FeatureCollection",
            "features": [],
        }
        mock_detector.detect_all.return_value = mock_detections

        # Override dependency
        app.dependency_overrides[get_detector] = lambda: mock_detector

        image_data = self._create_test_image()

        parcel_polygon = {
            "type": "Polygon",
            "coordinates": [[[-122.0, 37.0], [-121.9, 37.0], [-121.9, 36.9]]],
        }

        response = self.client.post(
            "/detect",
            files={"image": ("test.jpg", image_data, "image/jpeg")},
            data={
                "center_lat": "37.7749",
                "center_lon": "-122.4194",
                "zoom_level": "20",
                "detect_fences": "true",
                "regrid_parcel_polygon": json.dumps(parcel_polygon),
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Should have parcel polygon in features
        regrid_features = [
            f
            for f in data["features"]
            if f["properties"].get("feature_type") == "regrid_parcel"
        ]
        self.assertEqual(len(regrid_features), 1)


if __name__ == "__main__":
    unittest.main()
