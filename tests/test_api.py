"""Tests for FastAPI REST service."""

import unittest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import tempfile
from pathlib import Path

from parcel_ai_json.api import app
from parcel_ai_json.property_detector import PropertyDetections
from parcel_ai_json.vehicle_detector import VehicleDetection
from parcel_ai_json.tree_detector import TreeDetection


class TestAPI(unittest.TestCase):
    """Test FastAPI endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("detector_loaded", data)

    @patch("parcel_ai_json.api.get_detector")
    def test_detect_endpoint_summary(self, mock_get_detector):
        """Test /detect endpoint with summary format."""
        # Mock detector
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
        mock_get_detector.return_value = mock_detector

        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect",
                    files={"image": ("test.jpg", f, "image/jpeg")},
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
        finally:
            Path(tmp_path).unlink()

    @patch("parcel_ai_json.api.get_detector")
    def test_detect_endpoint_geojson(self, mock_get_detector):
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
        mock_get_detector.return_value = mock_detector

        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect",
                    files={"image": ("test.jpg", f, "image/jpeg")},
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
        finally:
            Path(tmp_path).unlink()

    def test_detect_endpoint_invalid_zoom(self):
        """Test /detect endpoint with invalid zoom level."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect",
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "center_lat": "37.7749",
                        "center_lon": "-122.4194",
                        "zoom_level": "25",  # Invalid: > 22
                        "format": "summary",
                    },
                )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Zoom level must be between", response.json()["detail"])
        finally:
            Path(tmp_path).unlink()

    def test_detect_endpoint_invalid_format(self):
        """Test /detect endpoint with invalid format."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect",
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "center_lat": "37.7749",
                        "center_lon": "-122.4194",
                        "zoom_level": "20",
                        "format": "invalid",
                    },
                )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Format must be", response.json()["detail"])
        finally:
            Path(tmp_path).unlink()

    @patch("parcel_ai_json.api.get_detector")
    def test_detect_vehicles_endpoint(self, mock_get_detector):
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
        mock_get_detector.return_value = mock_detector

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect/vehicles",
                    files={"image": ("test.jpg", f, "image/jpeg")},
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
        finally:
            Path(tmp_path).unlink()

    @patch("parcel_ai_json.api.get_detector")
    def test_detect_pools_endpoint(self, mock_get_detector):
        """Test /detect/pools endpoint."""
        mock_detector = Mock()
        mock_detector.pool_detector.detect_swimming_pools.return_value = []
        mock_get_detector.return_value = mock_detector

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect/pools",
                    files={"image": ("test.jpg", f, "image/jpeg")},
                    data={
                        "center_lat": "37.7749",
                        "center_lon": "-122.4194",
                    },
                )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertEqual(len(data["features"]), 0)
        finally:
            Path(tmp_path).unlink()

    @patch("parcel_ai_json.api.get_sam_service")
    @patch("parcel_ai_json.api.get_detector")
    def test_detect_endpoint_with_sam(
        self,
        mock_get_detector,
        mock_get_sam
    ):
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
        mock_get_detector.return_value = mock_detector

        # Mock SAM service
        mock_sam = Mock()
        mock_sam.points_per_side = 32
        mock_sam.segment_image.return_value = []
        mock_get_sam.return_value = mock_sam

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                response = self.client.post(
                    "/detect",
                    files={"image": ("test.jpg", f, "image/jpeg")},
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
        finally:
            Path(tmp_path).unlink()

if __name__ == "__main__":
    unittest.main()
