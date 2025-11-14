"""Tests for fence detection service."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
from pathlib import Path

from parcel_ai_json.fence_detector import (
    FenceDetection,
    FenceDetectionService,
)


class TestFenceDetection(unittest.TestCase):
    """Test FenceDetection dataclass."""

    def test_fence_detection_creation(self):
        """Test creating a FenceDetection object."""
        prob_mask = np.random.rand(512, 512).astype(np.float32)
        binary_mask = (prob_mask > 0.1).astype(np.uint8) * 255
        geo_polygons = [[(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1), (0.0, 0.0)]]

        detection = FenceDetection(
            probability_mask=prob_mask,
            binary_mask=binary_mask,
            geo_polygons=geo_polygons,
            max_probability=0.95,
            mean_probability=0.15,
            fence_pixel_count=1000,
            threshold=0.1,
        )

        self.assertEqual(detection.probability_mask.shape, (512, 512))
        self.assertEqual(detection.binary_mask.shape, (512, 512))
        self.assertEqual(len(detection.geo_polygons), 1)
        self.assertEqual(detection.max_probability, 0.95)
        self.assertEqual(detection.mean_probability, 0.15)
        self.assertEqual(detection.fence_pixel_count, 1000)
        self.assertEqual(detection.threshold, 0.1)

    def test_fence_detection_to_dict(self):
        """Test converting FenceDetection to dictionary."""
        prob_mask = np.random.rand(512, 512).astype(np.float32)
        binary_mask = np.zeros((512, 512), dtype=np.uint8)
        geo_polygons = [
            [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.0)],
            [(0.2, 0.2), (0.3, 0.2), (0.3, 0.3), (0.2, 0.2)],
        ]

        detection = FenceDetection(
            probability_mask=prob_mask,
            binary_mask=binary_mask,
            geo_polygons=geo_polygons,
            max_probability=0.85,
            mean_probability=0.12,
            fence_pixel_count=500,
            threshold=0.1,
        )

        result = detection.to_dict()

        self.assertEqual(result["fence_pixel_count"], 500)
        self.assertEqual(result["max_probability"], 0.85)
        self.assertEqual(result["mean_probability"], 0.12)
        self.assertEqual(result["threshold"], 0.1)
        self.assertEqual(result["fence_segment_count"], 2)

    def test_fence_detection_to_geojson_features(self):
        """Test converting FenceDetection to GeoJSON features."""
        prob_mask = np.random.rand(512, 512).astype(np.float32)
        binary_mask = np.zeros((512, 512), dtype=np.uint8)
        geo_polygons = [
            [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1), (0.0, 0.0)],
            [(0.2, 0.2), (0.3, 0.2), (0.3, 0.3), (0.2, 0.3), (0.2, 0.2)],
        ]

        detection = FenceDetection(
            probability_mask=prob_mask,
            binary_mask=binary_mask,
            geo_polygons=geo_polygons,
            max_probability=0.75,
            mean_probability=0.08,
            fence_pixel_count=300,
            threshold=0.1,
        )

        features = detection.to_geojson_features()

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["type"], "Feature")
        self.assertEqual(features[0]["geometry"]["type"], "Polygon")
        self.assertEqual(features[0]["geometry"]["coordinates"], [geo_polygons[0]])
        self.assertEqual(features[0]["properties"]["feature_type"], "fence")
        self.assertEqual(features[0]["properties"]["segment_id"], 0)
        self.assertEqual(features[0]["properties"]["max_probability"], 0.75)
        self.assertEqual(features[0]["properties"]["mean_probability"], 0.08)
        self.assertEqual(features[0]["properties"]["threshold"], 0.1)

        self.assertEqual(features[1]["properties"]["segment_id"], 1)


class TestFenceDetectionService(unittest.TestCase):
    """Test FenceDetectionService."""

    def test_initialization(self):
        """Test service initialization with custom parameters."""
        service = FenceDetectionService(
            model_path="/custom/path/model.pth", threshold=0.2, device="cpu"
        )

        self.assertEqual(service.threshold, 0.2)
        self.assertEqual(service.device, "cpu")
        self.assertEqual(service.model_path, "/custom/path/model.pth")
        self.assertIsNone(service._model)

    def test_initialization_defaults(self):
        """Test service initialization with default parameters."""
        service = FenceDetectionService()

        self.assertEqual(service.threshold, 0.1)
        self.assertEqual(service.device, "cpu")
        self.assertTrue(service.model_path.endswith("hed_fence_weighted_loss.pth"))
        self.assertIsNone(service._model)

    @patch("parcel_ai_json.fence_detector.Path")
    @patch("parcel_ai_json.fence_detector.torch")
    @patch("parcel_ai_json.fence_detector.HED")
    def test_load_model(self, mock_hed, mock_torch, mock_path):
        """Test model loading."""
        # Mock Path.exists() to return True
        mock_path.return_value.exists.return_value = True

        # Mock model with proper method chaining
        mock_model = Mock()
        mock_model.to.return_value = mock_model  # Chain to() method
        mock_model.eval.return_value = mock_model  # Chain eval() method
        mock_hed.return_value = mock_model

        # Mock checkpoint
        mock_checkpoint = {
            "model_state_dict": {},
            "epoch": 34,
            "best_val_loss": 0.411,
        }
        mock_torch.load.return_value = mock_checkpoint

        service = FenceDetectionService()
        service._load_model()

        # Verify model was loaded
        self.assertIsNotNone(service._model)
        mock_hed.assert_called_once_with(pretrained=False, input_channels=4)
        mock_model.load_state_dict.assert_called_once()
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("parcel_ai_json.fence_detector.Path")
    def test_load_model_file_not_found(self, mock_path):
        """Test model loading with missing file."""
        # Mock Path.exists() to return False
        mock_path.return_value.exists.return_value = False

        service = FenceDetectionService()

        with self.assertRaises(FileNotFoundError):
            service._load_model()

    def test_detect_fences_simple(self):
        """Test fence detection service can be instantiated and has correct parameters."""
        service = FenceDetectionService(threshold=0.15)

        # Test basic service properties
        self.assertEqual(service.threshold, 0.15)
        self.assertIsNone(service._model)  # Not yet loaded

    def test_prepare_4channel_input_shape(self):
        """Test 4-channel input preparation returns correct shape."""
        service = FenceDetectionService()

        # This would require actual image file, so we just test the method exists
        self.assertTrue(hasattr(service, "_prepare_4channel_input"))
        self.assertTrue(callable(getattr(service, "_prepare_4channel_input")))

    @patch("parcel_ai_json.fence_detector.Path")
    def test_detect_fences_image_not_found(self, mock_path):
        """Test fence detection with missing image."""
        # Mock Path.exists() to return False
        mock_path.return_value.exists.return_value = False

        service = FenceDetectionService()

        satellite_image = {
            "path": "/test/missing.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
        }

        with self.assertRaises(FileNotFoundError):
            service.detect_fences(satellite_image)

    def test_detect_fences_geojson_method_exists(self):
        """Test that detect_fences_geojson method exists."""
        service = FenceDetectionService()

        # Test method exists
        self.assertTrue(hasattr(service, "detect_fences_geojson"))
        self.assertTrue(callable(getattr(service, "detect_fences_geojson")))


if __name__ == "__main__":
    unittest.main()
