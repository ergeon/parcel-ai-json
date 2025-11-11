"""Tests for amenity detection service."""

import unittest
from unittest.mock import Mock, patch

from parcel_ai_json.amenity_detector import (
    AmenityDetection,
    AmenityDetectionService,
)


class TestAmenityDetection(unittest.TestCase):
    """Test AmenityDetection dataclass."""

    def test_amenity_detection_creation(self):
        """Test creating an AmenityDetection object."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]
        detection = AmenityDetection(
            amenity_type="tennis-court",
            confidence=0.85,
            pixel_bbox=(100, 200, 150, 250),
            geo_polygon=geo_polygon,
            area_sqm=500.5,
        )

        self.assertEqual(detection.amenity_type, "tennis-court")
        self.assertEqual(detection.confidence, 0.85)
        self.assertEqual(detection.geo_polygon, geo_polygon)
        self.assertEqual(detection.pixel_bbox, (100, 200, 150, 250))
        self.assertEqual(detection.area_sqm, 500.5)

    def test_amenity_detection_to_dict(self):
        """Test converting AmenityDetection to dictionary."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]
        detection = AmenityDetection(
            amenity_type="basketball-court",
            confidence=0.75,
            pixel_bbox=(100, 200, 150, 250),
            geo_polygon=geo_polygon,
            area_sqm=300.0,
        )

        result = detection.to_dict()

        self.assertEqual(result["amenity_type"], "basketball-court")
        self.assertEqual(result["confidence"], 0.75)
        self.assertEqual(result["geo_polygon"], geo_polygon)
        self.assertEqual(result["pixel_bbox"], [100, 200, 150, 250])
        self.assertEqual(result["area_sqm"], 300.0)

    def test_amenity_detection_to_geojson(self):
        """Test converting AmenityDetection to GeoJSON feature."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]
        detection = AmenityDetection(
            amenity_type="soccer-field",
            confidence=0.90,
            pixel_bbox=(100, 200, 150, 250),
            geo_polygon=geo_polygon,
            area_sqm=1000.0,
        )

        geojson = detection.to_geojson_feature()

        self.assertEqual(geojson["type"], "Feature")
        self.assertEqual(geojson["geometry"]["type"], "Polygon")
        self.assertEqual(geojson["geometry"]["coordinates"], [geo_polygon])
        self.assertEqual(geojson["properties"]["feature_type"], "amenity")
        self.assertEqual(geojson["properties"]["amenity_type"], "soccer-field")
        self.assertEqual(geojson["properties"]["confidence"], 0.90)
        self.assertEqual(geojson["properties"]["area_sqm"], 1000.0)


class TestAmenityDetectionService(unittest.TestCase):
    """Test AmenityDetectionService."""

    def test_initialization(self):
        """Test service initialization with custom parameters."""
        service = AmenityDetectionService(
            model_path="custom_model.pt",
            confidence_threshold=0.4,
            device="cuda",
        )

        self.assertEqual(service.model_path, "custom_model.pt")
        self.assertEqual(service.confidence_threshold, 0.4)
        self.assertEqual(service.device, "cuda")
        self.assertIsNone(service._model)

    def test_initialization_defaults(self):
        """Test service initialization with default parameters."""
        service = AmenityDetectionService()

        self.assertIsNone(service.model_path)
        self.assertEqual(service.confidence_threshold, 0.3)
        self.assertEqual(service.device, "cpu")
        self.assertIn("tennis court", service.amenity_classes)
        self.assertIn("basketball court", service.amenity_classes)

    @patch("ultralytics.YOLO")
    def test_load_model_default(self, mock_yolo):
        """Test loading default YOLO-OBB model."""
        service = AmenityDetectionService()
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service._load_model()

        # Check that YOLO was called with either short or full path to yolov8m-obb.pt
        call_args = mock_yolo.call_args[0][0]
        self.assertTrue(
            call_args == "yolov8m-obb.pt" or call_args.endswith("models/yolov8m-obb.pt"),
            f"Expected YOLO to be called with yolov8m-obb.pt or full path, got: {call_args}"
        )
        mock_model.to.assert_called_once_with("cpu")
        self.assertEqual(service._model, mock_model)

    @patch("ultralytics.YOLO")
    def test_detect_amenities_obb_format(self, mock_yolo):
        """Test detecting amenities with OBB format results."""
        service = AmenityDetectionService(confidence_threshold=0.3)

        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        service._model = mock_model

        # Mock OBB detection results
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.__len__ = Mock(return_value=1)  # Make it have length of 1

        # Create mock tensor that supports indexing
        mock_cls_item = Mock()
        mock_cls_item.item.return_value = 2  # tennis court
        mock_result.obb.cls = [mock_cls_item]

        mock_conf_item = Mock()
        mock_conf_item.item.return_value = 0.85
        mock_result.obb.conf = [mock_conf_item]

        mock_xyxy_item = Mock()
        mock_xyxy_item.tolist.return_value = [100, 200, 150, 250]
        mock_result.obb.xyxy = [mock_xyxy_item]

        # Set names on the model, not the result
        mock_model.names = {
            0: "plane",
            1: "ship",
            2: "tennis court",
            3: "basketball court",
        }

        mock_model.predict.return_value = [mock_result]

        # Test image
        satellite_image = {
            "path": "/tmp/test.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        # Mock Path.exists
        with patch("pathlib.Path.exists", return_value=True):
            # Mock PIL Image
            with patch("PIL.Image") as mock_image:
                mock_img = Mock()
                mock_img.size = (512, 512)
                mock_img.__enter__ = Mock(return_value=mock_img)
                mock_img.__exit__ = Mock(return_value=False)
                mock_image.open.return_value = mock_img

                detections = service.detect_amenities(satellite_image)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].amenity_type, "tennis court")
        self.assertEqual(detections[0].confidence, 0.85)

    @patch("ultralytics.YOLO")
    def test_detect_amenities_filter_non_amenities(self, mock_yolo):
        """Test that non-amenity classes are filtered out."""
        service = AmenityDetectionService(confidence_threshold=0.3)

        mock_model = Mock()
        mock_yolo.return_value = mock_model
        service._model = mock_model

        # Mock results with mixed classes
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.__len__ = Mock(return_value=3)  # Make it have length of 3

        # Create mock tensors that support indexing
        mock_cls_items = []
        for class_id in [0, 2, 1]:  # plane, tennis, ship
            item = Mock()
            item.item.return_value = class_id
            mock_cls_items.append(item)
        mock_result.obb.cls = mock_cls_items

        mock_conf_items = []
        for conf in [0.9, 0.85, 0.8]:
            item = Mock()
            item.item.return_value = conf
            mock_conf_items.append(item)
        mock_result.obb.conf = mock_conf_items

        mock_xyxy_items = []
        for bbox in [[10, 20, 30, 40], [100, 200, 150, 250], [50, 60, 70, 80]]:
            item = Mock()
            item.tolist.return_value = bbox
            mock_xyxy_items.append(item)
        mock_result.obb.xyxy = mock_xyxy_items

        # Set names on the model, not the result
        mock_model.names = {
            0: "plane",
            1: "ship",
            2: "tennis court",
        }

        mock_model.predict.return_value = [mock_result]

        satellite_image = {
            "path": "/tmp/test.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("PIL.Image") as mock_image:
                mock_img = Mock()
                mock_img.size = (512, 512)
                mock_img.__enter__ = Mock(return_value=mock_img)
                mock_img.__exit__ = Mock(return_value=False)
                mock_image.open.return_value = mock_img

                detections = service.detect_amenities(satellite_image)

        # Should only detect tennis court
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].amenity_type, "tennis court")

    @patch("ultralytics.YOLO")
    def test_detect_amenities_no_detections(self, mock_yolo):
        """Test when no amenities are detected."""
        service = AmenityDetectionService()

        mock_model = Mock()
        service._model = mock_model

        # Empty OBB results
        mock_result = Mock()
        mock_result.obb = None

        mock_model.predict.return_value = [mock_result]

        satellite_image = {
            "path": "/tmp/test.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("PIL.Image") as mock_image:
                mock_img = Mock()
                mock_img.size = (512, 512)
                mock_img.__enter__ = Mock(return_value=mock_img)
                mock_img.__exit__ = Mock(return_value=False)
                mock_image.open.return_value = mock_img

                detections = service.detect_amenities(satellite_image)

        self.assertEqual(len(detections), 0)

    def test_detect_amenities_image_not_found(self):
        """Test error when image file doesn't exist."""
        service = AmenityDetectionService()

        satellite_image = {
            "path": "/nonexistent/image.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
        }

        with self.assertRaises(FileNotFoundError):
            service.detect_amenities(satellite_image)


if __name__ == "__main__":
    unittest.main()
