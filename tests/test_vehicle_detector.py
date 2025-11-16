"""Tests for VehicleDetectionService."""

import pytest
from unittest.mock import Mock, patch
import torch

from parcel_ai_json.vehicle_detector import (
    VehicleDetection,
    VehicleDetectionService,
)


class TestVehicleDetection:
    """Test VehicleDetection dataclass."""

    def test_vehicle_detection_creation(self):
        """Test creating a VehicleDetection object."""
        detection = VehicleDetection(
            pixel_bbox=(10.0, 20.0, 50.0, 60.0),
            geo_polygon=[(-122.0, 37.0), (-122.1, 37.0), (-122.1, 37.1)],
            confidence=0.85,
            class_name="car",
        )

        assert detection.pixel_bbox == (10.0, 20.0, 50.0, 60.0)
        assert len(detection.geo_polygon) == 3
        assert detection.confidence == 0.85
        assert detection.class_name == "car"

    def test_vehicle_detection_to_dict(self):
        """Test converting VehicleDetection to dict."""
        detection = VehicleDetection(
            pixel_bbox=(10.0, 20.0, 50.0, 60.0),
            geo_polygon=[(-122.0, 37.0), (-122.1, 37.0)],
            confidence=0.75,
            class_name="truck",
        )

        result = detection.to_dict()

        assert result["pixel_bbox"] == [10.0, 20.0, 50.0, 60.0]
        assert result["geo_polygon"] == [(-122.0, 37.0), (-122.1, 37.0)]
        assert result["confidence"] == 0.75
        assert result["class_name"] == "truck"

    def test_vehicle_detection_to_geojson(self):
        """Test converting VehicleDetection to GeoJSON feature."""
        detection = VehicleDetection(
            pixel_bbox=(10.0, 20.0, 50.0, 60.0),
            geo_polygon=[
                (-122.0, 37.0),
                (-122.1, 37.0),
                (-122.1, 37.1),
                (-122.0, 37.1),
                (-122.0, 37.0),
            ],
            confidence=0.85,
            class_name="car",
        )

        geojson = detection.to_geojson_feature()

        assert geojson["type"] == "Feature"
        assert geojson["geometry"]["type"] == "Polygon"
        assert geojson["geometry"]["coordinates"] == [detection.geo_polygon]
        assert geojson["properties"]["feature_type"] == "vehicle"
        assert geojson["properties"]["vehicle_class"] == "car"
        assert geojson["properties"]["confidence"] == 0.85
        assert geojson["properties"]["pixel_bbox"] == [10.0, 20.0, 50.0, 60.0]


class TestVehicleDetectionService:
    """Test VehicleDetectionService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = VehicleDetectionService(
            model_path="yolov8m.pt", confidence_threshold=0.3, device="cpu"
        )

    def test_initialization(self):
        """Test VehicleDetectionService initialization."""
        service = VehicleDetectionService(
            model_path="yolov8n.pt", confidence_threshold=0.5, device="cpu"
        )

        assert service.model_path == "yolov8n.pt"
        assert service.confidence_threshold == 0.5
        assert service.device == "cpu"
        assert service._model is None
        assert "car" in service.vehicle_classes
        assert "truck" in service.vehicle_classes

    def test_initialization_defaults(self):
        """Test VehicleDetectionService with default parameters."""
        service = VehicleDetectionService()

        assert service.model_path is None
        assert service.confidence_threshold == 0.3
        assert service.device == "cpu"

    @patch("ultralytics.YOLO")
    def test_load_model_standard(self, mock_yolo):
        """Test loading a standard YOLO model."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = VehicleDetectionService(model_path="yolov8m.pt")
        service._load_model()

        mock_yolo.assert_called_once_with("yolov8m.pt")
        mock_model.to.assert_called_once_with("cpu")
        assert service._model == mock_model

    @patch("ultralytics.YOLO")
    def test_load_model_default(self, mock_yolo):
        """Test loading default YOLO model (yolov8m-obb)."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = VehicleDetectionService()
        service._load_model()

        # Check that YOLO was called with either short or full path to yolov8m-obb.pt
        call_args = mock_yolo.call_args[0][0]
        assert call_args == "yolov8m-obb.pt" or call_args.endswith(
            "models/yolov8m-obb.pt"
        ), f"Expected YOLO with yolov8m-obb.pt, got: {call_args}"

    @patch("ultralytics.YOLO")
    @patch("parcel_ai_json.vehicle_detector.Path")
    def test_load_model_custom_path(self, mock_path, mock_yolo):
        """Test loading custom model with path validation."""
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = VehicleDetectionService(model_path="/custom/model.pt")
        service._load_model()

        mock_yolo.assert_called_once_with("/custom/model.pt")

    @patch("parcel_ai_json.vehicle_detector.Path")
    def test_load_model_custom_not_found(self, mock_path):
        """Test loading custom model that doesn't exist."""
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        service = VehicleDetectionService(model_path="/nonexistent/model.pt")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            service._load_model()

    def test_load_model_no_ultralytics(self):
        """Test loading model when ultralytics is not installed."""
        service = VehicleDetectionService()

        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(
                ImportError, match="Vehicle detection requires ultralytics"
            ):
                service._load_model()

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_regular_bbox(self, mock_image, mock_path):
        """Test detecting vehicles with regular bounding box model."""
        # Create service and mock the model directly
        service = VehicleDetectionService(model_path="yolov8m.pt")

        # Mock the model
        mock_model = Mock()
        mock_model.names = {0: "car", 1: "truck"}

        # Mock detection results (regular bbox format)
        mock_result = Mock()
        mock_result.obb = None
        mock_result.masks = None  # Not a segmentation model
        mock_result.boxes = Mock()

        # Create mock boxes
        mock_box = Mock()
        mock_box.cls = torch.tensor([0])  # car class
        mock_box.conf = torch.tensor([0.85])
        mock_box.xyxy = torch.tensor([[100.0, 200.0, 150.0, 250.0]])

        mock_result.boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes.__len__ = Mock(return_value=1)
        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        # Mock image file existence
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        detections = service.detect_vehicles(satellite_image)

        assert len(detections) == 1
        assert detections[0].class_name == "car"
        assert detections[0].confidence == pytest.approx(0.85, rel=1e-5)
        assert detections[0].pixel_bbox == (100.0, 200.0, 150.0, 250.0)
        assert len(detections[0].geo_polygon) == 5  # Closed polygon

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_obb_format(self, mock_image, mock_path):
        """Test detecting vehicles with OBB (Oriented Bounding Box) model."""
        # Create service and mock the model directly
        service = VehicleDetectionService(model_path="yolov8m-obb.pt")

        # Mock the model
        mock_model = Mock()
        mock_model.names = {10: "small vehicle", 11: "large vehicle"}

        # Mock detection results (OBB format)
        mock_result = Mock()
        mock_result.masks = None  # Not a segmentation model
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([10, 11])  # small and large vehicle
        mock_result.obb.conf = torch.tensor([0.75, 0.65])
        mock_result.obb.xyxy = torch.tensor(
            [[50.0, 60.0, 100.0, 110.0], [200.0, 210.0, 250.0, 260.0]]
        )

        # Make OBB iterable and have length
        mock_result.obb.__len__ = Mock(return_value=2)

        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        # Mock image file existence
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        detections = service.detect_vehicles(satellite_image)

        assert len(detections) == 2
        assert detections[0].class_name == "small vehicle"
        assert detections[0].confidence == pytest.approx(0.75, rel=1e-5)
        assert len(detections[0].geo_polygon) == 5
        assert detections[1].class_name == "large vehicle"
        assert detections[1].confidence == pytest.approx(0.65, rel=1e-5)
        assert len(detections[1].geo_polygon) == 5

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_filter_non_vehicles(self, mock_image, mock_path):
        """Test that non-vehicle detections are filtered out."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        # Mock the model
        mock_model = Mock()
        mock_model.names = {0: "person", 1: "car", 2: "bicycle"}

        # Mock detection results
        mock_result = Mock()
        mock_result.masks = None  # Not a segmentation model
        mock_result.obb = None
        mock_result.boxes = Mock()

        # Create mock boxes - person, car, bicycle
        mock_box_person = Mock()
        mock_box_person.cls = torch.tensor([0])
        mock_box_person.conf = torch.tensor([0.95])
        mock_box_person.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

        mock_box_car = Mock()
        mock_box_car.cls = torch.tensor([1])
        mock_box_car.conf = torch.tensor([0.85])
        mock_box_car.xyxy = torch.tensor([[100.0, 200.0, 150.0, 250.0]])

        mock_box_bicycle = Mock()
        mock_box_bicycle.cls = torch.tensor([2])
        mock_box_bicycle.conf = torch.tensor([0.75])
        mock_box_bicycle.xyxy = torch.tensor([[50.0, 60.0, 70.0, 80.0]])

        mock_result.boxes.__iter__ = Mock(
            return_value=iter([mock_box_person, mock_box_car, mock_box_bicycle])
        )
        mock_result.boxes.__len__ = Mock(return_value=3)
        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        # Mock file
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        detections = service.detect_vehicles(satellite_image)

        # Should only detect the car, not person or bicycle
        assert len(detections) == 1
        assert detections[0].class_name == "car"

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_no_detections(self, mock_image, mock_path):
        """Test detecting vehicles when no vehicles are found."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        mock_model = Mock()
        mock_model.names = {0: "car"}

        # Mock empty detection results
        mock_result = Mock()
        mock_result.masks = None  # Not a segmentation model
        mock_result.obb = None
        mock_result.boxes = None

        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        detections = service.detect_vehicles(satellite_image)

        assert len(detections) == 0

    def test_detect_vehicles_image_not_found(self):
        """Test detecting vehicles when image file doesn't exist."""
        service = VehicleDetectionService()

        satellite_image = {
            "path": "/nonexistent/definitely_does_not_exist_12345.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        with pytest.raises(FileNotFoundError, match="Image not found"):
            service.detect_vehicles(satellite_image)

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_geojson(self, mock_image, mock_path):
        """Test detect_vehicles_geojson method."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        mock_model = Mock()
        mock_model.names = {0: "car"}

        # Mock detection
        mock_result = Mock()
        mock_result.masks = None  # Not a segmentation model
        mock_result.obb = None
        mock_result.boxes = Mock()

        mock_box = Mock()
        mock_box.cls = torch.tensor([0])
        mock_box.conf = torch.tensor([0.85])
        mock_box.xyxy = torch.tensor([[100.0, 200.0, 150.0, 250.0]])

        mock_result.boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes.__len__ = Mock(return_value=1)
        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        geojson = service.detect_vehicles_geojson(satellite_image)

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1
        assert geojson["features"][0]["type"] == "Feature"
        assert geojson["features"][0]["geometry"]["type"] == "Polygon"
        assert geojson["features"][0]["properties"]["vehicle_class"] == "car"
        assert geojson["features"][0]["properties"]["confidence"] == pytest.approx(
            0.85, rel=1e-5
        )

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_empty_obb_boxes(self, mock_image, mock_path):
        """Test handling OBB result with empty boxes."""
        service = VehicleDetectionService()

        mock_model = Mock()
        mock_model.names = {0: "car"}

        # Mock OBB detection with empty boxes
        mock_result = Mock()
        mock_result.masks = None
        mock_result.obb = Mock()
        mock_result.obb.__len__ = Mock(return_value=0)  # Empty boxes

        mock_model.return_value = [mock_result]
        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        detections = service.detect_vehicles(satellite_image)

        assert len(detections) == 0

    @patch("parcel_ai_json.vehicle_detector.Path")
    @patch("PIL.Image")
    def test_detect_vehicles_obb_filter_non_vehicles(self, mock_image, mock_path):
        """Test filtering non-vehicles in OBB format."""
        service = VehicleDetectionService()

        mock_model = Mock()
        mock_model.names = {0: "person", 1: "car"}

        # Mock OBB detection with person and car
        mock_result = Mock()
        mock_result.masks = None
        mock_result.obb = Mock()
        mock_result.obb.__len__ = Mock(return_value=2)

        # Mock tensor attributes for OBB
        mock_result.obb.cls = [
            Mock(item=Mock(return_value=0)),  # person
            Mock(item=Mock(return_value=1)),  # car
        ]
        mock_result.obb.conf = [
            Mock(item=Mock(return_value=0.95)),
            Mock(item=Mock(return_value=0.85)),
        ]
        mock_result.obb.xyxy = [
            Mock(tolist=Mock(return_value=[10, 20, 30, 40])),
            Mock(tolist=Mock(return_value=[100, 200, 150, 250])),
        ]

        mock_model.return_value = [mock_result]
        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        mock_img = Mock()
        mock_img.size = (512, 512)
        mock_image.open.return_value.__enter__.return_value = mock_img

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        detections = service.detect_vehicles(satellite_image)

        # Should only detect car, not person
        assert len(detections) == 1
        assert detections[0].class_name == "car"
