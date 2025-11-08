"""Tests for VehicleDetectionService."""

import pytest
from unittest.mock import Mock, patch
import torch

from parcel_geojson.services.vehicle_detector import (
    VehicleDetection,
    VehicleDetectionService,
)
from parcel_geojson.core.image_coordinates import ImageCoordinateConverter


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
        """Test loading default YOLO model (yolov8n)."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = VehicleDetectionService()
        service._load_model()

        mock_yolo.assert_called_once_with("yolov8n.pt")

    @patch("ultralytics.YOLO")
    @patch("parcel_geojson.services.vehicle_detector.Path")
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

    @patch("parcel_geojson.services.vehicle_detector.Path")
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
            with pytest.raises(ImportError, match="Vehicle detection requires ultralytics"):
                service._load_model()

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_regular_bbox(self, mock_path):
        """Test detecting vehicles with regular bounding box model."""
        # Create service and mock the model directly
        service = VehicleDetectionService(model_path="yolov8m.pt")

        # Mock the model
        mock_model = Mock()
        mock_model.names = {0: "car", 1: "truck"}

        # Mock detection results (regular bbox format)
        mock_result = Mock()
        mock_result.obb = None
        mock_result.boxes = Mock()
        mock_result.orig_shape = (512, 512)

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

        # Mock coordinate converter
        mock_converter = Mock(spec=ImageCoordinateConverter)
        mock_converter.image_width_px = 512
        mock_converter.image_height_px = 512
        mock_converter.bbox_to_polygon.return_value = [
            (-122.0, 37.0),
            (-122.1, 37.0),
            (-122.1, 37.1),
            (-122.0, 37.1),
        ]

        detections = service.detect_vehicles("/path/to/image.jpg", mock_converter)

        assert len(detections) == 1
        assert detections[0].class_name == "car"
        assert detections[0].confidence == pytest.approx(0.85, rel=1e-5)
        assert detections[0].pixel_bbox == (100.0, 200.0, 150.0, 250.0)
        assert len(detections[0].geo_polygon) == 4

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_obb_format(self, mock_path):
        """Test detecting vehicles with OBB (Oriented Bounding Box) model."""
        # Create service and mock the model directly
        service = VehicleDetectionService(model_path="yolov8m-obb.pt")

        # Mock the model
        mock_model = Mock()
        mock_model.names = {10: "small vehicle", 11: "large vehicle"}

        # Mock detection results (OBB format)
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([10, 11])  # small and large vehicle
        mock_result.obb.conf = torch.tensor([0.75, 0.65])
        mock_result.obb.xyxy = torch.tensor(
            [[50.0, 60.0, 100.0, 110.0], [200.0, 210.0, 250.0, 260.0]]
        )
        mock_result.orig_shape = (512, 512)

        # Make OBB iterable and have length
        mock_result.obb.__len__ = Mock(return_value=2)

        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        # Mock image file existence
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Mock coordinate converter
        mock_converter = Mock(spec=ImageCoordinateConverter)
        mock_converter.image_width_px = 512
        mock_converter.image_height_px = 512
        mock_converter.bbox_to_polygon.return_value = [(-122.0, 37.0)]

        detections = service.detect_vehicles("/path/to/image.jpg", mock_converter)

        assert len(detections) == 2
        assert detections[0].class_name == "small vehicle"
        assert detections[0].confidence == pytest.approx(0.75, rel=1e-5)
        assert detections[1].class_name == "large vehicle"
        assert detections[1].confidence == pytest.approx(0.65, rel=1e-5)

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_filter_non_vehicles(self, mock_path):
        """Test that non-vehicle detections are filtered out."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        # Mock the model
        mock_model = Mock()
        mock_model.names = {0: "person", 1: "car", 2: "bicycle"}

        # Mock detection results
        mock_result = Mock()
        mock_result.obb = None
        mock_result.boxes = Mock()
        mock_result.orig_shape = (512, 512)

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

        # Mock file and converter
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        mock_converter = Mock(spec=ImageCoordinateConverter)
        mock_converter.image_width_px = 512
        mock_converter.image_height_px = 512
        mock_converter.bbox_to_polygon.return_value = [(-122.0, 37.0)]

        detections = service.detect_vehicles("/path/to/image.jpg", mock_converter)

        # Should only detect the car, not person or bicycle
        assert len(detections) == 1
        assert detections[0].class_name == "car"

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_no_detections(self, mock_path):
        """Test detecting vehicles when no vehicles are found."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        mock_model = Mock()
        mock_model.names = {0: "car"}

        # Mock empty detection results
        mock_result = Mock()
        mock_result.obb = None
        mock_result.boxes = None
        mock_result.orig_shape = (512, 512)

        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        mock_converter = Mock(spec=ImageCoordinateConverter)
        mock_converter.image_width_px = 512
        mock_converter.image_height_px = 512

        detections = service.detect_vehicles("/path/to/image.jpg", mock_converter)

        assert len(detections) == 0

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_image_not_found(self, mock_path):
        """Test detecting vehicles when image file doesn't exist."""
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj

        mock_converter = Mock(spec=ImageCoordinateConverter)

        service = VehicleDetectionService()

        with pytest.raises(FileNotFoundError, match="Image not found"):
            service.detect_vehicles("/nonexistent/image.jpg", mock_converter)

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_with_scaling(self, mock_path):
        """Test vehicle detection with image scaling."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        # Mock the model
        mock_model = Mock()
        mock_model.names = {0: "car"}

        # Mock detection results with different image size
        mock_result = Mock()
        mock_result.obb = None
        mock_result.boxes = Mock()
        mock_result.orig_shape = (256, 256)  # YOLO resized to 256x256

        mock_box = Mock()
        mock_box.cls = torch.tensor([0])
        mock_box.conf = torch.tensor([0.85])
        mock_box.xyxy = torch.tensor([[50.0, 60.0, 75.0, 85.0]])  # In 256x256 space

        mock_result.boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes.__len__ = Mock(return_value=1)
        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        # Original image is 512x512
        mock_converter = Mock(spec=ImageCoordinateConverter)
        mock_converter.image_width_px = 512
        mock_converter.image_height_px = 512
        mock_converter.bbox_to_polygon.return_value = [(-122.0, 37.0)]

        detections = service.detect_vehicles("/path/to/image.jpg", mock_converter)

        # Coordinates should be scaled by 2x
        assert detections[0].pixel_bbox == (100.0, 120.0, 150.0, 170.0)

    @patch("parcel_geojson.services.vehicle_detector.Path")
    def test_detect_vehicles_from_metadata(self, mock_path):
        """Test detecting vehicles from satellite image metadata."""
        # Create service and mock the model directly
        service = VehicleDetectionService()

        mock_model = Mock()
        mock_model.names = {0: "car"}

        # Mock detection
        mock_result = Mock()
        mock_result.obb = None
        mock_result.boxes = Mock()
        mock_result.orig_shape = (512, 512)

        mock_box = Mock()
        mock_box.cls = torch.tensor([0])
        mock_box.conf = torch.tensor([0.85])
        mock_box.xyxy = torch.tensor([[100.0, 200.0, 150.0, 250.0]])

        mock_result.boxes.__iter__ = Mock(return_value=iter([mock_box]))
        mock_result.boxes.__len__ = Mock(return_value=1)
        mock_model.return_value = [mock_result]

        # Set the model directly on the service
        service._model = mock_model

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
            "width_px": 512,
            "height_px": 512,
        }

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        detections = service.detect_vehicles_from_metadata(satellite_image)

        assert len(detections) == 1
        assert detections[0].class_name == "car"
        assert detections[0].confidence == pytest.approx(0.85, rel=1e-5)
