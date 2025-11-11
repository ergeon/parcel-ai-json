"""Tests for SwimmingPoolDetectionService."""

import pytest
from unittest.mock import Mock, patch
import torch

from parcel_ai_json.swimming_pool_detector import (
    SwimmingPoolDetection,
    SwimmingPoolDetectionService,
)


class TestSwimmingPoolDetection:
    """Test SwimmingPoolDetection dataclass."""

    def test_swimming_pool_detection_creation(self):
        """Test creating a SwimmingPoolDetection object."""
        detection = SwimmingPoolDetection(
            pixel_bbox=(10.0, 20.0, 50.0, 60.0),
            geo_polygon=[(-122.0, 37.0), (-122.1, 37.0), (-122.1, 37.1)],
            confidence=0.85,
            area_sqm=45.5,
        )

        assert detection.pixel_bbox == (10.0, 20.0, 50.0, 60.0)
        assert len(detection.geo_polygon) == 3
        assert detection.confidence == 0.85
        assert detection.area_sqm == 45.5

    def test_swimming_pool_detection_to_dict(self):
        """Test converting SwimmingPoolDetection to dict."""
        detection = SwimmingPoolDetection(
            pixel_bbox=(10.0, 20.0, 50.0, 60.0),
            geo_polygon=[(-122.0, 37.0), (-122.1, 37.0)],
            confidence=0.75,
            area_sqm=32.0,
        )

        result = detection.to_dict()

        assert result["pixel_bbox"] == [10.0, 20.0, 50.0, 60.0]
        assert result["geo_polygon"] == [(-122.0, 37.0), (-122.1, 37.0)]
        assert result["confidence"] == 0.75
        assert result["area_sqm"] == 32.0

    def test_swimming_pool_detection_to_geojson(self):
        """Test converting SwimmingPoolDetection to GeoJSON feature."""
        detection = SwimmingPoolDetection(
            pixel_bbox=(10.0, 20.0, 50.0, 60.0),
            geo_polygon=[
                (-122.0, 37.0),
                (-122.1, 37.0),
                (-122.1, 37.1),
                (-122.0, 37.1),
                (-122.0, 37.0),
            ],
            confidence=0.85,
            area_sqm=50.0,
        )

        geojson = detection.to_geojson_feature()

        assert geojson["type"] == "Feature"
        assert geojson["geometry"]["type"] == "Polygon"
        assert geojson["geometry"]["coordinates"] == [detection.geo_polygon]
        assert geojson["properties"]["feature_type"] == "swimming_pool"
        assert geojson["properties"]["confidence"] == 0.85
        assert geojson["properties"]["area_sqm"] == 50.0
        assert geojson["properties"]["pixel_bbox"] == [10.0, 20.0, 50.0, 60.0]


class TestSwimmingPoolDetectionService:
    """Test SwimmingPoolDetectionService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = SwimmingPoolDetectionService(
            model_path="yolov8m-obb.pt", confidence_threshold=0.3, device="cpu"
        )

    def test_initialization(self):
        """Test SwimmingPoolDetectionService initialization."""
        service = SwimmingPoolDetectionService(
            model_path="yolov8m-obb.pt", confidence_threshold=0.5, device="cpu"
        )

        assert service.model_path == "yolov8m-obb.pt"
        assert service.confidence_threshold == 0.5
        assert service.device == "cpu"
        assert service._model is None

    def test_initialization_defaults(self):
        """Test SwimmingPoolDetectionService with default parameters."""
        service = SwimmingPoolDetectionService()

        assert service.model_path is None
        assert service.confidence_threshold == 0.3
        assert service.device == "cpu"
        assert service._model is None

    @patch("ultralytics.YOLO")
    def test_load_model_custom_path(self, mock_yolo):
        """Test loading custom model."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = SwimmingPoolDetectionService(model_path="custom-obb.pt")
        service._load_model()

        mock_yolo.assert_called_once_with("custom-obb.pt")
        mock_model.to.assert_called_once_with("cpu")
        assert service._model == mock_model

    @patch("ultralytics.YOLO")
    def test_load_model_default(self, mock_yolo):
        """Test loading default YOLO model (yolov8m-obb)."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = SwimmingPoolDetectionService()
        service._load_model()

        # Check that YOLO was called with either short or full path to yolov8m-obb.pt
        call_args = mock_yolo.call_args[0][0]
        assert call_args == "yolov8m-obb.pt" or call_args.endswith(
            "models/yolov8m-obb.pt"
        ), f"Expected YOLO to be called with yolov8m-obb.pt or full path, got: {call_args}"

    @patch("ultralytics.YOLO")
    def test_load_model_only_once(self, mock_yolo):
        """Test that model is loaded only once (lazy loading)."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        service = SwimmingPoolDetectionService()
        service._load_model()
        service._load_model()  # Second call should not reload

        mock_yolo.assert_called_once()

    def test_load_model_missing_ultralytics(self):
        """Test error when ultralytics is not installed."""
        service = SwimmingPoolDetectionService()

        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(
                ImportError, match="Swimming pool detection requires ultralytics"
            ):
                service._load_model()

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_detect_swimming_pools_obb_format(self, mock_image, mock_path):
        """Test detecting swimming pools with OBB (Oriented Bounding Box) model."""
        # Create service and mock the model directly
        service = SwimmingPoolDetectionService(model_path="yolov8m-obb.pt")

        # Mock the model
        mock_model = Mock()
        # DOTA dataset: class 14 is "swimming pool"
        mock_model.names = {14: "swimming pool", 10: "small vehicle"}

        # Mock detection results (OBB format)
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([14, 14])  # two swimming pools
        mock_result.obb.conf = torch.tensor([0.85, 0.75])
        mock_result.obb.xyxy = torch.tensor(
            [[50.0, 60.0, 100.0, 110.0], [200.0, 210.0, 280.0, 290.0]]
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

        detections = service.detect_swimming_pools(satellite_image)

        assert len(detections) == 2
        assert detections[0].confidence == pytest.approx(0.85, rel=1e-5)
        assert len(detections[0].geo_polygon) == 5  # Polygon is closed
        assert detections[0].area_sqm > 0
        assert detections[1].confidence == pytest.approx(0.75, rel=1e-5)
        assert len(detections[1].geo_polygon) == 5
        assert detections[1].area_sqm > 0

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_detect_swimming_pools_filter_non_pools(self, mock_image, mock_path):
        """Test that non-pool detections are filtered out."""
        # Create service and mock the model directly
        service = SwimmingPoolDetectionService()

        # Mock the model
        mock_model = Mock()
        # DOTA classes: 14 = swimming pool, 10 = small vehicle, 0 = plane
        mock_model.names = {0: "plane", 10: "small vehicle", 14: "swimming pool"}

        # Mock detection results with mixed classes
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([0, 14, 10])  # plane, pool, vehicle
        mock_result.obb.conf = torch.tensor([0.95, 0.85, 0.75])
        mock_result.obb.xyxy = torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0],  # plane
                [50.0, 60.0, 100.0, 110.0],  # pool
                [200.0, 210.0, 250.0, 260.0],  # vehicle
            ]
        )

        mock_result.obb.__len__ = Mock(return_value=3)
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

        detections = service.detect_swimming_pools(satellite_image)

        # Should only detect the swimming pool, not plane or vehicle
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.85, rel=1e-5)

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_detect_swimming_pools_no_detections(self, mock_image, mock_path):
        """Test detecting swimming pools when no pools are found."""
        # Create service and mock the model directly
        service = SwimmingPoolDetectionService()

        mock_model = Mock()
        mock_model.names = {14: "swimming pool"}

        # Mock empty detection results
        mock_result = Mock()
        mock_result.obb = None

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

        detections = service.detect_swimming_pools(satellite_image)

        assert len(detections) == 0

    def test_detect_swimming_pools_image_not_found(self):
        """Test detecting swimming pools when image file doesn't exist."""
        service = SwimmingPoolDetectionService()

        satellite_image = {
            "path": "/nonexistent/definitely_does_not_exist_12345.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
        }

        with pytest.raises(FileNotFoundError, match="Image not found"):
            service.detect_swimming_pools(satellite_image)

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_detect_swimming_pools_geojson(self, mock_image, mock_path):
        """Test detect_swimming_pools_geojson method."""
        # Create service and mock the model directly
        service = SwimmingPoolDetectionService()

        mock_model = Mock()
        mock_model.names = {14: "swimming pool"}

        # Mock detection
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([14])
        mock_result.obb.conf = torch.tensor([0.85])
        mock_result.obb.xyxy = torch.tensor([[50.0, 60.0, 100.0, 110.0]])

        mock_result.obb.__len__ = Mock(return_value=1)
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

        geojson = service.detect_swimming_pools_geojson(satellite_image)

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1
        assert geojson["features"][0]["type"] == "Feature"
        assert geojson["features"][0]["geometry"]["type"] == "Polygon"
        assert geojson["features"][0]["properties"]["feature_type"] == "swimming_pool"
        assert geojson["features"][0]["properties"]["confidence"] == pytest.approx(
            0.85, rel=1e-5
        )
        assert geojson["features"][0]["properties"]["area_sqm"] > 0

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_detect_swimming_pools_with_dimensions(self, mock_image, mock_path):
        """Test detecting swimming pools with pre-specified image dimensions."""
        service = SwimmingPoolDetectionService()

        mock_model = Mock()
        mock_model.names = {14: "swimming pool"}

        # Mock detection
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([14])
        mock_result.obb.conf = torch.tensor([0.75])
        mock_result.obb.xyxy = torch.tensor([[50.0, 60.0, 100.0, 110.0]])

        mock_result.obb.__len__ = Mock(return_value=1)
        mock_model.return_value = [mock_result]

        service._model = mock_model

        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path.return_value = mock_path_obj

        satellite_image = {
            "path": "/path/to/image.jpg",
            "center_lat": 37.0,
            "center_lon": -122.0,
            "width_px": 640,
            "height_px": 640,
        }

        detections = service.detect_swimming_pools(satellite_image)

        assert len(detections) == 1
        # Should not have called PIL.Image.open since dimensions were provided
        mock_image.open.assert_not_called()

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_area_calculation(self, mock_image, mock_path):
        """Test that area calculation is approximately correct."""
        service = SwimmingPoolDetectionService()

        mock_model = Mock()
        mock_model.names = {14: "swimming pool"}

        # Create a detection with known dimensions
        # At zoom 20, meters_per_pixel ≈ 0.149 meters/pixel
        # A 50x50 pixel pool should be roughly 7.45 x 7.45 = ~55.5 sqm
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.cls = torch.tensor([14])
        mock_result.obb.conf = torch.tensor([0.85])
        mock_result.obb.xyxy = torch.tensor(
            [[200.0, 200.0, 250.0, 250.0]]  # 50x50 pixel box
        )

        mock_result.obb.__len__ = Mock(return_value=1)
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
            "zoom_level": 20,
        }

        detections = service.detect_swimming_pools(satellite_image)

        assert len(detections) == 1
        # Area should be reasonable for a 50x50 pixel pool at zoom 20
        # At latitude 37°, meters_per_pixel ≈ 0.119, so 50*0.119 ≈ 5.95, area ≈ 35.4
        assert 30 < detections[0].area_sqm < 45  # Allow some margin

    @patch("parcel_ai_json.swimming_pool_detector.Path")
    @patch("PIL.Image")
    def test_detect_swimming_pools_empty_obb(self, mock_image, mock_path):
        """Test detecting swimming pools with empty OBB results."""
        service = SwimmingPoolDetectionService()

        mock_model = Mock()
        mock_model.names = {14: "swimming pool"}

        # Mock detection with empty OBB
        mock_result = Mock()
        mock_result.obb = Mock()
        mock_result.obb.__len__ = Mock(return_value=0)

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

        detections = service.detect_swimming_pools(satellite_image)

        assert len(detections) == 0
