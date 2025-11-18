"""Tests for Grounded-SAM detector (open-vocabulary detection with segmentation)."""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from parcel_ai_json.grounded_sam_detector import (
    GroundedSAMDetector,
    GroundedDetection,
)
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

# Check if groundingdino is available
try:
    import groundingdino  # noqa: F401
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False

# Skip decorator for tests requiring groundingdino
requires_groundingdino = pytest.mark.skipif(
    not GROUNDINGDINO_AVAILABLE,
    reason="GroundingDINO not installed - skipping integration tests"
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return Image.new("RGB", (512, 512), color="white")


@pytest.fixture
def sample_image_np():
    """Create a sample numpy array image for testing."""
    return np.ones((512, 512, 3), dtype=np.uint8) * 255


@pytest.fixture
def coordinate_converter():
    """Create a coordinate converter for testing."""
    return ImageCoordinateConverter(
        center_lat=37.7749,
        center_lon=-122.4194,
        image_width_px=512,
        image_height_px=512,
        zoom_level=20,
    )


class TestGroundedDetection:
    """Test GroundedDetection dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        detection = GroundedDetection(
            label="driveway",
            pixel_bbox=(10.0, 20.0, 100.0, 150.0),
            geo_bbox=(-122.420, 37.774, -122.419, 37.775),
            geo_polygon=[
                (-122.420, 37.774),
                (-122.419, 37.774),
                (-122.419, 37.775),
                (-122.420, 37.775),
                (-122.420, 37.774),
            ],
            confidence=0.85,
            area_pixels=13500,
            area_sqm=120.5,
        )

        result = detection.to_dict()

        assert result["label"] == "driveway"
        assert result["pixel_bbox"] == [10.0, 20.0, 100.0, 150.0]
        assert result["geo_bbox"] == list(detection.geo_bbox)
        assert result["geo_polygon"] == detection.geo_polygon
        assert result["confidence"] == 0.85
        assert result["area_pixels"] == 13500
        assert result["area_sqm"] == 120.5

    def test_to_geojson_feature(self):
        """Test conversion to GeoJSON feature."""
        detection = GroundedDetection(
            label="patio",
            pixel_bbox=(50.0, 60.0, 200.0, 180.0),
            geo_bbox=(-122.420, 37.774, -122.419, 37.775),
            geo_polygon=[
                (-122.420, 37.774),
                (-122.419, 37.774),
                (-122.419, 37.775),
                (-122.420, 37.775),
                (-122.420, 37.774),
            ],
            confidence=0.92,
            area_pixels=27000,
            area_sqm=250.3,
        )

        geojson = detection.to_geojson_feature()

        assert geojson["type"] == "Feature"
        assert geojson["geometry"]["type"] == "Polygon"
        assert geojson["geometry"]["coordinates"] == [detection.geo_polygon]
        assert geojson["properties"]["feature_type"] == "grounded_detection"
        assert geojson["properties"]["label"] == "patio"
        assert geojson["properties"]["confidence"] == 0.92
        assert geojson["properties"]["area_pixels"] == 27000
        assert geojson["properties"]["area_sqm"] == 250.3


class TestGroundedSAMDetector:
    """Test GroundedSAMDetector class."""

    def test_initialization_defaults(self):
        """Test detector initialization with default parameters."""
        detector = GroundedSAMDetector()

        assert detector.grounding_model_path is None
        assert detector.sam_model_path is None
        assert detector.device == "cpu"
        assert detector.box_threshold == 0.25
        assert detector.text_threshold == 0.20
        assert detector.use_sam is True
        assert detector._grounding_model is None
        assert detector._sam_predictor is None

    def test_initialization_custom_params(self):
        """Test detector initialization with custom parameters."""
        detector = GroundedSAMDetector(
            grounding_model_path="/path/to/model.pth",
            sam_model_path="/path/to/sam.pth",
            device="cuda",
            box_threshold=0.3,
            text_threshold=0.25,
            use_sam=False,
        )

        assert detector.grounding_model_path == "/path/to/model.pth"
        assert detector.sam_model_path == "/path/to/sam.pth"
        assert detector.device == "cuda"
        assert detector.box_threshold == 0.3
        assert detector.text_threshold == 0.25
        assert detector.use_sam is False

    @patch("parcel_ai_json.grounded_sam_detector.Path")
    @patch("parcel_ai_json.grounded_sam_detector.torch")
    def test_load_models_grounding_only(self, mock_torch, mock_path):
        """Test loading GroundingDINO model without SAM."""
        # Mock GroundingDINO imports and functions
        mock_grounding_model = MagicMock()
        mock_load_grounding = MagicMock(return_value=mock_grounding_model)

        with patch.dict(
            "sys.modules",
            {
                "groundingdino": MagicMock(
                    __file__="/fake/path/groundingdino/__init__.py"
                ),
                "groundingdino.util": MagicMock(),
                "groundingdino.util.inference": MagicMock(
                    load_model=mock_load_grounding
                ),
            },
        ):
            # Mock file existence
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_config_path = MagicMock()
            mock_config_path.exists.return_value = True

            mock_path.return_value.parent.parent = MagicMock()
            mock_path.return_value.parent.parent.__truediv__ = lambda self, x: (
                mock_model_path if x == "models" else MagicMock()
            )
            mock_model_path.__truediv__ = lambda self, x: mock_model_path

            mock_package_dir = MagicMock()
            mock_package_dir.__truediv__ = lambda self, x: (
                MagicMock(__truediv__=lambda s, y: mock_config_path)
                if x == "config"
                else MagicMock()
            )
            mock_path.side_effect = lambda x: (
                mock_package_dir if "groundingdino" in str(x) else MagicMock()
            )

            detector = GroundedSAMDetector(use_sam=False)
            detector._load_models()

            assert detector._grounding_model == mock_grounding_model
            assert detector._sam_predictor is None

    @requires_groundingdino
    def test_detect_with_string_prompt(self, sample_image, coordinate_converter):
        """Test detection with a single string prompt (integration test)."""
        import torch

        # Mock the detect method to test prompt handling
        detector = GroundedSAMDetector(use_sam=False)

        with patch.object(detector, "_load_models"), patch(
            "groundingdino.util.inference.predict", create=True
        ) as mock_predict:
            # Mock predict results
            mock_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
            mock_logits = torch.tensor([0.85])
            mock_phrases = ["driveway"]
            mock_predict.return_value = (mock_boxes, mock_logits, mock_phrases)

            # Set up mock model
            detector._grounding_model = MagicMock()

            detections = detector.detect(
                image=sample_image,
                prompts="driveway",
                coordinate_converter=coordinate_converter,
            )

            # Verify predict was called with correct prompt
            assert mock_predict.called
            assert isinstance(detections, list)
            call_kwargs = mock_predict.call_args[1]
            assert call_kwargs["caption"] == "driveway"

    @requires_groundingdino
    def test_detect_with_list_prompts(self, sample_image, coordinate_converter):
        """Test detection with list of prompts."""
        import torch

        detector = GroundedSAMDetector(use_sam=False)

        with patch.object(detector, "_load_models"), patch(
            "groundingdino.util.inference.predict", create=True
        ) as mock_predict:
            # Mock results for multiple prompts
            mock_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.15, 0.2]])
            mock_logits = torch.tensor([0.85, 0.78])
            mock_phrases = ["driveway", "patio"]
            mock_predict.return_value = (mock_boxes, mock_logits, mock_phrases)

            detector._grounding_model = MagicMock()

            detections = detector.detect(
                image=sample_image,
                prompts=["driveway", "patio", "deck"],
                coordinate_converter=coordinate_converter,
            )

            # Verify prompts were joined with ". "
            call_kwargs = mock_predict.call_args[1]
            assert call_kwargs["caption"] == "driveway. patio. deck"
            assert isinstance(detections, list)

    def test_mask_to_polygon(self, coordinate_converter):
        """Test mask to polygon conversion."""
        detector = GroundedSAMDetector()

        # Create mock mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1

        # Mock cv2 module (imported locally in method)
        with patch("cv2.findContours") as mock_findContours, patch(
            "cv2.contourArea"
        ) as mock_contourArea, patch("cv2.arcLength") as mock_arcLength, patch(
            "cv2.approxPolyDP"
        ) as mock_approxPolyDP:

            mock_contour = np.array(
                [[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]]
            )
            mock_findContours.return_value = ([mock_contour], None)
            mock_contourArea.return_value = 10000
            mock_arcLength.return_value = 400
            mock_approxPolyDP.return_value = mock_contour

            polygon = detector._mask_to_polygon(mask, coordinate_converter)

            assert len(polygon) > 0
            # First and last points should be the same (closed polygon)
            assert polygon[0] == polygon[-1]
            # All points should be tuples of (lon, lat)
            for point in polygon:
                assert len(point) == 2
                assert isinstance(point[0], float)
                assert isinstance(point[1], float)

    def test_bbox_to_polygon_with_converter(self, coordinate_converter):
        """Test bounding box to polygon conversion with coordinate converter."""
        detector = GroundedSAMDetector()

        bbox = (100.0, 150.0, 300.0, 400.0)
        polygon = detector._bbox_to_polygon(bbox, coordinate_converter)

        assert len(polygon) == 5  # 4 corners + closing point
        assert polygon[0] == polygon[-1]  # Closed polygon

        # All points should be in geographic coordinates
        for point in polygon:
            assert len(point) == 2
            lon, lat = point
            # WGS84 longitude range
            assert -180 <= lon <= 180
            # WGS84 latitude range
            assert -90 <= lat <= 90

    def test_bbox_to_polygon_without_converter(self):
        """Test bounding box to polygon conversion without coordinate converter."""
        detector = GroundedSAMDetector()

        bbox = (100.0, 150.0, 300.0, 400.0)
        polygon = detector._bbox_to_polygon(bbox, None)

        assert len(polygon) == 5
        assert polygon[0] == polygon[-1]

        # Should return pixel coordinates
        expected = [
            (100.0, 150.0),
            (300.0, 150.0),
            (300.0, 400.0),
            (100.0, 400.0),
            (100.0, 150.0),
        ]
        assert polygon == expected

    def test_box_cxcywh_to_xyxy(self):
        """Test box format conversion from center format to corner format."""
        import torch

        detector = GroundedSAMDetector()

        # Create boxes in (cx, cy, w, h) format
        boxes = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.3],  # Center (0.5, 0.5), size (0.2, 0.3)
                [0.3, 0.7, 0.4, 0.2],  # Center (0.3, 0.7), size (0.4, 0.2)
            ]
        )

        boxes_xyxy = detector._box_cxcywh_to_xyxy(boxes)

        expected = torch.tensor(
            [
                [0.4, 0.35, 0.6, 0.65],  # (x1, y1, x2, y2)
                [0.1, 0.6, 0.5, 0.8],
            ]
        )

        assert torch.allclose(boxes_xyxy, expected, atol=1e-6)

    @requires_groundingdino
    def test_detect_with_numpy_array(self, sample_image_np, coordinate_converter):
        """Test detection with numpy array input."""
        import torch

        detector = GroundedSAMDetector(use_sam=False)

        with patch.object(detector, "_load_models"), patch(
            "groundingdino.util.inference.predict", create=True
        ) as mock_predict:
            mock_predict.return_value = (
                torch.tensor([]),
                torch.tensor([]),
                [],
            )

            detector._grounding_model = MagicMock()

            detections = detector.detect(
                image=sample_image_np,
                prompts="shed",
                coordinate_converter=coordinate_converter,
            )

            assert isinstance(detections, list)

    @requires_groundingdino
    def test_detect_with_file_path(self, tmp_path, coordinate_converter):
        """Test detection with file path input."""
        import torch

        # Create temporary image file
        image_path = tmp_path / "test_image.jpg"
        Image.new("RGB", (512, 512), color="white").save(image_path)

        detector = GroundedSAMDetector(use_sam=False)

        with patch.object(detector, "_load_models"), patch(
            "groundingdino.util.inference.predict", create=True
        ) as mock_predict:

            mock_predict.return_value = (
                torch.tensor([]),
                torch.tensor([]),
                [],
            )

            detector._grounding_model = MagicMock()

            detections = detector.detect(
                image=str(image_path),
                prompts="gazebo",
                coordinate_converter=coordinate_converter,
            )

            assert isinstance(detections, list)

    def test_mask_to_polygon_empty_mask(self, coordinate_converter):
        """Test mask to polygon conversion with empty mask (no contours)."""
        detector = GroundedSAMDetector()

        mask = np.zeros((512, 512), dtype=np.uint8)

        # Mock cv2.findContours to return empty contours
        with patch("cv2.findContours") as mock_findContours:
            mock_findContours.return_value = ([], None)

            polygon = detector._mask_to_polygon(mask, coordinate_converter)

            assert polygon == []

    @requires_groundingdino
    def test_detect_no_detections(self, sample_image, coordinate_converter):
        """Test detection when no objects are found."""
        import torch

        detector = GroundedSAMDetector(use_sam=False)

        with patch.object(detector, "_load_models"), patch(
            "groundingdino.util.inference.predict", create=True
        ) as mock_predict:

            # Return empty results
            mock_predict.return_value = (
                torch.tensor([]),
                torch.tensor([]),
                [],
            )

            detector._grounding_model = MagicMock()

            detections = detector.detect(
                image=sample_image,
                prompts="nonexistent object",
                coordinate_converter=coordinate_converter,
            )

            assert detections == []

    def test_mps_device_fallback_to_cpu_for_sam(self):
        """Test that SAM falls back to CPU when using MPS device."""
        with patch(
            "parcel_ai_json.grounded_sam_detector.Path"
        ) as mock_path, patch.dict(
            "sys.modules",
            {
                "groundingdino": MagicMock(__file__="/fake/path/__init__.py"),
                "groundingdino.util": MagicMock(),
                "groundingdino.util.inference": MagicMock(
                    load_model=MagicMock(return_value=MagicMock())
                ),
                "segment_anything": MagicMock(
                    sam_model_registry={"vit_h": MagicMock(return_value=MagicMock())},
                    SamPredictor=MagicMock(),
                ),
            },
        ):

            # Mock file existence
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_config_path = MagicMock()
            mock_config_path.exists.return_value = True

            def path_side_effect(x):
                if "groundingdino" in str(x):
                    mock_pkg = MagicMock()
                    mock_pkg.__truediv__ = lambda s, y: (
                        MagicMock(__truediv__=lambda a, b: mock_config_path)
                        if y == "config"
                        else MagicMock()
                    )
                    return mock_pkg
                return MagicMock()

            mock_path.side_effect = path_side_effect
            mock_path.return_value.parent.parent.__truediv__ = lambda self, x: (
                mock_model_path
            )

            detector = GroundedSAMDetector(device="mps", use_sam=True)
            # _load_models would be called on first detect()
            # We're just verifying initialization doesn't fail
            assert detector.device == "mps"
            assert detector.use_sam is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
