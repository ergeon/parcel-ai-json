"""Tests for SAM3 segmentation service."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
from PIL import Image

# Add sam3 to path for testing (before imports)
sam3_path = Path(__file__).parent.parent / "models" / "sam3"
if sam3_path.exists() and str(sam3_path) not in sys.path:
    sys.path.insert(0, str(sam3_path))

from parcel_ai_json.sam3_segmentation import (  # noqa: E402
    SAM3SegmentationService,
    SAM3Detection,
)


@pytest.fixture
def mock_sam3_model():
    """Mock SAM3 model and processor."""
    with patch('sam3.model_builder.build_sam3_image_model') as mock_build, \
         patch('sam3.model.sam3_image_processor.Sam3Processor') as mock_processor:

        # Mock model
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_build.return_value = mock_model

        # Mock processor
        mock_proc_instance = Mock()

        # Mock set_image
        mock_state = Mock()
        mock_proc_instance.set_image = Mock(return_value=mock_state)

        # Mock set_text_prompt - returns detection results
        mock_output = {
            "masks": [
                Mock(squeeze=Mock(return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.ones((100, 100), dtype=bool))))))),
                Mock(squeeze=Mock(return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.ones((100, 100), dtype=bool) * 0.5)))))),
            ],
            "boxes": [
                Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([10, 10, 50, 50]))))),
                Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([60, 60, 90, 90]))))),
            ],
            "scores": [0.95, 0.35],
        }
        mock_proc_instance.set_text_prompt = Mock(return_value=mock_output)

        mock_processor.return_value = mock_proc_instance

        yield mock_build, mock_processor, mock_proc_instance


@pytest.fixture
def test_image():
    """Create a test satellite image."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a simple test image
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        img.save(f.name, 'JPEG')
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def satellite_image_metadata(test_image):
    """Create satellite image metadata."""
    return {
        "path": test_image,
        "center_lat": 37.7749,
        "center_lon": -122.4194,
        "zoom_level": 20
    }


class TestSAM3Detection:
    """Test SAM3Detection dataclass."""

    def test_to_dict(self):
        """Test converting detection to dictionary."""
        detection = SAM3Detection(
            detection_id=0,
            class_name="house",
            confidence=0.95,
            pixel_mask=np.ones((100, 100)),
            pixel_bbox=(10, 10, 50, 50),
            geo_polygon=[(-122.42, 37.77), (-122.41, 37.77), (-122.41, 37.78)],
            area_pixels=1600,
            area_sqm=120.5,
        )

        result = detection.to_dict()

        assert result["detection_id"] == 0
        assert result["class_name"] == "house"
        assert result["confidence"] == 0.95
        assert result["pixel_bbox"] == [10, 10, 50, 50]
        assert result["area_pixels"] == 1600
        assert result["area_sqm"] == 120.5

    def test_to_geojson_feature(self):
        """Test converting detection to GeoJSON feature."""
        detection = SAM3Detection(
            detection_id=0,
            class_name="house",
            confidence=0.95,
            pixel_mask=np.ones((100, 100)),
            pixel_bbox=(10, 10, 50, 50),
            geo_polygon=[(-122.42, 37.77), (-122.41, 37.77), (-122.41, 37.78)],
            area_pixels=1600,
            area_sqm=120.5,
        )

        feature = detection.to_geojson_feature()

        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"
        assert feature["properties"]["feature_type"] == "sam3_detection"
        assert feature["properties"]["class_name"] == "house"
        assert feature["properties"]["confidence"] == 0.95


class TestSAM3SegmentationService:
    """Test SAM3SegmentationService."""

    def test_init_requires_hf_token(self):
        """Test that initialization requires HuggingFace token."""
        # Remove HF tokens from environment
        old_hf_token = os.environ.pop('HF_TOKEN', None)
        old_hf_hub_token = os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)

        try:
            with pytest.raises(ValueError, match="HuggingFace token required"):
                SAM3SegmentationService()
        finally:
            # Restore tokens
            if old_hf_token:
                os.environ['HF_TOKEN'] = old_hf_token
            if old_hf_hub_token:
                os.environ['HUGGING_FACE_HUB_TOKEN'] = old_hf_hub_token

    def test_init_with_token(self, mock_sam3_model):
        """Test successful initialization with token."""
        os.environ['HF_TOKEN'] = 'test_token'

        service = SAM3SegmentationService(
            device='cpu',
            confidence_threshold=0.3
        )

        assert service.confidence_threshold == 0.3
        assert service._device == 'cpu'

    def test_segment_image_single_prompt(
        self, mock_sam3_model, satellite_image_metadata
    ):
        """Test segmentation with a single prompt."""
        os.environ['HF_TOKEN'] = 'test_token'

        # Mock coordinate converter and cv2
        with patch('parcel_ai_json.sam3_segmentation.ImageCoordinateConverter') as mock_conv, \
             patch('parcel_ai_json.sam3_segmentation.cv2') as mock_cv2:

            mock_converter = Mock()
            mock_converter.pixel_to_geo = Mock(return_value=(-122.42, 37.77))
            mock_conv.from_satellite_image = Mock(return_value=mock_converter)

            # Mock cv2.findContours
            mock_contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])
            mock_cv2.findContours.return_value = ([mock_contour], None)
            mock_cv2.contourArea.return_value = 1600
            mock_cv2.arcLength.return_value = 160
            mock_cv2.approxPolyDP.return_value = mock_contour
            mock_cv2.RETR_EXTERNAL = 0
            mock_cv2.CHAIN_APPROX_SIMPLE = 1

            service = SAM3SegmentationService(device='cpu', confidence_threshold=0.3)

            results = service.segment_image(
                satellite_image_metadata,
                prompts=["houses"]
            )

            assert "houses" in results
            assert len(results["houses"]) > 0

            detection = results["houses"][0]
            assert isinstance(detection, SAM3Detection)
            assert detection.class_name == "houses"
            assert detection.confidence >= 0.3

    def test_segment_image_multiple_prompts(
        self, mock_sam3_model, satellite_image_metadata
    ):
        """Test segmentation with multiple prompts."""
        os.environ['HF_TOKEN'] = 'test_token'

        with patch('parcel_ai_json.sam3_segmentation.ImageCoordinateConverter') as mock_conv, \
             patch('parcel_ai_json.sam3_segmentation.cv2') as mock_cv2:

            mock_converter = Mock()
            mock_converter.pixel_to_geo = Mock(return_value=(-122.42, 37.77))
            mock_conv.from_satellite_image = Mock(return_value=mock_converter)

            mock_contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])
            mock_cv2.findContours.return_value = ([mock_contour], None)
            mock_cv2.contourArea.return_value = 1600
            mock_cv2.arcLength.return_value = 160
            mock_cv2.approxPolyDP.return_value = mock_contour
            mock_cv2.RETR_EXTERNAL = 0
            mock_cv2.CHAIN_APPROX_SIMPLE = 1

            service = SAM3SegmentationService(device='cpu')

            results = service.segment_image(
                satellite_image_metadata,
                prompts=["houses", "cars", "trees"]
            )

            assert len(results) == 3
            assert "houses" in results
            assert "cars" in results
            assert "trees" in results

    def test_segment_image_single(
        self, mock_sam3_model, satellite_image_metadata
    ):
        """Test segment_image_single convenience method."""
        os.environ['HF_TOKEN'] = 'test_token'

        with patch('parcel_ai_json.sam3_segmentation.ImageCoordinateConverter') as mock_conv, \
             patch('parcel_ai_json.sam3_segmentation.cv2') as mock_cv2:

            mock_converter = Mock()
            mock_converter.pixel_to_geo = Mock(return_value=(-122.42, 37.77))
            mock_conv.from_satellite_image = Mock(return_value=mock_converter)

            mock_contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])
            mock_cv2.findContours.return_value = ([mock_contour], None)
            mock_cv2.contourArea.return_value = 1600
            mock_cv2.arcLength.return_value = 160
            mock_cv2.approxPolyDP.return_value = mock_contour
            mock_cv2.RETR_EXTERNAL = 0
            mock_cv2.CHAIN_APPROX_SIMPLE = 1

            service = SAM3SegmentationService(device='cpu')

            detections = service.segment_image_single(
                satellite_image_metadata,
                prompt="houses"
            )

            assert isinstance(detections, list)
            if len(detections) > 0:
                assert all(isinstance(d, SAM3Detection) for d in detections)
                assert all(d.class_name == "houses" for d in detections)

    def test_segment_image_geojson(
        self, mock_sam3_model, satellite_image_metadata
    ):
        """Test GeoJSON output format."""
        os.environ['HF_TOKEN'] = 'test_token'

        with patch('parcel_ai_json.sam3_segmentation.ImageCoordinateConverter') as mock_conv, \
             patch('parcel_ai_json.sam3_segmentation.cv2') as mock_cv2:

            mock_converter = Mock()
            mock_converter.pixel_to_geo = Mock(return_value=(-122.42, 37.77))
            mock_conv.from_satellite_image = Mock(return_value=mock_converter)

            mock_contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])
            mock_cv2.findContours.return_value = ([mock_contour], None)
            mock_cv2.contourArea.return_value = 1600
            mock_cv2.arcLength.return_value = 160
            mock_cv2.approxPolyDP.return_value = mock_contour
            mock_cv2.RETR_EXTERNAL = 0
            mock_cv2.CHAIN_APPROX_SIMPLE = 1

            service = SAM3SegmentationService(device='cpu')

            geojson = service.segment_image_geojson(
                satellite_image_metadata,
                prompts=["houses", "cars"]
            )

            assert geojson["type"] == "FeatureCollection"
            assert "features" in geojson
            assert "properties" in geojson
            assert geojson["properties"]["model"] == "sam3"
            assert geojson["properties"]["prompts"] == ["houses", "cars"]

    def test_image_not_found(self, mock_sam3_model):
        """Test error handling for missing image."""
        os.environ['HF_TOKEN'] = 'test_token'

        service = SAM3SegmentationService(device='cpu')

        with pytest.raises(FileNotFoundError):
            service.segment_image(
                {"path": "/nonexistent/image.jpg", "center_lat": 37.7749, "center_lon": -122.4194},
                prompts=["houses"]
            )

    def test_confidence_filtering(
        self, satellite_image_metadata
    ):
        """Test that detections below confidence threshold are filtered."""
        os.environ['HF_TOKEN'] = 'test_token'

        # Mock with two detections: one above threshold, one below
        with patch('sam3.model_builder.build_sam3_image_model'), \
             patch('sam3.model.sam3_image_processor.Sam3Processor') as mock_processor, \
             patch('parcel_ai_json.sam3_segmentation.ImageCoordinateConverter') as mock_conv, \
             patch('parcel_ai_json.sam3_segmentation.cv2') as mock_cv2:

            mock_proc_instance = Mock()
            mock_proc_instance.set_image = Mock(return_value=Mock())

            # Mock output with two detections at different confidence levels
            mock_proc_instance.set_text_prompt = Mock(return_value={
                "masks": [
                    Mock(squeeze=Mock(return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.ones((100, 100), dtype=bool))))))),
                    Mock(squeeze=Mock(return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.ones((100, 100), dtype=bool))))))),
                ],
                "boxes": [
                    Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([10, 10, 50, 50]))))),
                    Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([60, 60, 90, 90]))))),
                ],
                "scores": [0.95, 0.15],  # One above 0.5, one below
            })
            mock_processor.return_value = mock_proc_instance

            mock_converter = Mock()
            mock_converter.pixel_to_geo = Mock(return_value=(-122.42, 37.77))
            mock_conv.from_satellite_image = Mock(return_value=mock_converter)

            mock_contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])
            mock_cv2.findContours.return_value = ([mock_contour], None)
            mock_cv2.contourArea.return_value = 1600
            mock_cv2.arcLength.return_value = 160
            mock_cv2.approxPolyDP.return_value = mock_contour
            mock_cv2.RETR_EXTERNAL = 0
            mock_cv2.CHAIN_APPROX_SIMPLE = 1

            service = SAM3SegmentationService(device='cpu', confidence_threshold=0.5)

            results = service.segment_image(
                satellite_image_metadata,
                prompts=["houses"]
            )

            # Should only have one detection (the one with 0.95 confidence)
            assert len(results["houses"]) == 1
            assert results["houses"][0].confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
