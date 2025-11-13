"""Tests for SAM segmentation service."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import tempfile
from pathlib import Path

from parcel_ai_json.sam_segmentation import (
    SAMSegment,
    SAMSegmentationService,
)


class TestSAMSegment(unittest.TestCase):
    """Test SAMSegment dataclass."""

    def test_sam_segment_creation(self):
        """Test creating a SAMSegment object."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1

        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]

        segment = SAMSegment(
            segment_id=0,
            pixel_mask=mask,
            pixel_bbox=(10, 10, 30, 30),
            geo_polygon=geo_polygon,
            area_pixels=400,
            area_sqm=50.5,
            stability_score=0.95,
            predicted_iou=0.88,
        )

        self.assertEqual(segment.segment_id, 0)
        self.assertEqual(segment.pixel_bbox, (10, 10, 30, 30))
        self.assertEqual(segment.area_pixels, 400)
        self.assertEqual(segment.area_sqm, 50.5)
        self.assertEqual(segment.stability_score, 0.95)
        self.assertEqual(segment.predicted_iou, 0.88)

    def test_sam_segment_to_dict(self):
        """Test converting SAMSegment to dictionary."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        geo_polygon = [(-122.4194, 37.7749), (-122.4193, 37.7749)]

        segment = SAMSegment(
            segment_id=1,
            pixel_mask=mask,
            pixel_bbox=(5, 5, 25, 25),
            geo_polygon=geo_polygon,
            area_pixels=200,
            area_sqm=25.0,
            stability_score=0.92,
            predicted_iou=0.85,
        )

        result = segment.to_dict()

        self.assertEqual(result["segment_id"], 1)
        self.assertEqual(result["pixel_bbox"], [5, 5, 25, 25])
        self.assertEqual(result["geo_polygon"], geo_polygon)
        self.assertEqual(result["area_pixels"], 200)
        self.assertEqual(result["area_sqm"], 25.0)
        self.assertEqual(result["stability_score"], 0.92)
        self.assertEqual(result["predicted_iou"], 0.85)

    def test_sam_segment_to_geojson_feature(self):
        """Test converting SAMSegment to GeoJSON feature."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]

        segment = SAMSegment(
            segment_id=2,
            pixel_mask=mask,
            pixel_bbox=(10, 10, 40, 40),
            geo_polygon=geo_polygon,
            area_pixels=900,
            area_sqm=75.5,
            stability_score=0.98,
            predicted_iou=0.91,
        )

        geojson = segment.to_geojson_feature()

        self.assertEqual(geojson["type"], "Feature")
        self.assertEqual(geojson["geometry"]["type"], "Polygon")
        self.assertEqual(geojson["geometry"]["coordinates"], [geo_polygon])
        self.assertEqual(geojson["properties"]["feature_type"], "sam_segment")
        self.assertEqual(geojson["properties"]["segment_id"], 2)
        self.assertEqual(geojson["properties"]["area_pixels"], 900)
        self.assertEqual(geojson["properties"]["area_sqm"], 75.5)
        self.assertEqual(geojson["properties"]["stability_score"], 0.98)
        self.assertEqual(geojson["properties"]["predicted_iou"], 0.91)


class TestSAMSegmentationService(unittest.TestCase):
    """Test SAMSegmentationService."""

    def test_service_initialization(self):
        """Test initializing SAM service with default parameters."""
        service = SAMSegmentationService()

        self.assertEqual(service.model_type, "vit_h")
        self.assertIsNone(service.model_path)
        self.assertEqual(service.device, "cpu")
        self.assertEqual(service.points_per_side, 32)
        self.assertEqual(service.pred_iou_thresh, 0.88)
        self.assertEqual(service.stability_score_thresh, 0.95)
        self.assertEqual(service.min_mask_region_area, 100)
        self.assertIsNone(service._sam)
        self.assertIsNone(service._mask_generator)

    def test_service_initialization_custom_params(self):
        """Test initializing SAM service with custom parameters."""
        service = SAMSegmentationService(
            model_type="vit_b",
            device="cuda",
            points_per_side=16,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.90,
            min_mask_region_area=50,
        )

        self.assertEqual(service.model_type, "vit_b")
        self.assertEqual(service.device, "cuda")
        self.assertEqual(service.points_per_side, 16)
        self.assertEqual(service.pred_iou_thresh, 0.85)
        self.assertEqual(service.stability_score_thresh, 0.90)
        self.assertEqual(service.min_mask_region_area, 50)

    @patch("parcel_ai_json.sam_segmentation.Path")
    def test_load_model_import_error(self, mock_path):
        """Test model loading handles missing segment_anything package."""
        service = SAMSegmentationService()

        with patch.dict("sys.modules", {"segment_anything": None}):
            with self.assertRaises(ImportError) as context:
                service._load_model()

            self.assertIn("segment_anything package", str(context.exception))

    def test_load_model_invalid_type(self):
        """Test model loading with invalid model type."""
        service = SAMSegmentationService(model_type="invalid_type")

        # Mock the models directory to exist but not have the checkpoint
        with patch("pathlib.Path.exists", return_value=False):
            # Mock segment_anything import
            with patch.dict(
                "sys.modules",
                {
                    "segment_anything": Mock(),
                },
            ):
                with self.assertRaises(ValueError) as context:
                    service._load_model()

                self.assertIn("Invalid model_type", str(context.exception))

    def test_calculate_area_sqm(self):
        """Test calculating polygon area in square meters."""
        service = SAMSegmentationService()

        # Create a simple rectangular polygon
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]

        area = service._calculate_area_sqm(geo_polygon)

        # Should return a positive area value
        self.assertGreater(area, 0)
        self.assertIsInstance(area, float)

    def test_calculate_area_sqm_invalid_polygon(self):
        """Test area calculation with invalid polygon."""
        service = SAMSegmentationService()

        # Polygon with < 3 points
        geo_polygon = [(-122.4194, 37.7749), (-122.4193, 37.7749)]

        area = service._calculate_area_sqm(geo_polygon)

        self.assertEqual(area, 0.0)

    def test_calculate_area_sqm_empty_polygon(self):
        """Test area calculation with empty polygon."""
        service = SAMSegmentationService()

        area = service._calculate_area_sqm([])

        self.assertEqual(area, 0.0)

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    @patch("cv2.arcLength")
    @patch("cv2.approxPolyDP")
    def test_mask_to_geo_polygon(
        self,
        mock_approx,
        mock_arc_length,
        mock_contour_area,
        mock_find_contours,
    ):
        """Test converting binary mask to geographic polygon."""
        service = SAMSegmentationService()

        # Create a simple binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1

        # Mock contour detection
        mock_contour = np.array(
            [
                [[10, 10]],
                [[30, 10]],
                [[30, 30]],
                [[10, 30]],
            ]
        )
        mock_find_contours.return_value = ([mock_contour], None)
        mock_contour_area.return_value = 400
        mock_arc_length.return_value = 80
        mock_approx.return_value = mock_contour

        # Mock coordinate converter
        mock_converter = Mock()
        mock_converter.pixel_to_geo.side_effect = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
        ]

        geo_polygon = service._mask_to_geo_polygon(mask, mock_converter)

        # Should have converted pixels to geo coordinates
        self.assertGreater(len(geo_polygon), 3)
        # First and last point should be the same (closed polygon)
        self.assertEqual(geo_polygon[0], geo_polygon[-1])

    @patch("cv2.findContours")
    def test_mask_to_geo_polygon_no_contours(self, mock_find_contours):
        """Test mask to polygon with no contours found."""
        service = SAMSegmentationService()

        mask = np.zeros((100, 100), dtype=np.uint8)
        mock_find_contours.return_value = ([], None)

        mock_converter = Mock()
        geo_polygon = service._mask_to_geo_polygon(mask, mock_converter)

        self.assertEqual(geo_polygon, [])

    @patch("cv2.findContours")
    @patch("cv2.contourArea")
    def test_mask_to_geo_polygon_too_few_points(
        self, mock_contour_area, mock_find_contours
    ):
        """Test mask to polygon with contour having too few points."""
        service = SAMSegmentationService()

        mask = np.zeros((100, 100), dtype=np.uint8)

        # Mock contour with only 2 points
        mock_contour = np.array([[[10, 10]], [[30, 10]]])
        mock_find_contours.return_value = ([mock_contour], None)
        mock_contour_area.return_value = 10

        mock_converter = Mock()
        geo_polygon = service._mask_to_geo_polygon(mask, mock_converter)

        self.assertEqual(geo_polygon, [])

    @patch("PIL.Image.open")
    @patch.object(SAMSegmentationService, "_load_model")
    def test_segment_image_file_not_found(self, mock_load_model, mock_image_open):
        """Test segmentation with non-existent image file."""
        service = SAMSegmentationService()

        satellite_image = {
            "path": "/nonexistent/image.jpg",
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        with self.assertRaises(FileNotFoundError):
            service.segment_image(satellite_image)

    @patch("PIL.Image.open")
    @patch.object(SAMSegmentationService, "_load_model")
    @patch("parcel_ai_json.coordinate_converter.ImageCoordinateConverter")
    def test_segment_image_geojson(
        self, mock_coord_converter, mock_load_model, mock_image_open
    ):
        """Test segmentation returning GeoJSON format."""
        service = SAMSegmentationService()

        # Mock image
        mock_img = Mock()
        mock_rgb = Mock()
        mock_img.convert.return_value = mock_rgb
        mock_rgb.__array__ = lambda: np.random.randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock mask generator
        mock_mask_dict = {
            "segmentation": np.ones((100, 100), dtype=np.uint8),
            "bbox": [10, 10, 20, 20],
            "area": 400,
            "stability_score": 0.95,
            "predicted_iou": 0.88,
        }
        service._mask_generator = Mock()
        service._mask_generator.generate.return_value = [mock_mask_dict]

        # Mock coordinate converter and polygon extraction
        with patch.object(service, "_mask_to_geo_polygon") as mock_mask_to_poly:
            mock_mask_to_poly.return_value = [
                (-122.4194, 37.7749),
                (-122.4193, 37.7749),
                (-122.4193, 37.7748),
                (-122.4194, 37.7749),
            ]

            with patch.object(service, "_calculate_area_sqm", return_value=50.5):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    satellite_image = {
                        "path": str(tmp_path),
                        "center_lat": 37.7749,
                        "center_lon": -122.4194,
                        "zoom_level": 20,
                    }

                    geojson = service.segment_image_geojson(satellite_image)

                    self.assertEqual(geojson["type"], "FeatureCollection")
                    self.assertIn("features", geojson)
                    self.assertIsInstance(geojson["features"], list)

                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
