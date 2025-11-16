"""Tests for SAM segment labeling service."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from shapely.geometry import Polygon

from parcel_ai_json.sam_labeler import (
    LabeledSAMSegment,
    SAMSegmentLabeler,
    LABEL_SCHEMA,
)


class TestLabelSchema(unittest.TestCase):
    """Test label schema structure."""

    def test_label_schema_exists(self):
        """Test that label schema is defined."""
        self.assertIsNotNone(LABEL_SCHEMA)
        self.assertIsInstance(LABEL_SCHEMA, dict)

    def test_label_schema_has_required_labels(self):
        """Test that schema has expected labels."""
        required_labels = [
            "vehicle",
            "pool",
            "building",
            "tree",
            "vegetation",
            "unknown",
        ]
        for label in required_labels:
            self.assertIn(label, LABEL_SCHEMA)

    def test_label_schema_properties(self):
        """Test that each label has required properties."""
        for label, props in LABEL_SCHEMA.items():
            self.assertIn("color", props)
            self.assertIn("priority", props)
            self.assertIn("min_area_sqm", props)
            self.assertIn("max_area_sqm", props)


class TestLabeledSAMSegment(unittest.TestCase):
    """Test LabeledSAMSegment dataclass."""

    def test_labeled_segment_creation(self):
        """Test creating a labeled SAM segment."""
        segment = LabeledSAMSegment(
            segment_id=1,
            pixel_mask=np.zeros((10, 10)),
            pixel_bbox=(10, 20, 30, 40),
            geo_polygon=[
                (-122.4194, 37.7749),
                (-122.4193, 37.7749),
                (-122.4193, 37.7748),
                (-122.4194, 37.7748),
                (-122.4194, 37.7749),
            ],
            area_pixels=400,
            area_sqm=50.0,
            stability_score=0.95,
            predicted_iou=0.88,
            primary_label="vehicle",
            label_confidence=0.9,
            label_source="overlap",
        )

        self.assertEqual(segment.segment_id, 1)
        self.assertEqual(segment.primary_label, "vehicle")
        self.assertEqual(segment.label_confidence, 0.9)
        self.assertEqual(segment.label_source, "overlap")

    def test_labeled_segment_to_dict(self):
        """Test converting labeled segment to dictionary."""
        segment = LabeledSAMSegment(
            segment_id=1,
            pixel_mask=np.zeros((10, 10)),
            pixel_bbox=(10, 20, 30, 40),
            geo_polygon=[(-122.0, 37.0)],
            area_pixels=100,
            area_sqm=10.0,
            stability_score=0.9,
            predicted_iou=0.8,
            primary_label="pool",
            label_confidence=0.85,
            label_source="overlap",
        )

        result = segment.to_dict()

        self.assertIn("segment_id", result)
        self.assertIn("primary_label", result)
        self.assertIn("label_confidence", result)
        self.assertNotIn("pixel_mask", result)  # Should exclude pixel mask
        self.assertEqual(result["primary_label"], "pool")

    def test_labeled_segment_to_geojson(self):
        """Test converting labeled segment to GeoJSON."""
        segment = LabeledSAMSegment(
            segment_id=2,
            pixel_mask=np.zeros((10, 10)),
            pixel_bbox=(10, 20, 30, 40),
            geo_polygon=[
                (-122.4194, 37.7749),
                (-122.4193, 37.7749),
                (-122.4193, 37.7748),
                (-122.4194, 37.7748),
                (-122.4194, 37.7749),
            ],
            area_pixels=200,
            area_sqm=25.0,
            stability_score=0.92,
            predicted_iou=0.85,
            primary_label="building",
            label_confidence=0.95,
            label_source="osm",
            labeling_reason="OSM building overlap",
        )

        feature = segment.to_geojson_feature()

        self.assertEqual(feature["type"], "Feature")
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertIn("properties", feature)
        self.assertEqual(feature["properties"]["primary_label"], "building")
        self.assertEqual(feature["properties"]["label_source"], "osm")
        self.assertIn("color", feature["properties"])


class TestSAMSegmentLabeler(unittest.TestCase):
    """Test SAM segment labeling service."""

    def test_labeler_init_defaults(self):
        """Test labeler initialization with defaults."""
        labeler = SAMSegmentLabeler()

        self.assertEqual(labeler.overlap_threshold, 0.3)
        self.assertEqual(labeler.containment_threshold, 0.7)
        self.assertEqual(labeler.osm_overlap_threshold, 0.5)
        self.assertTrue(labeler.use_osm)

    def test_labeler_init_custom_params(self):
        """Test labeler initialization with custom parameters."""
        labeler = SAMSegmentLabeler(
            overlap_threshold=0.5,
            containment_threshold=0.8,
            osm_overlap_threshold=0.6,
            use_osm=False,
        )

        self.assertEqual(labeler.overlap_threshold, 0.5)
        self.assertEqual(labeler.containment_threshold, 0.8)
        self.assertEqual(labeler.osm_overlap_threshold, 0.6)
        self.assertFalse(labeler.use_osm)

    def test_last_osm_buildings_empty(self):
        """Test last_osm_buildings when no OSM data."""
        labeler = SAMSegmentLabeler()
        self.assertEqual(labeler.last_osm_buildings, [])

    def test_last_osm_roads_empty(self):
        """Test last_osm_roads when no OSM data."""
        labeler = SAMSegmentLabeler()
        self.assertEqual(labeler.last_osm_roads, [])

    def test_last_osm_buildings_with_data(self):
        """Test last_osm_buildings with cached data."""
        labeler = SAMSegmentLabeler()
        labeler._last_osm_data = {"buildings": [Mock(), Mock()], "roads": []}

        buildings = labeler.last_osm_buildings
        self.assertEqual(len(buildings), 2)

    def test_last_osm_roads_with_data(self):
        """Test last_osm_roads with cached data."""
        labeler = SAMSegmentLabeler()
        labeler._last_osm_data = {"buildings": [], "roads": [Mock()]}

        roads = labeler.last_osm_roads
        self.assertEqual(len(roads), 1)

    def test_calculate_iou_identical_polygons(self):
        """Test IoU calculation with identical polygons."""
        labeler = SAMSegmentLabeler()
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        iou = labeler._calculate_iou(poly, poly)
        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        labeler = SAMSegmentLabeler()
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

        iou = labeler._calculate_iou(poly1, poly2)
        self.assertEqual(iou, 0.0)

    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        labeler = SAMSegmentLabeler()
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

        iou = labeler._calculate_iou(poly1, poly2)
        # Overlap is 1x1 = 1, Union is 4 + 4 - 1 = 7, IoU = 1/7
        expected_iou = 1.0 / 7.0
        self.assertAlmostEqual(iou, expected_iou, places=5)

    def test_detection_to_polygon(self):
        """Test converting detection to polygon."""
        labeler = SAMSegmentLabeler()

        mock_detection = Mock()
        mock_detection.geo_polygon = [
            (-122.0, 37.0),
            (-121.9, 37.0),
            (-121.9, 36.9),
            (-122.0, 36.9),
            (-122.0, 37.0),
        ]

        polygon = labeler._detection_to_polygon(mock_detection)

        self.assertIsInstance(polygon, Polygon)
        self.assertTrue(polygon.is_valid)

    def test_calculate_polygon_area_sqm(self):
        """Test calculating polygon area in square meters."""
        labeler = SAMSegmentLabeler()

        # Small square approximately 10m x 10m (rough estimate)
        polygon = Polygon(
            [
                (-122.0000, 37.0000),
                (-122.0001, 37.0000),
                (-122.0001, 37.0001),
                (-122.0000, 37.0001),
                (-122.0000, 37.0000),
            ]
        )

        area = labeler._calculate_polygon_area_sqm(polygon)

        # Area should be positive
        self.assertGreater(area, 0)
        # Should be roughly 100-150 sqm (10m x 10m)
        self.assertLess(area, 1000)  # Sanity check

    @patch("parcel_ai_json.sam_labeler.SAMSegmentLabeler._label_by_overlap")
    def test_label_segments_empty(self, mock_label_overlap):
        """Test labeling with no segments."""
        labeler = SAMSegmentLabeler(use_osm=False)

        result = labeler.label_segments(
            sam_segments=[], detections={"vehicles": [], "pools": []}
        )

        self.assertEqual(result, [])
        mock_label_overlap.assert_not_called()

    @patch("parcel_ai_json.sam_labeler.SAMSegmentLabeler._label_by_overlap")
    def test_label_segments_with_overlap_match(self, mock_label_overlap):
        """Test labeling segments with overlap match."""
        labeler = SAMSegmentLabeler(use_osm=False)

        # Mock SAM segment
        mock_segment = Mock()
        mock_segment.segment_id = 1
        mock_segment.pixel_mask = np.zeros((10, 10))
        mock_segment.pixel_bbox = (10, 20, 30, 40)
        mock_segment.geo_polygon = [(-122.0, 37.0), (-121.9, 37.0)]
        mock_segment.area_pixels = 100
        mock_segment.area_sqm = 10.0
        mock_segment.stability_score = 0.9
        mock_segment.predicted_iou = 0.85

        # Mock overlap result (uses internal format)
        mock_label_overlap.return_value = {
            "label": "vehicle",
            "confidence": 0.9,
            "source": "overlap",
            "related_ids": ["vehicle_0"],
            "reason": "Overlaps with vehicle detection",
        }

        result = labeler.label_segments(
            sam_segments=[mock_segment], detections={"vehicles": [Mock()], "pools": []}
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], LabeledSAMSegment)
        self.assertEqual(result[0].primary_label, "vehicle")
        self.assertEqual(result[0].label_confidence, 0.9)

    def test_label_by_overlap_vehicle_match(self):
        """Test overlap-based labeling with vehicle match."""
        labeler = SAMSegmentLabeler(overlap_threshold=0.3)

        # Create segment polygon
        segment_poly = Polygon(
            [
                (-122.0000, 37.0000),
                (-122.0001, 37.0000),
                (-122.0001, 37.0001),
                (-122.0000, 37.0001),
            ]
        )

        mock_segment = Mock()
        mock_segment.geo_polygon = list(segment_poly.exterior.coords)

        # Create overlapping vehicle detection
        mock_vehicle = Mock()
        mock_vehicle.geo_polygon = list(segment_poly.exterior.coords)

        detections = {"vehicles": [mock_vehicle], "pools": [], "amenities": []}

        result = labeler._label_by_overlap(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "vehicle")
        self.assertGreater(result["confidence"], 0.3)

    def test_label_by_overlap_no_match(self):
        """Test overlap-based labeling with no match."""
        labeler = SAMSegmentLabeler(overlap_threshold=0.5)

        # Non-overlapping polygons
        segment_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        vehicle_poly = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])

        mock_segment = Mock()
        mock_segment.geo_polygon = list(segment_poly.exterior.coords)

        mock_vehicle = Mock()
        mock_vehicle.geo_polygon = list(vehicle_poly.exterior.coords)

        detections = {"vehicles": [mock_vehicle], "pools": [], "amenities": []}

        result = labeler._label_by_overlap(mock_segment, detections)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
