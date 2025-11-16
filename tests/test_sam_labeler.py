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

    def test_label_by_overlap_pool_match(self):
        """Test overlap-based labeling with pool match."""
        labeler = SAMSegmentLabeler(overlap_threshold=0.3)

        # Create overlapping segment and pool
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

        # Create overlapping pool
        mock_pool = Mock()
        mock_pool.geo_polygon = list(segment_poly.exterior.coords)

        detections = {"vehicles": [], "pools": [mock_pool], "amenities": []}

        result = labeler._label_by_overlap(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "pool")
        self.assertGreater(result["confidence"], 0.3)

    def test_label_by_overlap_amenity_match(self):
        """Test overlap-based labeling with amenity match."""
        from parcel_ai_json.amenity_detector import AmenityDetection

        labeler = SAMSegmentLabeler(overlap_threshold=0.3)

        # Create overlapping segment and amenity
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

        # Create overlapping amenity
        amenity = AmenityDetection(
            amenity_type="tennis_court",
            pixel_bbox=(10, 20, 30, 40),
            geo_polygon=list(segment_poly.exterior.coords),
            confidence=0.9,
        )

        detections = {"vehicles": [], "pools": [], "amenities": [amenity]}

        result = labeler._label_by_overlap(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "amenity")
        self.assertEqual(result["subtype"], "tennis_court")
        self.assertGreater(result["confidence"], 0.3)

    def test_label_by_overlap_tree_polygon_match(self):
        """Test overlap-based labeling with tree polygon match."""
        labeler = SAMSegmentLabeler(overlap_threshold=0.3)

        # Create overlapping segment and tree polygon
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

        # Create overlapping tree polygon
        mock_tree_poly = Mock()
        mock_tree_poly.geo_polygon = list(segment_poly.exterior.coords)

        detections = {
            "vehicles": [],
            "pools": [],
            "amenities": [],
            "tree_polygons": [mock_tree_poly],
        }

        result = labeler._label_by_overlap(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "tree")
        self.assertGreater(result["confidence"], 0.3)

    def test_label_by_containment_driveway(self):
        """Test containment-based labeling identifies driveway."""
        labeler = SAMSegmentLabeler(containment_threshold=0.7)

        # Large segment containing a vehicle
        segment_poly = Polygon(
            [
                (-122.0000, 37.0000),
                (-122.0010, 37.0000),
                (-122.0010, 37.0010),
                (-122.0000, 37.0010),
            ]
        )

        mock_segment = Mock()
        mock_segment.geo_polygon = list(segment_poly.exterior.coords)
        mock_segment.area_sqm = 1000.0  # ~1000 sqm segment

        # Small vehicle inside segment
        vehicle_poly = Polygon(
            [
                (-122.0002, 37.0002),
                (-122.0003, 37.0002),
                (-122.0003, 37.0003),
                (-122.0002, 37.0003),
            ]
        )

        mock_vehicle = Mock()
        mock_vehicle.geo_polygon = list(vehicle_poly.exterior.coords)

        detections = {"vehicles": [mock_vehicle], "pools": [], "tree_polygons": []}

        result = labeler._label_by_containment(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "driveway")
        self.assertEqual(result["source"], "containment")

    def test_label_by_containment_tree_canopy(self):
        """Test containment-based labeling identifies tree canopy."""
        labeler = SAMSegmentLabeler()

        # Small segment inside tree coverage
        segment_poly = Polygon(
            [
                (-122.0002, 37.0002),
                (-122.0003, 37.0002),
                (-122.0003, 37.0003),
                (-122.0002, 37.0003),
            ]
        )

        mock_segment = Mock()
        mock_segment.geo_polygon = list(segment_poly.exterior.coords)
        mock_segment.area_sqm = 10.0

        # Large tree polygon containing segment
        tree_poly = Polygon(
            [
                (-122.0000, 37.0000),
                (-122.0010, 37.0000),
                (-122.0010, 37.0010),
                (-122.0000, 37.0010),
            ]
        )

        mock_tree_poly = Mock()
        mock_tree_poly.geo_polygon = list(tree_poly.exterior.coords)

        detections = {"vehicles": [], "pools": [], "tree_polygons": [mock_tree_poly]}

        result = labeler._label_by_containment(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "tree_canopy")
        self.assertEqual(result["source"], "containment")

    def test_label_by_containment_pool_deck(self):
        """Test containment-based labeling identifies pool deck."""
        labeler = SAMSegmentLabeler()

        # Large segment containing pool
        segment_poly = Polygon(
            [
                (-122.0000, 37.0000),
                (-122.0010, 37.0000),
                (-122.0010, 37.0010),
                (-122.0000, 37.0010),
            ]
        )

        mock_segment = Mock()
        mock_segment.geo_polygon = list(segment_poly.exterior.coords)
        mock_segment.area_sqm = 150.0

        # Pool inside segment
        pool_poly = Polygon(
            [
                (-122.0002, 37.0002),
                (-122.0003, 37.0002),
                (-122.0003, 37.0003),
                (-122.0002, 37.0003),
            ]
        )

        mock_pool = Mock()
        mock_pool.geo_polygon = list(pool_poly.exterior.coords)

        detections = {"vehicles": [], "pools": [mock_pool], "tree_polygons": []}

        result = labeler._label_by_containment(mock_segment, detections)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "pavement")
        self.assertEqual(result["subtype"], "pool_deck")

    def test_label_by_containment_no_match(self):
        """Test containment-based labeling with no match."""
        labeler = SAMSegmentLabeler()

        segment_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        mock_segment = Mock()
        mock_segment.geo_polygon = list(segment_poly.exterior.coords)
        mock_segment.area_sqm = 100.0

        detections = {"vehicles": [], "pools": [], "tree_polygons": []}

        result = labeler._label_by_containment(mock_segment, detections)

        self.assertIsNone(result)

    def test_calculate_iou_invalid_polygons(self):
        """Test IoU calculation with invalid polygons."""
        labeler = SAMSegmentLabeler()

        # Create invalid polygon (self-intersecting)
        invalid_poly = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        iou = labeler._calculate_iou(invalid_poly, valid_poly)

        # Should return 0 for invalid polygons
        self.assertEqual(iou, 0.0)

    def test_calculate_iou_exception_handling(self):
        """Test IoU calculation handles exceptions."""
        labeler = SAMSegmentLabeler()

        # Create polygons that might cause issues
        poly1 = Polygon()
        poly2 = Polygon([(0, 0), (1, 0), (1, 1)])

        iou = labeler._calculate_iou(poly1, poly2)

        # Should return 0 on exception
        self.assertEqual(iou, 0.0)

    def test_detection_to_polygon_exception_handling(self):
        """Test detection to polygon conversion handles exceptions."""
        labeler = SAMSegmentLabeler()

        # Create mock detection with invalid geo_polygon
        mock_detection = Mock()
        mock_detection.geo_polygon = "invalid"

        polygon = labeler._detection_to_polygon(mock_detection)

        # Should return empty polygon on exception
        self.assertTrue(polygon.is_empty)

    def test_calculate_polygon_area_sqm_invalid_polygon(self):
        """Test area calculation with invalid polygon."""
        labeler = SAMSegmentLabeler()

        # Invalid polygon
        invalid_poly = Polygon()

        area = labeler._calculate_polygon_area_sqm(invalid_poly)

        self.assertEqual(area, 0.0)

    @patch("parcel_ai_json.osm_data_fetcher.OSMDataFetcher")
    def test_fetch_osm_data_success(self, mock_osm_fetcher_class):
        """Test OSM data fetching succeeds."""
        labeler = SAMSegmentLabeler(use_osm=True)

        # Mock OSM fetcher
        mock_fetcher = Mock()
        mock_fetcher.fetch_buildings_and_roads.return_value = {
            "buildings": [],
            "roads": [],
        }
        mock_osm_fetcher_class.return_value = mock_fetcher

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        osm_data = labeler._fetch_osm_data(satellite_image)

        self.assertIsNotNone(osm_data)
        self.assertIn("buildings", osm_data)
        self.assertIn("roads", osm_data)

    @patch("parcel_ai_json.osm_data_fetcher.OSMDataFetcher")
    def test_fetch_osm_data_exception_handling(self, mock_osm_fetcher_class):
        """Test OSM data fetching handles exceptions."""
        labeler = SAMSegmentLabeler(use_osm=True)

        # Mock OSM fetcher to raise exception
        mock_osm_fetcher_class.side_effect = Exception("OSM fetch failed")

        satellite_image = {
            "center_lat": 37.7749,
            "center_lon": -122.4194,
            "zoom_level": 20,
        }

        osm_data = labeler._fetch_osm_data(satellite_image)

        # Should return None on exception
        self.assertIsNone(osm_data)

    def test_label_segments_with_osm_data(self):
        """Test label_segments with OSM data enabled."""
        from parcel_ai_json.osm_data_fetcher import OSMBuilding

        labeler = SAMSegmentLabeler(use_osm=True)

        # Create mock segment
        mock_segment = Mock()
        mock_segment.segment_id = 1
        mock_segment.pixel_mask = np.zeros((10, 10))
        mock_segment.pixel_bbox = (10, 20, 30, 40)
        mock_segment.geo_polygon = [(-122.0, 37.0), (-121.9, 37.0), (-121.9, 36.9)]
        mock_segment.area_pixels = 100
        mock_segment.area_sqm = 10.0
        mock_segment.stability_score = 0.9
        mock_segment.predicted_iou = 0.85

        # Create OSM building overlapping with segment
        osm_building = OSMBuilding(
            osm_id=123,
            geo_polygon=[(-122.0, 37.0), (-121.9, 37.0), (-121.9, 36.9)],
            building_type="house",
        )

        # Mock OSM data fetch
        with patch.object(labeler, "_fetch_osm_data") as mock_fetch:
            mock_fetch.return_value = {"buildings": [osm_building], "roads": []}

            satellite_image = {
                "center_lat": 37.0,
                "center_lon": -122.0,
                "zoom_level": 20,
            }

            result = labeler.label_segments(
                sam_segments=[mock_segment],
                detections={"vehicles": [], "pools": [], "amenities": []},
                satellite_image=satellite_image,
            )

            self.assertEqual(len(result), 1)
            # Should be labeled as building from OSM
            self.assertEqual(result[0].primary_label, "building")
            self.assertEqual(result[0].label_source, "osm")

    def test_label_by_osm_building_match(self):
        """Test OSM-based labeling with building match."""
        from parcel_ai_json.osm_data_fetcher import OSMBuilding

        labeler = SAMSegmentLabeler(osm_overlap_threshold=0.5)

        # Create segment
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

        # Create OSM building overlapping
        osm_building = OSMBuilding(
            osm_id=456,
            geo_polygon=list(segment_poly.exterior.coords),
            building_type="residential",
        )

        osm_data = {"buildings": [osm_building], "roads": []}

        result = labeler._label_by_osm(mock_segment, osm_data)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "building")
        self.assertEqual(result["source"], "osm")
        self.assertEqual(result["subtype"], "residential")

    def test_label_by_osm_road_driveway(self):
        """Test OSM-based labeling identifies driveway from OSM road."""
        from parcel_ai_json.osm_data_fetcher import OSMRoad

        labeler = SAMSegmentLabeler(osm_overlap_threshold=0.3)

        # Create segment near road
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

        # Create OSM driveway passing through segment
        osm_road = OSMRoad(
            osm_id=789,
            geo_linestring=[(-122.00005, 37.0000), (-122.00005, 37.0001)],
            highway_type="driveway",
            width_m=4.0,
        )

        osm_data = {"buildings": [], "roads": [osm_road]}

        result = labeler._label_by_osm(mock_segment, osm_data)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "driveway")
        self.assertEqual(result["source"], "osm")
        self.assertEqual(result["subtype"], "driveway")

    def test_label_by_osm_road_no_match(self):
        """Test OSM-based labeling with road that doesn't overlap."""
        from parcel_ai_json.osm_data_fetcher import OSMRoad

        labeler = SAMSegmentLabeler(osm_overlap_threshold=0.5)

        # Create segment far from road
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

        # Create OSM road far away from segment
        osm_road = OSMRoad(
            osm_id=999,
            geo_linestring=[(-122.1000, 37.1000), (-122.1000, 37.1001)],
            highway_type="residential",
            width_m=8.0,
        )

        osm_data = {"buildings": [], "roads": [osm_road]}

        result = labeler._label_by_osm(mock_segment, osm_data)

        # Should return None when no match found
        self.assertIsNone(result)

    def test_label_segments_unknown_fallback(self):
        """Test that segments with no match are labeled as unknown."""
        labeler = SAMSegmentLabeler(use_osm=False)

        # Create segment that won't match anything
        mock_segment = Mock()
        mock_segment.segment_id = 1
        mock_segment.pixel_mask = np.zeros((10, 10))
        mock_segment.pixel_bbox = (10, 20, 30, 40)
        mock_segment.geo_polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        mock_segment.area_pixels = 100
        mock_segment.area_sqm = 10.0
        mock_segment.stability_score = 0.9
        mock_segment.predicted_iou = 0.85

        # Empty detections
        detections = {"vehicles": [], "pools": [], "amenities": []}

        result = labeler.label_segments(
            sam_segments=[mock_segment], detections=detections
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].primary_label, "unknown")
        self.assertEqual(result[0].label_source, "none")


if __name__ == "__main__":
    unittest.main()
