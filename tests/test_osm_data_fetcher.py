#!/usr/bin/env python3
"""Tests for OSM data fetcher."""

import unittest
from unittest.mock import Mock, patch
from shapely.geometry import Polygon

from parcel_ai_json.osm_data_fetcher import (
    OSMDataFetcher,
    OSMBuilding,
    OSMRoad,
)


class TestOSMBuilding(unittest.TestCase):
    """Test OSMBuilding dataclass."""

    def test_osm_building_creation(self):
        """Test creating an OSMBuilding."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]

        building = OSMBuilding(
            osm_id=123456,
            geo_polygon=geo_polygon,
            building_type="house",
            area_sqm=100.5,
            tags={"name": "Test House", "height": "10.0"},
        )

        self.assertEqual(building.osm_id, 123456)
        self.assertEqual(building.building_type, "house")
        self.assertEqual(building.area_sqm, 100.5)
        self.assertEqual(building.tags["name"], "Test House")
        self.assertEqual(len(building.geo_polygon), 5)

    def test_osm_building_to_geojson_feature(self):
        """Test converting OSMBuilding to GeoJSON feature."""
        geo_polygon = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4193, 37.7748),
            (-122.4194, 37.7748),
            (-122.4194, 37.7749),
        ]

        building = OSMBuilding(
            osm_id=123456,
            geo_polygon=geo_polygon,
            building_type="house",
        )

        feature = building.to_geojson_feature()

        self.assertEqual(feature["type"], "Feature")
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertEqual(feature["geometry"]["coordinates"], [geo_polygon])
        self.assertEqual(feature["properties"]["osm_id"], 123456)
        self.assertEqual(feature["properties"]["building_type"], "house")
        self.assertEqual(feature["properties"]["feature_type"], "osm_building")

    def test_osm_building_to_shapely(self):
        """Test converting OSMBuilding to Shapely polygon."""
        geo_polygon = [
            (-122.0, 37.0),
            (-121.9, 37.0),
            (-121.9, 36.9),
            (-122.0, 36.9),
            (-122.0, 37.0),
        ]

        building = OSMBuilding(osm_id=123, geo_polygon=geo_polygon)
        polygon = building.to_shapely()

        self.assertIsInstance(polygon, Polygon)
        self.assertTrue(polygon.is_valid)


class TestOSMRoad(unittest.TestCase):
    """Test OSMRoad dataclass."""

    def test_osm_road_creation(self):
        """Test creating an OSMRoad."""
        geo_linestring = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
            (-122.4192, 37.7748),
        ]

        road = OSMRoad(
            osm_id=789012,
            geo_linestring=geo_linestring,
            highway_type="residential",
            width_m=6.0,
            tags={"name": "Main Street"},
        )

        self.assertEqual(road.osm_id, 789012)
        self.assertEqual(road.highway_type, "residential")
        self.assertEqual(road.width_m, 6.0)
        self.assertEqual(road.tags["name"], "Main Street")
        self.assertEqual(len(road.geo_linestring), 3)


class TestOSMDataFetcher(unittest.TestCase):
    """Test OSMDataFetcher."""

    def test_initialization_defaults(self):
        """Test fetcher initialization with defaults."""
        fetcher = OSMDataFetcher()

        self.assertEqual(fetcher.max_retries, 5)
        self.assertEqual(fetcher.timeout, 25)
        self.assertEqual(fetcher.buffer_meters, 50.0)
        self.assertEqual(fetcher.initial_backoff, 2.0)

    def test_initialization_custom_params(self):
        """Test fetcher initialization with custom parameters."""
        fetcher = OSMDataFetcher(
            max_retries=3,
            timeout=60,
            buffer_meters=100.0,
            initial_backoff=5.0,
        )

        self.assertEqual(fetcher.max_retries, 3)
        self.assertEqual(fetcher.timeout, 60)
        self.assertEqual(fetcher.buffer_meters, 100.0)
        self.assertEqual(fetcher.initial_backoff, 5.0)

    def test_calculate_bbox(self):
        """Test bounding box calculation."""
        fetcher = OSMDataFetcher(buffer_meters=100.0)

        bbox = fetcher._calculate_bbox(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=512,
            image_height_px=512,
            zoom_level=20,
        )

        self.assertEqual(len(bbox), 4)
        south_lat, west_lon, north_lat, east_lon = bbox

        # Verify it's a valid bbox
        self.assertLess(west_lon, east_lon)
        self.assertLess(south_lat, north_lat)
        # Verify center is roughly in the middle
        self.assertAlmostEqual((west_lon + east_lon) / 2, -122.4194, delta=0.01)
        self.assertAlmostEqual((south_lat + north_lat) / 2, 37.7749, delta=0.01)

    def test_calculate_polygon_area(self):
        """Test polygon area calculation."""
        fetcher = OSMDataFetcher()

        # Square approximately 10m x 10m
        geo_polygon = [
            (-122.0000, 37.0000),
            (-122.0001, 37.0000),
            (-122.0001, 37.0001),
            (-122.0000, 37.0001),
            (-122.0000, 37.0000),
        ]

        area = fetcher._calculate_polygon_area(geo_polygon)

        # Should be positive
        self.assertGreater(area, 0)
        # Should be roughly 100 sqm (10m x 10m)
        self.assertLess(area, 1000)  # Sanity check

    def test_parse_building_element_valid(self):
        """Test parsing valid OSM building element."""
        fetcher = OSMDataFetcher()

        element = {
            "id": 123456,
            "type": "way",
            "tags": {"building": "house", "name": "Test House"},
            "geometry": [
                {"lat": 37.7749, "lon": -122.4194},
                {"lat": 37.7750, "lon": -122.4194},
                {"lat": 37.7750, "lon": -122.4193},
                {"lat": 37.7749, "lon": -122.4193},
                {"lat": 37.7749, "lon": -122.4194},
            ],
        }

        building = fetcher._parse_building_element(element)

        self.assertIsNotNone(building)
        self.assertEqual(building.osm_id, 123456)
        self.assertEqual(building.building_type, "house")
        self.assertEqual(len(building.geo_polygon), 5)
        self.assertIn("name", building.tags)

    def test_parse_building_element_invalid_polygon(self):
        """Test parsing building with invalid polygon."""
        fetcher = OSMDataFetcher()

        # Only 2 points - not a valid polygon
        element = {
            "id": 123,
            "type": "way",
            "tags": {"building": "yes"},
            "geometry": [
                {"lat": 37.0, "lon": -122.0},
                {"lat": 37.1, "lon": -122.1},
            ],
        }

        building = fetcher._parse_building_element(element)
        self.assertIsNone(building)

    def test_parse_road_element_valid(self):
        """Test parsing valid OSM road element."""
        fetcher = OSMDataFetcher()

        element = {
            "id": 789012,
            "type": "way",
            "tags": {"highway": "residential", "name": "Main St", "width": "6"},
            "geometry": [
                {"lat": 37.7749, "lon": -122.4194},
                {"lat": 37.7750, "lon": -122.4193},
                {"lat": 37.7751, "lon": -122.4192},
            ],
        }

        road = fetcher._parse_road_element(element)

        self.assertIsNotNone(road)
        self.assertEqual(road.osm_id, 789012)
        self.assertEqual(road.highway_type, "residential")
        self.assertEqual(road.width_m, 6.0)
        self.assertEqual(len(road.geo_linestring), 3)

    def test_parse_road_element_invalid_linestring(self):
        """Test parsing road with invalid linestring."""
        fetcher = OSMDataFetcher()

        # Only 1 point - not a valid linestring
        element = {
            "id": 789,
            "type": "way",
            "tags": {"highway": "residential"},
            "geometry": [{"lat": 37.0, "lon": -122.0}],
        }

        road = fetcher._parse_road_element(element)
        self.assertIsNone(road)

    def test_fetch_buildings_and_roads_returns_dict(self):
        """Test that fetch_buildings_and_roads returns proper structure."""
        fetcher = OSMDataFetcher()

        # Mock both private fetch methods
        with patch.object(fetcher, "_fetch_buildings") as mock_buildings:
            with patch.object(fetcher, "_fetch_roads") as mock_roads:
                mock_buildings.return_value = []
                mock_roads.return_value = []

                result = fetcher.fetch_buildings_and_roads(
                    center_lat=37.0,
                    center_lon=-122.0,
                    image_width_px=512,
                    image_height_px=512,
                    zoom_level=20,
                )

                self.assertIn("buildings", result)
                self.assertIn("roads", result)
                self.assertIsInstance(result["buildings"], list)
                self.assertIsInstance(result["roads"], list)

    @patch("requests.get")
    def test_retry_request_with_failures(self, mock_get):
        """Test retry logic with server errors returns empty dict."""
        fetcher = OSMDataFetcher(max_retries=2, initial_backoff=0.1)

        # Mock failure with server error
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500  # Server error
        mock_response_fail.raise_for_status.side_effect = Exception("Server error")

        mock_get.return_value = mock_response_fail

        # This should fail after retries and return empty dict
        def request_func():
            return mock_get()

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = fetcher._retry_request(request_func, "test_resource")

        # Should return empty dict after all retries fail
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
