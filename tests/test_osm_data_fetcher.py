#!/usr/bin/env python3
"""Tests for OSM data fetcher."""

import unittest
from unittest.mock import Mock, patch
import requests

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
            name="Test House",
            height=10.0,
            levels=2,
        )

        self.assertEqual(building.osm_id, 123456)
        self.assertEqual(building.building_type, "house")
        self.assertEqual(building.name, "Test House")
        self.assertEqual(building.height, 10.0)
        self.assertEqual(building.levels, 2)
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
        self.assertEqual(
            feature["geometry"]["coordinates"],
            [geo_polygon]
        )
        self.assertEqual(feature["properties"]["osm_id"], 123456)
        self.assertEqual(feature["properties"]["building_type"], "house")
        self.assertEqual(feature["properties"]["feature_type"], "osm_building")


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
            name="Main Street",
        )

        self.assertEqual(road.osm_id, 789012)
        self.assertEqual(road.highway_type, "residential")
        self.assertEqual(road.name, "Main Street")
        self.assertEqual(len(road.geo_linestring), 3)

    def test_osm_road_to_geojson_feature(self):
        """Test converting OSMRoad to GeoJSON feature."""
        geo_linestring = [
            (-122.4194, 37.7749),
            (-122.4193, 37.7749),
        ]

        road = OSMRoad(
            osm_id=789012,
            geo_linestring=geo_linestring,
            highway_type="residential",
        )

        feature = road.to_geojson_feature()

        self.assertEqual(feature["type"], "Feature")
        self.assertEqual(feature["geometry"]["type"], "LineString")
        self.assertEqual(
            feature["geometry"]["coordinates"],
            geo_linestring
        )
        self.assertEqual(feature["properties"]["osm_id"], 789012)
        self.assertEqual(feature["properties"]["highway_type"], "residential")


class TestOSMDataFetcher(unittest.TestCase):
    """Test OSMDataFetcher."""

    def test_initialization_defaults(self):
        """Test fetcher initialization with defaults."""
        fetcher = OSMDataFetcher()

        self.assertEqual(fetcher.max_retries, 3)
        self.assertEqual(fetcher.timeout, 30)
        self.assertEqual(fetcher.initial_backoff, 2.0)

    def test_initialization_custom_params(self):
        """Test fetcher initialization with custom parameters."""
        fetcher = OSMDataFetcher(
            max_retries=5,
            timeout=60,
            initial_backoff=5.0,
        )

        self.assertEqual(fetcher.max_retries, 5)
        self.assertEqual(fetcher.timeout, 60)
        self.assertEqual(fetcher.initial_backoff, 5.0)

    @patch('requests.post')
    def test_fetch_buildings_success(self, mock_post):
        """Test successful building fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "elements": [
                {
                    "type": "way",
                    "id": 123456,
                    "tags": {"building": "house", "name": "Test House"},
                    "geometry": [
                        {"lon": -122.4194, "lat": 37.7749},
                        {"lon": -122.4193, "lat": 37.7749},
                        {"lon": -122.4193, "lat": 37.7748},
                        {"lon": -122.4194, "lat": 37.7748},
                        {"lon": -122.4194, "lat": 37.7749},
                    ]
                }
            ]
        }
        mock_post.return_value = mock_response

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
            zoom_level=20,
        )

        self.assertIn("buildings", result)
        self.assertIn("roads", result)
        self.assertEqual(len(result["buildings"]), 1)

        building = result["buildings"][0]
        self.assertEqual(building.osm_id, 123456)
        self.assertEqual(building.building_type, "house")
        self.assertEqual(building.name, "Test House")
        self.assertEqual(len(building.geo_polygon), 5)

    @patch('requests.post')
    def test_fetch_roads_success(self, mock_post):
        """Test successful road fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "elements": [
                {
                    "type": "way",
                    "id": 789012,
                    "tags": {"highway": "residential", "name": "Main St"},
                    "geometry": [
                        {"lon": -122.4194, "lat": 37.7749},
                        {"lon": -122.4193, "lat": 37.7749},
                    ]
                }
            ]
        }
        mock_post.return_value = mock_response

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
        )

        self.assertEqual(len(result["roads"]), 1)

        road = result["roads"][0]
        self.assertEqual(road.osm_id, 789012)
        self.assertEqual(road.highway_type, "residential")
        self.assertEqual(road.name, "Main St")

    @patch('requests.post')
    def test_retry_on_timeout(self, mock_post):
        """Test retry logic on timeout."""
        # First call times out, second succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout("Connection timeout"),
            Mock(json=lambda: {"elements": []}),
        ]

        fetcher = OSMDataFetcher(max_retries=2)
        result = fetcher.fetch_buildings_and_roads(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
        )

        # Should retry once and succeed
        self.assertEqual(mock_post.call_count, 2)
        self.assertIn("buildings", result)

    @patch('requests.post')
    def test_no_retry_on_4xx_error(self, mock_post):
        """Test no retry on client errors."""
        response = Mock()
        response.status_code = 400
        error = requests.exceptions.HTTPError(response=response)
        error.response = response
        mock_post.return_value.raise_for_status.side_effect = error

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
        )

        # Should not retry on 4xx errors
        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(result["buildings"], [])

    @patch('requests.post')
    def test_retry_on_5xx_error(self, mock_post):
        """Test retry on server errors."""
        response = Mock()
        response.status_code = 503
        error = requests.exceptions.HTTPError(response=response)
        error.response = response

        # First call fails with 503, second succeeds
        mock_post.side_effect = [
            Mock(raise_for_status=Mock(side_effect=error)),
            Mock(
                raise_for_status=Mock(),
                json=lambda: {"elements": []}
            ),
        ]

        fetcher = OSMDataFetcher(max_retries=2)
        result = fetcher.fetch_buildings_and_roads(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
        )

        # Should retry on 5xx errors
        self.assertEqual(mock_post.call_count, 2)

    @patch('requests.post')
    def test_parse_building_with_all_tags(self, mock_post):
        """Test parsing building with all optional tags."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "elements": [
                {
                    "type": "way",
                    "id": 123,
                    "tags": {
                        "building": "commercial",
                        "name": "Shopping Mall",
                        "height": "15.5",
                        "building:levels": "3",
                    },
                    "geometry": [
                        {"lon": -122.4194, "lat": 37.7749},
                        {"lon": -122.4193, "lat": 37.7749},
                        {"lon": -122.4193, "lat": 37.7748},
                        {"lon": -122.4194, "lat": 37.7749},
                    ]
                }
            ]
        }
        mock_post.return_value = mock_response

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            37.7749, -122.4194, 640, 640
        )

        building = result["buildings"][0]
        self.assertEqual(building.building_type, "commercial")
        self.assertEqual(building.name, "Shopping Mall")
        self.assertEqual(building.height, 15.5)
        self.assertEqual(building.levels, 3)

    @patch('requests.post')
    def test_skip_building_without_geometry(self, mock_post):
        """Test skipping buildings without geometry."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "elements": [
                {
                    "type": "way",
                    "id": 123,
                    "tags": {"building": "yes"},
                    # No geometry field
                }
            ]
        }
        mock_post.return_value = mock_response

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            37.7749, -122.4194, 640, 640
        )

        # Should skip building without geometry
        self.assertEqual(len(result["buildings"]), 0)

    @patch('requests.post')
    def test_skip_building_with_too_few_points(self, mock_post):
        """Test skipping buildings with < 3 points."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "elements": [
                {
                    "type": "way",
                    "id": 123,
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lon": -122.4194, "lat": 37.7749},
                        {"lon": -122.4193, "lat": 37.7749},
                    ]  # Only 2 points
                }
            ]
        }
        mock_post.return_value = mock_response

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            37.7749, -122.4194, 640, 640
        )

        # Should skip building with < 3 points
        self.assertEqual(len(result["buildings"]), 0)

    @patch('requests.post')
    def test_calculate_bounding_box(self, mock_post):
        """Test bounding box calculation."""
        mock_post.return_value = Mock(json=lambda: {"elements": []})

        fetcher = OSMDataFetcher()
        fetcher.fetch_buildings_and_roads(
            center_lat=37.7749,
            center_lon=-122.4194,
            image_width_px=640,
            image_height_px=640,
            zoom_level=20,
        )

        # Check that post was called with correct query
        self.assertTrue(mock_post.called)
        call_kwargs = mock_post.call_args[1]
        query = call_kwargs['data']

        # Should contain bounding box coordinates
        self.assertIn("bbox", query)

    @patch('requests.post')
    def test_empty_response(self, mock_post):
        """Test handling empty OSM response."""
        mock_response = Mock()
        mock_response.json.return_value = {"elements": []}
        mock_post.return_value = mock_response

        fetcher = OSMDataFetcher()
        result = fetcher.fetch_buildings_and_roads(
            37.7749, -122.4194, 640, 640
        )

        self.assertEqual(len(result["buildings"]), 0)
        self.assertEqual(len(result["roads"]), 0)


if __name__ == "__main__":
    unittest.main()
