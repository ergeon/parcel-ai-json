#!/usr/bin/env python3
"""Tests for OSM data fetcher."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
