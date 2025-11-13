"""OpenStreetMap data fetcher for building footprints and roads.

Fetches OSM data to enhance SAM segment classification using Overpass API.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import requests
from shapely.geometry import Polygon
from pyproj import Geod


@dataclass
class OSMBuilding:
    """OSM building footprint.

    Attributes:
        osm_id: OSM way/relation ID
        geo_polygon: Geographic polygon coordinates [(lon, lat), ...]
        building_type: Building type tag (e.g., 'house', 'garage', 'shed')
        area_sqm: Area in square meters
        tags: All OSM tags
    """

    osm_id: int
    geo_polygon: List[Tuple[float, float]]
    building_type: Optional[str] = None
    area_sqm: Optional[float] = None
    tags: Dict = None

    def to_shapely(self) -> Polygon:
        """Convert to Shapely Polygon."""
        return Polygon(self.geo_polygon)


@dataclass
class OSMRoad:
    """OSM road/highway feature.

    Attributes:
        osm_id: OSM way ID
        geo_linestring: Geographic linestring coordinates [(lon, lat), ...]
        highway_type: Highway type tag (e.g., 'residential', 'service', 'driveway')
        width_m: Width in meters (if available)
        tags: All OSM tags
    """

    osm_id: int
    geo_linestring: List[Tuple[float, float]]
    highway_type: Optional[str] = None
    width_m: Optional[float] = None
    tags: Dict = None


class OSMDataFetcher:
    """Fetches OSM building footprints and roads using Overpass API.

    Uses bounding box queries to fetch relevant features for SAM classification.
    """

    OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

    def __init__(self, timeout: int = 25, buffer_meters: float = 50.0):
        """Initialize OSM data fetcher.

        Args:
            timeout: Overpass API timeout in seconds
            buffer_meters: Buffer distance around image bounds for fetching data
        """
        self.timeout = timeout
        self.buffer_meters = buffer_meters

    def fetch_buildings_and_roads(
        self,
        center_lat: float,
        center_lon: float,
        image_width_px: int,
        image_height_px: int,
        zoom_level: int = 20,
    ) -> Dict:
        """Fetch OSM buildings and roads for satellite image bounds.

        Args:
            center_lat: Image center latitude (WGS84)
            center_lon: Image center longitude (WGS84)
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            zoom_level: Zoom level (default: 20)

        Returns:
            Dict with keys:
                - 'buildings': List of OSMBuilding objects
                - 'roads': List of OSMRoad objects
        """
        # Calculate bounding box from image metadata
        bbox = self._calculate_bbox(
            center_lat, center_lon, image_width_px, image_height_px, zoom_level
        )

        # Fetch buildings
        buildings = self._fetch_buildings(bbox)

        # Fetch roads
        roads = self._fetch_roads(bbox)

        return {"buildings": buildings, "roads": roads}

    def _calculate_bbox(
        self,
        center_lat: float,
        center_lon: float,
        image_width_px: int,
        image_height_px: int,
        zoom_level: int,
    ) -> Tuple[float, float, float, float]:
        """Calculate bounding box (south, west, north, east) for image.

        Uses pyproj for accurate geodesic calculations.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            zoom_level: Zoom level

        Returns:
            Tuple of (south, west, north, east) in WGS84 degrees
        """
        from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

        converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            zoom_level=zoom_level,
        )

        # Get corners
        top_left = converter.pixel_to_geo(0, 0)
        bottom_right = converter.pixel_to_geo(image_width_px, image_height_px)

        # Extract bounds
        west, north = top_left
        east, south = bottom_right

        # Add buffer using geodesic calculations
        geod = Geod(ellps="WGS84")

        # Buffer north
        north_lon, north_lat, _ = geod.fwd(
            center_lon, north, 0, self.buffer_meters  # bearing 0° = north
        )

        # Buffer south
        south_lon, south_lat, _ = geod.fwd(
            center_lon, south, 180, self.buffer_meters  # bearing 180° = south
        )

        # Buffer east
        east_lon, east_lat, _ = geod.fwd(
            east, center_lat, 90, self.buffer_meters  # bearing 90° = east
        )

        # Buffer west
        west_lon, west_lat, _ = geod.fwd(
            west, center_lat, 270, self.buffer_meters  # bearing 270° = west
        )

        return (south_lat, west_lon, north_lat, east_lon)

    def _fetch_buildings(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[OSMBuilding]:
        """Fetch building footprints from OSM.

        Args:
            bbox: Bounding box (south, west, north, east)

        Returns:
            List of OSMBuilding objects
        """
        south, west, north, east = bbox

        # Overpass QL query for buildings
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["building"]({south},{west},{north},{east});
          relation["building"]({south},{west},{north},{east});
        );
        out geom;
        """

        try:
            response = requests.post(
                self.OVERPASS_ENDPOINT, data={"data": query}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Warning: OSM building fetch failed: {e}")
            return []

        buildings = []
        for element in data.get("elements", []):
            building = self._parse_building_element(element)
            if building:
                buildings.append(building)

        print(f"Fetched {len(buildings)} OSM buildings")
        return buildings

    def _fetch_roads(self, bbox: Tuple[float, float, float, float]) -> List[OSMRoad]:
        """Fetch roads and driveways from OSM.

        Args:
            bbox: Bounding box (south, west, north, east)

        Returns:
            List of OSMRoad objects
        """
        south, west, north, east = bbox

        # Overpass QL query for roads (including driveways, service roads)
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["highway"]({south},{west},{north},{east});
        );
        out geom;
        """

        try:
            response = requests.post(
                self.OVERPASS_ENDPOINT, data={"data": query}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Warning: OSM road fetch failed: {e}")
            return []

        roads = []
        for element in data.get("elements", []):
            road = self._parse_road_element(element)
            if road:
                roads.append(road)

        print(f"Fetched {len(roads)} OSM roads")
        return roads

    def _parse_building_element(self, element: Dict) -> Optional[OSMBuilding]:
        """Parse OSM building element to OSMBuilding.

        Args:
            element: OSM element from Overpass API

        Returns:
            OSMBuilding or None if invalid
        """
        try:
            osm_id = element["id"]
            tags = element.get("tags", {})
            building_type = tags.get("building", "yes")

            # Extract geometry
            if element["type"] == "way":
                geo_polygon = [
                    (node["lon"], node["lat"]) for node in element.get("geometry", [])
                ]
            elif element["type"] == "relation":
                # For relations, use outer members
                outer_ways = [
                    member
                    for member in element.get("members", [])
                    if member.get("role") == "outer"
                ]
                if not outer_ways:
                    return None
                geo_polygon = [
                    (node["lon"], node["lat"])
                    for node in outer_ways[0].get("geometry", [])
                ]
            else:
                return None

            # Validate polygon
            if len(geo_polygon) < 3:
                return None

            # Close polygon if not already closed
            if geo_polygon[0] != geo_polygon[-1]:
                geo_polygon.append(geo_polygon[0])

            # Calculate area
            area_sqm = self._calculate_polygon_area(geo_polygon)

            return OSMBuilding(
                osm_id=osm_id,
                geo_polygon=geo_polygon,
                building_type=building_type,
                area_sqm=area_sqm,
                tags=tags,
            )

        except Exception as e:
            print(f"Warning: Failed to parse OSM building: {e}")
            return None

    def _parse_road_element(self, element: Dict) -> Optional[OSMRoad]:
        """Parse OSM road element to OSMRoad.

        Args:
            element: OSM element from Overpass API

        Returns:
            OSMRoad or None if invalid
        """
        try:
            osm_id = element["id"]
            tags = element.get("tags", {})
            highway_type = tags.get("highway", "unknown")

            # Extract geometry
            geo_linestring = [
                (node["lon"], node["lat"]) for node in element.get("geometry", [])
            ]

            # Validate linestring
            if len(geo_linestring) < 2:
                return None

            # Extract width if available
            width_m = None
            if "width" in tags:
                try:
                    width_m = float(tags["width"].replace("m", "").strip())
                except Exception:
                    pass

            return OSMRoad(
                osm_id=osm_id,
                geo_linestring=geo_linestring,
                highway_type=highway_type,
                width_m=width_m,
                tags=tags,
            )

        except Exception as e:
            print(f"Warning: Failed to parse OSM road: {e}")
            return None

    def _calculate_polygon_area(self, geo_polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area in square meters using pyproj.

        Args:
            geo_polygon: List of (lon, lat) coordinates

        Returns:
            Area in square meters
        """
        if len(geo_polygon) < 3:
            return 0.0

        geod = Geod(ellps="WGS84")
        lons, lats = zip(*geo_polygon)

        try:
            area, _ = geod.polygon_area_perimeter(lons, lats)
            return abs(area)
        except Exception:
            return 0.0
