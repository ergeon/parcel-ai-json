"""Coordinate conversion for satellite imagery.

Converts between pixel coordinates and WGS84 geographic coordinates using
Web Mercator projection (EPSG:3857).
"""

from typing import Tuple, List
import math


class ImageCoordinateConverter:
    """Convert between pixel coordinates and geographic coordinates.

    Uses Web Mercator projection (EPSG:3857) for satellite imagery.
    Follows Google Maps/Mapbox tile coordinate system.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        image_width_px: int,
        image_height_px: int,
        zoom_level: int = 20,
        meters_per_pixel: float = None,
    ):
        """Initialize coordinate converter.

        Args:
            center_lat: Center latitude of image (WGS84)
            center_lon: Center longitude of image (WGS84)
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            zoom_level: Map zoom level (default: 20 for satellite imagery)
            meters_per_pixel: Override automatic calculation (optional)
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.zoom_level = zoom_level

        # Calculate meters per pixel at this zoom level and latitude
        # Using Web Mercator projection (EPSG:3857)
        if meters_per_pixel is not None:
            self.meters_per_pixel = meters_per_pixel
        else:
            # Web Mercator formula:
            # meters_per_pixel = (Earth circumference / 256) / (2^zoom) / cos(lat)
            earth_circumference = 40075016.686  # meters at equator
            self.meters_per_pixel = (
                earth_circumference
                / 256
                / (2**zoom_level)
                / math.cos(math.radians(center_lat))
            )

    def pixel_to_geo(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates (WGS84).

        Args:
            pixel_x: X coordinate in pixels (0 = left edge)
            pixel_y: Y coordinate in pixels (0 = top edge)

        Returns:
            Tuple of (longitude, latitude) in WGS84
        """
        # Calculate offset from center in pixels
        center_x = self.image_width_px / 2
        center_y = self.image_height_px / 2

        dx_pixels = pixel_x - center_x
        dy_pixels = pixel_y - center_y

        # Convert to meters
        dx_meters = dx_pixels * self.meters_per_pixel
        dy_meters = -dy_pixels * self.meters_per_pixel  # Y increases downward in images

        # Convert meters to degrees
        # Using simple equirectangular approximation (good enough for small areas)
        lat_offset = dy_meters / 111319.9  # meters per degree latitude
        lon_offset = dx_meters / (111319.9 * math.cos(math.radians(self.center_lat)))

        lon = self.center_lon + lon_offset
        lat = self.center_lat + lat_offset

        return (lon, lat)

    def bbox_to_polygon(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> List[Tuple[float, float]]:
        """Convert pixel bounding box to geographic polygon.

        Args:
            x1, y1: Top-left corner in pixels
            x2, y2: Bottom-right corner in pixels

        Returns:
            List of (lon, lat) tuples forming a closed polygon (5 points)
        """
        # Convert corners to geo coordinates
        top_left = self.pixel_to_geo(x1, y1)
        top_right = self.pixel_to_geo(x2, y1)
        bottom_right = self.pixel_to_geo(x2, y2)
        bottom_left = self.pixel_to_geo(x1, y2)

        # Return closed polygon (first point repeated at end)
        return [top_left, top_right, bottom_right, bottom_left, top_left]
