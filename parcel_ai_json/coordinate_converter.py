"""Coordinate conversion for satellite imagery.

Converts between pixel coordinates and WGS84 geographic coordinates using
geodesic calculations with pyproj for accuracy.
"""

from typing import Tuple, List, Dict, Optional
import math
from pyproj import CRS, Geod


def get_image_dimensions(
    satellite_image: Dict, image_path: Optional[str] = None
) -> Tuple[int, int]:
    """Get image dimensions from metadata or file.

    Args:
        satellite_image: Satellite image metadata dict
        image_path: Optional path to image file

    Returns:
        Tuple of (width_px, height_px)

    Raises:
        ValueError: If dimensions not in metadata and no image_path provided
        ImportError: If PIL not installed and needed to read dimensions
    """
    width_px = satellite_image.get("width_px")
    height_px = satellite_image.get("height_px")

    if width_px is None or height_px is None:
        if image_path is None:
            raise ValueError(
                "Image dimensions not in metadata. "
                "Must provide image_path to read from file."
            )

        try:
            from PIL import Image

            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            raise ImportError(
                "PIL (Pillow) is required to read image dimensions. "
                "Install with: pip install pillow"
            )

    return width_px, height_px


class ImageCoordinateConverter:
    """Convert between pixel coordinates and geographic coordinates.

    Uses geodesic calculations via pyproj.Geod for accurate conversion.
    Same implementation as parcel-geojson for consistency.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        image_width_px: int,
        image_height_px: int,
        zoom_level: int = 20,
    ):
        """Initialize converter with satellite image metadata.

        Uses EXACT same formulas as parcel-geojson for consistency.

        Args:
            center_lat: Latitude of image center
            center_lon: Longitude of image center
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            zoom_level: Google Maps zoom level (default 20)
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.zoom_level = zoom_level

        # Initialize pyproj Geod for geodesic calculations
        self.geod = Geod(ellps="WGS84")

        # Calculate meters per pixel using EXACT same formula as serializer
        # Get Earth's equatorial radius from WGS84 ellipsoid
        wgs84 = CRS("EPSG:4326")
        earth_radius = wgs84.ellipsoid.semi_major_metre

        google_map_magic_const = (2 * math.pi * earth_radius) / 256
        self.meters_per_pixel = (
            google_map_magic_const * math.cos(center_lat * math.pi / 180)
        ) / (2**zoom_level)

    @classmethod
    def from_satellite_image(
        cls, satellite_image: Dict, image_path: Optional[str] = None
    ) -> "ImageCoordinateConverter":
        """Create converter from satellite image metadata.

        Factory method that extracts dimensions and creates converter.

        Args:
            satellite_image: Satellite image metadata dict with keys:
                - center_lat: Latitude of image center
                - center_lon: Longitude of image center
                - zoom_level: Google Maps zoom level (optional, default 20)
                - width_px: Image width (optional, read from file if missing)
                - height_px: Image height (optional, read from file if missing)
            image_path: Optional path to image file (needed if dimensions
                not in metadata)

        Returns:
            ImageCoordinateConverter instance

        Raises:
            ValueError: If required metadata missing
            ImportError: If PIL needed but not installed
        """
        width_px, height_px = get_image_dimensions(satellite_image, image_path)

        return cls(
            center_lat=satellite_image["center_lat"],
            center_lon=satellite_image["center_lon"],
            image_width_px=width_px,
            image_height_px=height_px,
            zoom_level=satellite_image.get("zoom_level", 20),
        )

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert geographic coordinates (lon, lat) to image pixel coordinates.

        Inverse of pixel_to_geo. Uses geodesic calculations for accuracy.

        Args:
            lon: Longitude in WGS84 degrees
            lat: Latitude in WGS84 degrees

        Returns:
            Tuple of (pixel_x, pixel_y) from top-left of image
        """
        # Calculate geodesic distance and azimuth from image center to point
        azimuth, _, distance = self.geod.inv(self.center_lon, self.center_lat, lon, lat)

        # Decompose into East-West and North-South components
        # Azimuth is in degrees clockwise from North
        azimuth_rad = math.radians(azimuth)
        dx_meters = distance * math.sin(azimuth_rad)  # East-West offset
        dy_meters = distance * math.cos(azimuth_rad)  # North-South offset

        # Note: Image Y increases downward (South), so negate dy_meters
        dy_meters = -dy_meters

        # Convert meters to pixels
        dx_pixels = dx_meters / self.meters_per_pixel
        dy_pixels = dy_meters / self.meters_per_pixel

        # Convert offset from center to absolute pixel coordinates
        pixel_x = (self.image_width_px / 2.0) + dx_pixels
        pixel_y = (self.image_height_px / 2.0) + dy_pixels

        return (pixel_x, pixel_y)

    def pixel_to_geo(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert image pixel coordinates to geographic coordinates (lon, lat).

        Uses geodesic calculations via pyproj.Geod for accuracy.

        Args:
            pixel_x: X coordinate in pixels (from top-left of image)
            pixel_y: Y coordinate in pixels (from top-left of image)

        Returns:
            Tuple of (longitude, latitude) in WGS84
        """
        # Calculate offset from image center in pixels
        dx_pixels = pixel_x - (self.image_width_px / 2.0)
        dy_pixels = pixel_y - (self.image_height_px / 2.0)

        # Convert pixel offsets to meters
        dx_meters = dx_pixels * self.meters_per_pixel
        dy_meters = dy_pixels * self.meters_per_pixel

        # Use geodesic forward calculation to find new position
        # East/West offset (90° = East, 270° = West)
        azimuth_ew = 90 if dx_meters >= 0 else 270
        lon_temp, lat_temp, _ = self.geod.fwd(
            self.center_lon, self.center_lat, azimuth_ew, abs(dx_meters)
        )

        # North/South offset from the adjusted position (0° = North, 180° = South)
        # Note: Image Y increases downward, so positive dy_meters means South
        azimuth_ns = 180 if dy_meters >= 0 else 0
        lon, lat, _ = self.geod.fwd(lon_temp, lat_temp, azimuth_ns, abs(dy_meters))

        return (lon, lat)

    def bbox_to_polygon(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> List[Tuple[float, float]]:
        """Convert pixel bounding box to geographic polygon.

        Args:
            x1: Left x coordinate
            y1: Top y coordinate
            x2: Right x coordinate
            y2: Bottom y coordinate

        Returns:
            List of (lon, lat) tuples forming a closed polygon (5 points)
        """
        # Convert corners
        top_left = self.pixel_to_geo(x1, y1)
        top_right = self.pixel_to_geo(x2, y1)
        bottom_right = self.pixel_to_geo(x2, y2)
        bottom_left = self.pixel_to_geo(x1, y2)

        # Return closed polygon (first point repeated at end)
        return [top_left, top_right, bottom_right, bottom_left, top_left]

    def get_image_bounds(self) -> Dict[str, float]:
        """Get geographic bounds of the entire image.

        Returns:
            Dict with keys: north, south, east, west
        """
        top_left = self.pixel_to_geo(0, 0)
        bottom_right = self.pixel_to_geo(self.image_width_px, self.image_height_px)

        return {
            "west": top_left[0],
            "north": top_left[1],
            "east": bottom_right[0],
            "south": bottom_right[1],
        }
