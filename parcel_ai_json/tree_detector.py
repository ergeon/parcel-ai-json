"""Tree detection service using detectree library via Docker.

Due to macOS compatibility issues with detectree's C extensions, this service
runs detectree in a Docker container with Linux.
"""

from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import subprocess
import json


@dataclass
class TreePolygon:
    """Individual tree cluster polygon."""

    # Geographic polygon (list of lon/lat tuples)
    geo_polygon: List[Tuple[float, float]]

    # Pixel polygon (list of x/y tuples)
    pixel_polygon: List[Tuple[int, int]]

    # Area in square meters
    area_sqm: float

    # Area in pixels
    area_pixels: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "geo_polygon": self.geo_polygon,
            "pixel_polygon": self.pixel_polygon,
            "area_sqm": self.area_sqm,
            "area_pixels": self.area_pixels,
        }

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [self.geo_polygon],
            },
            "properties": {
                "feature_type": "tree_cluster",
                "area_sqm": self.area_sqm,
                "area_pixels": self.area_pixels,
            },
        }


@dataclass
class TreeDetection:
    """Tree detection results."""

    # Tree coverage statistics
    tree_pixel_count: int
    total_pixels: int
    tree_coverage_percent: float

    # Image dimensions
    width: int
    height: int

    # Tree mask image path (optional)
    tree_mask_path: Optional[str] = None

    # Tree polygons (optional, extracted from mask)
    tree_polygons: Optional[List[TreePolygon]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "tree_pixel_count": self.tree_pixel_count,
            "total_pixels": self.total_pixels,
            "tree_coverage_percent": self.tree_coverage_percent,
            "width": self.width,
            "height": self.height,
        }
        if self.tree_mask_path:
            result["tree_mask_path"] = self.tree_mask_path
        if self.tree_polygons is not None:
            result["tree_polygons"] = [p.to_dict() for p in self.tree_polygons]
        return result


class TreeDetectionService:
    """Tree detection service using detectree.

    Can run in two modes:
    1. Native mode (use_docker=False): Use detectree directly (Linux/Docker container)
    2. Docker mode (use_docker=True): Run detectree in Docker container (macOS compatibility)
    """

    def __init__(
        self,
        use_docker: bool = True,
        docker_image: str = "parcel-tree-detector",
        save_mask: bool = True,
        extract_polygons: bool = False,
        min_tree_area_pixels: int = 50,
        simplify_tolerance_meters: float = 0.5,
    ):
        """Initialize tree detection service.

        Args:
            use_docker: Whether to use Docker container
                (default: True for macOS compatibility)
            docker_image: Name of Docker image (when use_docker=True)
            save_mask: Whether to save tree mask as PNG
            extract_polygons: Whether to extract tree cluster polygons
            min_tree_area_pixels: Min area in pixels for clusters (default: 50)
            simplify_tolerance_meters: Tolerance in meters for polygon
                simplification using Shapely's topology-preserving algorithm
                (default: 0.5m). Set to 0.0 to disable simplification.
        """
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.save_mask = save_mask
        self.extract_polygons = extract_polygons
        self.min_tree_area_pixels = min_tree_area_pixels
        self.simplify_tolerance_meters = simplify_tolerance_meters
        self._clf = None

    def _load_native_classifier(self):
        """Load detectree classifier for native execution."""
        if self._clf is not None:
            return

        try:
            import detectree as dtr
        except ImportError:
            raise ImportError(
                "Tree detection requires detectree. "
                "Install with: pip install detectree"
            )

        self._clf = dtr.Classifier()

    def _extract_tree_polygons(
        self,
        mask,  # np.ndarray
        satellite_image: Dict,
        image_width: int,
        image_height: int,
    ) -> List[TreePolygon]:
        """Extract tree cluster polygons from binary mask.

        Args:
            mask: Binary mask where 1 = tree, 0 = not tree
            satellite_image: Satellite image metadata with center_lat, center_lon, zoom_level
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            List of TreePolygon objects for each detected tree cluster
        """
        import cv2
        import numpy as np
        from pyproj import Geod
        from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

        # Initialize coordinate converter
        converter = ImageCoordinateConverter(
            center_lat=satellite_image["center_lat"],
            center_lon=satellite_image["center_lon"],
            image_width_px=image_width,
            image_height_px=image_height,
            zoom_level=satellite_image.get("zoom_level", 20),
        )

        # Initialize geoid for area calculations
        geod = Geod(ellps="WGS84")

        # Find contours in the binary mask
        # Ensure mask is uint8
        mask_uint8 = mask.astype(np.uint8)

        # Find external contours only
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tree_polygons = []

        for contour in contours:
            # Get area in pixels
            area_pixels = cv2.contourArea(contour)

            # Filter out small noise regions
            if area_pixels < self.min_tree_area_pixels:
                continue

            # Convert contour to pixel polygon
            # Contour shape is (N, 1, 2), reshape to (N, 2)
            pixel_polygon_array = contour.reshape(-1, 2)
            pixel_polygon = [(int(x), int(y)) for x, y in pixel_polygon_array]

            # Ensure polygon is closed (first point == last point)
            if pixel_polygon[0] != pixel_polygon[-1]:
                pixel_polygon.append(pixel_polygon[0])

            # Convert to geographic coordinates
            geo_polygon = []
            for x, y in pixel_polygon:
                lon, lat = converter.pixel_to_geo(float(x), float(y))
                geo_polygon.append((lon, lat))

            # Simplify polygon if tolerance is set
            if self.simplify_tolerance_meters > 0:
                from shapely.geometry import Polygon
                from shapely import simplify

                # Create Shapely polygon
                # (exclude closing point for construction)
                shapely_poly = Polygon(geo_polygon[:-1])

                # Simplify using Shapely's topology-preserving algorithm
                # Convert meters to approximate degrees
                # At equator: 1 degree ≈ 111,320 meters
                tolerance_degrees = self.simplify_tolerance_meters / 111320.0
                simplified_poly = simplify(
                    shapely_poly,
                    tolerance_degrees,
                    preserve_topology=True,
                )

                # Extract coordinates from simplified polygon
                geo_polygon = list(simplified_poly.exterior.coords)

            # Calculate area in square meters using geodesic calculations
            # For polygons, we use the geod.polygon_area_perimeter method
            lons = [p[0] for p in geo_polygon[:-1]]  # Exclude last (duplicate) point
            lats = [p[1] for p in geo_polygon[:-1]]

            try:
                area_sqm, _ = geod.polygon_area_perimeter(lons, lats)
                area_sqm = abs(area_sqm)  # Area can be negative depending on winding
            except Exception:
                # If geodesic calculation fails, fall back to approximate calculation
                # using pixel area and meters per pixel squared
                area_sqm = area_pixels * (converter.meters_per_pixel ** 2)

            tree_polygons.append(
                TreePolygon(
                    geo_polygon=geo_polygon,
                    pixel_polygon=pixel_polygon,
                    area_sqm=round(area_sqm, 2),
                    area_pixels=int(area_pixels),
                )
            )

        return tree_polygons

    def _detect_native(
        self, img_path: Path, satellite_image: Dict
    ) -> TreeDetection:
        """Detect trees using native detectree (not Docker)."""
        import numpy as np
        from PIL import Image
        import tempfile

        # Load classifier
        self._load_native_classifier()

        # Load and ensure image is RGB (detectree requires 3-channel RGB)
        img = Image.open(img_path)
        img_rgb = img.convert("RGB")
        img_width, img_height = img.size

        # Save as temporary RGB image for detectree
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            img_rgb.save(tmp_path)

        try:
            # Run detectree - predict_img expects a file path
            y_pred = self._clf.predict_img(tmp_path)
        finally:
            # Clean up temporary file
            import os
            os.unlink(tmp_path)

        # Save tree mask as PNG with green trees on transparent background
        tree_mask_path = None
        if self.save_mask:
            mask_rgba = np.zeros((y_pred.shape[0], y_pred.shape[1], 4), dtype=np.uint8)
            mask_rgba[y_pred == 1] = [0, 255, 0, 180]  # Green with 70% opacity
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            tree_mask_path = str(img_path.parent / f"{img_path.stem}_tree_mask.png")
            mask_img.save(tree_mask_path)

        # Extract tree polygons if requested
        tree_polygons = None
        if self.extract_polygons:
            tree_polygons = self._extract_tree_polygons(
                y_pred, satellite_image, img_width, img_height
            )

        # Calculate statistics
        tree_pixels = int(np.sum(y_pred))
        total_pixels = int(y_pred.size)
        coverage_percent = float(100 * tree_pixels / total_pixels)

        return TreeDetection(
            tree_pixel_count=tree_pixels,
            total_pixels=total_pixels,
            tree_coverage_percent=coverage_percent,
            width=int(y_pred.shape[1]),
            height=int(y_pred.shape[0]),
            tree_mask_path=tree_mask_path if self.save_mask else None,
            tree_polygons=tree_polygons,
        )

    def detect_trees(self, satellite_image: Dict) -> TreeDetection:
        """Detect trees in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (required for polygon extraction)
                - center_lon: Center longitude (required for polygon extraction)
                - zoom_level: Zoom level (optional, default 20)

        Returns:
            TreeDetection with tree coverage statistics

        Raises:
            RuntimeError: If detection fails
        """
        img_path = Path(satellite_image["path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Use native detection if use_docker=False
        if not self.use_docker:
            return self._detect_native(img_path, satellite_image)

        # Prepare output path for tree mask
        tree_mask_path = None
        if self.save_mask:
            tree_mask_path = str(img_path.parent / f"{img_path.stem}_tree_mask.png")

        # Run detectree in Docker container
        try:
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{img_path.parent}:/images",
                self.docker_image,
                "python",
                "-c",
                f"""
import detectree as dtr
import numpy as np
from PIL import Image
import json

# Load and convert to RGB (images are indexed color)
img = Image.open('/images/{img_path.name}')
img_rgb = img.convert('RGB')

# Save temporarily
img_rgb.save('/tmp/temp_rgb.jpg')

# Run detectree
clf = dtr.Classifier()
y_pred = clf.predict_img('/tmp/temp_rgb.jpg')

# Save tree mask as PNG with green trees on transparent background
# Create RGBA image - green (0,255,0) where trees, transparent elsewhere
mask_rgba = np.zeros((y_pred.shape[0], y_pred.shape[1], 4), dtype=np.uint8)
mask_rgba[y_pred == 1] = [0, 255, 0, 180]  # Green with 70% opacity
mask_img = Image.fromarray(mask_rgba, 'RGBA')
mask_img.save('/images/{img_path.stem}_tree_mask.png')

# Output results as JSON
result = {{
    'tree_pixels': int(np.sum(y_pred)),
    'total_pixels': int(y_pred.size),
    'width': int(y_pred.shape[1]),
    'height': int(y_pred.shape[0]),
    'coverage_percent': float(100 * np.sum(y_pred) / y_pred.size)
}}
print(json.dumps(result))
""",
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Tree detection failed: {result.stderr}")

            # Parse JSON output from last line
            output_lines = result.stdout.strip().split("\n")
            json_output = output_lines[-1]
            data = json.loads(json_output)

            return TreeDetection(
                tree_pixel_count=data["tree_pixels"],
                total_pixels=data["total_pixels"],
                tree_coverage_percent=data["coverage_percent"],
                width=data["width"],
                height=data["height"],
                tree_mask_path=tree_mask_path if self.save_mask else None,
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError("Tree detection timed out after 60 seconds")
        except FileNotFoundError:
            raise RuntimeError(
                "Docker not found. Please install Docker to use tree detection."
            )
        except Exception as e:
            raise RuntimeError(f"Tree detection failed: {e}")
