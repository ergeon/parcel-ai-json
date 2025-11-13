"""Combined tree detection service using both DeepForest and detectree.

This module provides two complementary tree detection approaches:
1. DeepForest: Individual tree crown detection with bounding boxes and confidence scores
2. detectree: Tree coverage polygons and statistics

Both methods can be run in parallel to provide comprehensive tree analysis.
"""

from typing import Dict, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import subprocess
import json


@dataclass
class TreeBoundingBox:
    """Individual tree bounding box from DeepForest."""

    # Pixel bounding box (xmin, ymin, xmax, ymax)
    pixel_bbox: Tuple[float, float, float, float]

    # Geographic bounding box (lon_min, lat_min, lon_max, lat_max)
    geo_bbox: Tuple[float, float, float, float]

    # Confidence score (0.0 - 1.0)
    confidence: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "pixel_bbox": list(self.pixel_bbox),
            "geo_bbox": list(self.geo_bbox),
            "confidence": self.confidence,
        }

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON feature (bounding box as polygon)."""
        lon_min, lat_min, lon_max, lat_max = self.geo_bbox
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon_min, lat_min],
                        [lon_max, lat_min],
                        [lon_max, lat_max],
                        [lon_min, lat_max],
                        [lon_min, lat_min],
                    ]
                ],
            },
            "properties": {
                "feature_type": "tree",
                "confidence": self.confidence,
            },
        }


@dataclass
class TreePolygon:
    """Individual tree cluster polygon from detectree."""

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
    """Combined tree detection results from both DeepForest and detectree.

    Includes:
    - Individual tree crowns with confidence scores (DeepForest)
    - Tree coverage polygons and statistics (detectree)
    """

    # DeepForest results - individual tree bounding boxes
    trees: List[TreeBoundingBox]
    tree_count: int
    average_confidence: Optional[float] = None
    average_crown_area_sqm: Optional[float] = None

    # detectree results - tree coverage
    tree_pixel_count: Optional[int] = None
    total_pixels: Optional[int] = None
    tree_coverage_percent: Optional[float] = None
    tree_polygons: Optional[List[TreePolygon]] = None
    tree_mask_path: Optional[str] = None

    # Image dimensions
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            # DeepForest results
            "tree_count": self.tree_count,
            "trees": [tree.to_dict() for tree in self.trees],
        }

        # Optional DeepForest stats
        if self.average_confidence is not None:
            result["average_confidence"] = self.average_confidence
        if self.average_crown_area_sqm is not None:
            result["average_crown_area_sqm"] = self.average_crown_area_sqm

        # detectree results
        if self.tree_pixel_count is not None:
            result["tree_pixel_count"] = self.tree_pixel_count
        if self.total_pixels is not None:
            result["total_pixels"] = self.total_pixels
        if self.tree_coverage_percent is not None:
            result["tree_coverage_percent"] = self.tree_coverage_percent
        if self.tree_polygons is not None:
            result["tree_polygons"] = [p.to_dict() for p in self.tree_polygons]
        if self.tree_mask_path is not None:
            result["tree_mask_path"] = self.tree_mask_path

        # Image dimensions
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height

        return result


class DeepForestService:
    """Tree detection service using DeepForest.

    Uses DeepForest deep learning model trained on 22 NEON sites across the US.
    Detects individual tree crowns and returns bounding boxes with confidence scores.
    """

    def __init__(
        self,
        model_name: str = "weecology/deepforest-tree",
        confidence_threshold: float = 0.1,
    ):
        """Initialize DeepForest service.

        Args:
            model_name: Hugging Face model name (default: weecology/deepforest-tree)
            confidence_threshold: Minimum confidence threshold (0.0-1.0, default: 0.1)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self):
        """Lazy-load DeepForest model."""
        if self._model is not None:
            return

        try:
            from deepforest import main
        except ImportError:
            raise ImportError(
                "Tree detection requires deepforest. "
                "Install with: pip install deepforest"
            )

        # Initialize and load pretrained model
        self._model = main.deepforest()
        self._model.load_model(model_name=self.model_name)

    def detect_trees(self, satellite_image: Dict) -> Tuple[List[TreeBoundingBox], Dict]:
        """Detect trees in satellite image using DeepForest.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Zoom level (optional, default 20)

        Returns:
            Tuple of (list of TreeBoundingBox, statistics dict)

        Raises:
            FileNotFoundError: If image not found
            RuntimeError: If detection fails
        """
        img_path = Path(satellite_image["path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load model
        self._load_model()

        # Run detection
        try:
            from PIL import Image
            from parcel_ai_json.coordinate_converter import ImageCoordinateConverter
            from pyproj import Geod

            # Get image dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Initialize coordinate converter
            converter = ImageCoordinateConverter(
                center_lat=satellite_image["center_lat"],
                center_lon=satellite_image["center_lon"],
                image_width_px=img_width,
                image_height_px=img_height,
                zoom_level=satellite_image.get("zoom_level", 20),
            )

            # Initialize geoid for area calculations
            geod = Geod(ellps="WGS84")

            # Run DeepForest prediction
            boxes_df = self._model.predict_image(path=str(img_path))

            # Handle empty predictions (returns None if no trees detected)
            if boxes_df is None or len(boxes_df) == 0:
                return [], {
                    "tree_count": 0,
                    "average_confidence": None,
                    "average_crown_area_sqm": None,
                }

            # Filter by confidence threshold
            boxes_df = boxes_df[boxes_df["score"] >= self.confidence_threshold]

            # Check again after filtering
            if len(boxes_df) == 0:
                return [], {
                    "tree_count": 0,
                    "average_confidence": None,
                    "average_crown_area_sqm": None,
                }

            # Convert to TreeBoundingBox objects
            trees = []
            crown_areas = []

            for _, row in boxes_df.iterrows():
                xmin, ymin, xmax, ymax = (
                    row["xmin"],
                    row["ymin"],
                    row["xmax"],
                    row["ymax"],
                )
                confidence = row["score"]

                # Convert pixel bbox to geographic bbox
                lon_min, lat_min = converter.pixel_to_geo(float(xmin), float(ymax))
                lon_max, lat_max = converter.pixel_to_geo(float(xmax), float(ymin))

                # Calculate area in square meters
                # Approximate bbox as rectangle for area calculation
                try:
                    _, _, width_m = geod.inv(lon_min, lat_min, lon_max, lat_min)
                    _, _, height_m = geod.inv(lon_min, lat_min, lon_min, lat_max)
                    area_sqm = abs(width_m * height_m)
                    crown_areas.append(area_sqm)
                except Exception:
                    # If geodesic calculation fails, use pixel area
                    pixel_area = (xmax - xmin) * (ymax - ymin)
                    area_sqm = pixel_area * (converter.meters_per_pixel**2)
                    crown_areas.append(area_sqm)

                trees.append(
                    TreeBoundingBox(
                        pixel_bbox=(xmin, ymin, xmax, ymax),
                        geo_bbox=(lon_min, lat_min, lon_max, lat_max),
                        confidence=float(confidence),
                    )
                )

            # Calculate summary statistics
            tree_count = len(trees)
            avg_confidence = float(boxes_df["score"].mean()) if tree_count > 0 else None
            avg_crown_area = sum(crown_areas) / tree_count if tree_count > 0 else None

            stats = {
                "tree_count": tree_count,
                "average_confidence": avg_confidence,
                "average_crown_area_sqm": avg_crown_area,
            }

            return trees, stats

        except Exception as e:
            raise RuntimeError(f"DeepForest tree detection failed: {e}")


class DetectreeService:
    """Tree detection service using detectree library.

    Can run in two modes:
    1. Native (use_docker=False): Use detectree directly (Linux/Docker)
    2. Docker (use_docker=True): Run in Docker container (macOS compat)
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
        """Initialize detectree service.

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
            satellite_image: Satellite image metadata with center_lat,
                center_lon, zoom_level
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

        # Ensure mask is uint8
        mask_uint8 = mask.astype(np.uint8)

        # Apply morphological closing to merge nearby tree fragments
        # into continuous tree crowns (conservative to avoid false positives)
        closing_kernel_size = 10  # pixels (about 1m at zoom 20)
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )
        mask_processed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, closing_kernel)

        # Find external contours only
        contours, _ = cv2.findContours(
            mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
                area_sqm = area_pixels * (converter.meters_per_pixel**2)

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
    ) -> Tuple[Dict, Optional[str], Optional[List[TreePolygon]]]:
        """Detect trees using native detectree (not Docker).

        Returns:
            Tuple of (statistics dict, tree_mask_path, tree_polygons)
        """
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
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
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

        stats = {
            "tree_pixel_count": tree_pixels,
            "total_pixels": total_pixels,
            "tree_coverage_percent": coverage_percent,
            "width": int(y_pred.shape[1]),
            "height": int(y_pred.shape[0]),
        }

        return stats, tree_mask_path, tree_polygons

    def detect_trees(
        self, satellite_image: Dict
    ) -> Tuple[Dict, Optional[str], Optional[List[TreePolygon]]]:
        """Detect trees in satellite image using detectree.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (required for polygon extraction)
                - center_lon: Center longitude (required for polygon extraction)
                - zoom_level: Zoom level (optional, default 20)

        Returns:
            Tuple of (statistics dict, tree_mask_path, tree_polygons)

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

            stats = {
                "tree_pixel_count": data["tree_pixels"],
                "total_pixels": data["total_pixels"],
                "tree_coverage_percent": data["coverage_percent"],
                "width": data["width"],
                "height": data["height"],
            }

            # Docker mode doesn't support polygon extraction
            tree_polygons = None

            return stats, tree_mask_path if self.save_mask else None, tree_polygons

        except subprocess.TimeoutExpired:
            raise RuntimeError("Tree detection timed out after 60 seconds")
        except FileNotFoundError:
            raise RuntimeError(
                "Docker not found. Please install Docker to use tree detection."
            )
        except Exception as e:
            raise RuntimeError(f"detectree tree detection failed: {e}")


class CombinedTreeDetectionService:
    """Combined tree detection service using both DeepForest and detectree.

    Runs both detection methods in parallel to provide:
    - Individual tree crowns with confidence scores (DeepForest)
    - Tree coverage polygons and statistics (detectree)
    """

    def __init__(
        self,
        # DeepForest parameters
        deepforest_model_name: str = "weecology/deepforest-tree",
        deepforest_confidence_threshold: float = 0.1,
        # detectree parameters
        detectree_use_docker: bool = True,
        detectree_docker_image: str = "parcel-tree-detector",
        detectree_save_mask: bool = False,  # Disable mask, use polygons instead
        detectree_extract_polygons: bool = True,
        detectree_min_tree_area_pixels: int = 50,
        detectree_simplify_tolerance_meters: float = 0.5,
        # Control which services to run
        use_deepforest: bool = True,
        use_detectree: bool = True,
    ):
        """Initialize combined tree detection service.

        Args:
            deepforest_model_name: Hugging Face model name for DeepForest
            deepforest_confidence_threshold: Minimum confidence threshold for DeepForest
            detectree_use_docker: Whether to use Docker for detectree
            detectree_docker_image: Docker image name for detectree
            detectree_save_mask: Whether to save tree mask
            detectree_extract_polygons: Whether to extract tree polygons
            detectree_min_tree_area_pixels: Minimum tree area in pixels
            detectree_simplify_tolerance_meters: Polygon simplification tolerance
            use_deepforest: Whether to run DeepForest detection
            use_detectree: Whether to run detectree detection
        """
        self.use_deepforest = use_deepforest
        self.use_detectree = use_detectree

        # Initialize DeepForest service
        if self.use_deepforest:
            self.deepforest = DeepForestService(
                model_name=deepforest_model_name,
                confidence_threshold=deepforest_confidence_threshold,
            )
        else:
            self.deepforest = None

        # Initialize detectree service
        if self.use_detectree:
            self.detectree = DetectreeService(
                use_docker=detectree_use_docker,
                docker_image=detectree_docker_image,
                save_mask=detectree_save_mask,
                extract_polygons=detectree_extract_polygons,
                min_tree_area_pixels=detectree_min_tree_area_pixels,
                simplify_tolerance_meters=detectree_simplify_tolerance_meters,
            )
        else:
            self.detectree = None

    def detect_trees(self, satellite_image: Dict) -> TreeDetection:
        """Detect trees using both DeepForest and detectree.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Zoom level (optional, default 20)

        Returns:
            TreeDetection with combined results from both methods

        Raises:
            FileNotFoundError: If image not found
            RuntimeError: If detection fails
        """
        img_path = Path(satellite_image["path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Initialize result variables
        trees = []
        tree_count = 0
        average_confidence = None
        average_crown_area_sqm = None

        tree_pixel_count = None
        total_pixels = None
        tree_coverage_percent = None
        tree_polygons = None
        tree_mask_path = None
        width = None
        height = None

        # Run DeepForest detection
        if self.use_deepforest:
            try:
                trees, deepforest_stats = self.deepforest.detect_trees(satellite_image)
                tree_count = deepforest_stats["tree_count"]
                average_confidence = deepforest_stats["average_confidence"]
                average_crown_area_sqm = deepforest_stats["average_crown_area_sqm"]
            except Exception as e:
                raise RuntimeError(f"DeepForest detection failed: {e}")

        # Run detectree detection
        if self.use_detectree:
            try:
                detectree_stats, tree_mask_path, tree_polygons = (
                    self.detectree.detect_trees(satellite_image)
                )
                tree_pixel_count = detectree_stats["tree_pixel_count"]
                total_pixels = detectree_stats["total_pixels"]
                tree_coverage_percent = detectree_stats["tree_coverage_percent"]
                width = detectree_stats["width"]
                height = detectree_stats["height"]
            except Exception as e:
                raise RuntimeError(f"detectree detection failed: {e}")

        return TreeDetection(
            # DeepForest results
            trees=trees,
            tree_count=tree_count,
            average_confidence=average_confidence,
            average_crown_area_sqm=average_crown_area_sqm,
            # detectree results
            tree_pixel_count=tree_pixel_count,
            total_pixels=total_pixels,
            tree_coverage_percent=tree_coverage_percent,
            tree_polygons=tree_polygons,
            tree_mask_path=tree_mask_path,
            # Image dimensions
            width=width,
            height=height,
        )


# Maintain backward compatibility - TreeDetectionService points to combined service
TreeDetectionService = CombinedTreeDetectionService
