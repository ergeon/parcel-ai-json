"""Tree detection service using DeepForest.

DeepForest is a deep learning model for tree crown detection trained on
22 NEON sites across the United States. Returns bounding boxes for individual
tree crowns with confidence scores.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TreeBoundingBox:
    """Individual tree bounding box."""

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
class TreeDetection:
    """Tree detection results from DeepForest."""

    # List of detected tree bounding boxes
    trees: List[TreeBoundingBox]

    # Summary statistics
    tree_count: int
    average_confidence: Optional[float] = None
    average_crown_area_sqm: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "tree_count": self.tree_count,
            "trees": [tree.to_dict() for tree in self.trees],
        }
        if self.average_confidence is not None:
            result["average_confidence"] = self.average_confidence
        if self.average_crown_area_sqm is not None:
            result["average_crown_area_sqm"] = self.average_crown_area_sqm
        return result


class TreeDetectionService:
    """Tree detection service using DeepForest.

    Uses DeepForest deep learning model trained on 22 NEON sites across the US.
    Detects individual tree crowns and returns bounding boxes with confidence scores.
    """

    def __init__(
        self,
        model_name: str = "weecology/deepforest-tree",
        confidence_threshold: float = 0.1,
    ):
        """Initialize tree detection service.

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

    def detect_trees(self, satellite_image: Dict) -> TreeDetection:
        """Detect trees in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Zoom level (optional, default 20)

        Returns:
            TreeDetection with list of tree bounding boxes

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
            import pandas as pd
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

            # Filter by confidence threshold
            if len(boxes_df) > 0:
                boxes_df = boxes_df[boxes_df["score"] >= self.confidence_threshold]

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

            return TreeDetection(
                trees=trees,
                tree_count=tree_count,
                average_confidence=avg_confidence,
                average_crown_area_sqm=avg_crown_area,
            )

        except Exception as e:
            raise RuntimeError(f"Tree detection failed: {e}")
