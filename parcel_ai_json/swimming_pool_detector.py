"""Swimming pool detection service for satellite imagery.

Uses YOLOv8-OBB (DOTA dataset) to detect swimming pools in satellite images.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter


@dataclass
class SwimmingPoolDetection:
    """Represents a detected swimming pool with geographic coordinates."""

    # Pixel coordinates (bounding box)
    pixel_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    # Geographic coordinates (polygon)
    geo_polygon: List[Tuple[float, float]]  # [(lon, lat), ...]

    # Detection metadata
    confidence: float = 0.0
    area_sqm: float = 0.0  # Approximate area in square meters

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "confidence": float(self.confidence),
            "area_sqm": float(self.area_sqm),
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
                "feature_type": "swimming_pool",
                "confidence": self.confidence,
                "area_sqm": self.area_sqm,
                "pixel_bbox": list(self.pixel_bbox),
            },
        }


class SwimmingPoolDetectionService:
    """Service for detecting swimming pools in satellite imagery.

    Uses YOLOv8-OBB trained on DOTA dataset which includes 'swimming pool'
    as one of the 15 aerial imagery classes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ):
        """Initialize swimming pool detection service.

        Args:
            model_path: Path to YOLO-OBB model (default: yolov8m-obb.pt)
            confidence_threshold: Minimum confidence score (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the YOLOv8-OBB model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Swimming pool detection requires ultralytics. "
                "Install with: pip install parcel-ai-json"
            )

        # Use YOLOv8m-OBB (DOTA dataset includes swimming pool class)
        if self.model_path is None:
            model_file = "yolov8m-obb.pt"
            # Check models/ directory first
            models_dir = Path(__file__).parent.parent / "models"
            if (models_dir / model_file).exists():
                model_file = str(models_dir / model_file)
        else:
            model_file = self.model_path

        print(f"Loading YOLOv8-OBB model: {model_file}")
        print("  Model will be downloaded to ~/.ultralytics/ on first use")

        self._model = YOLO(model_file)
        self._model.to(self.device)

    def detect_swimming_pools(
        self,
        satellite_image: Dict,
    ) -> List[SwimmingPoolDetection]:
        """Detect swimming pools in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: str (image file path)
                - center_lat: float (center latitude WGS84)
                - center_lon: float (center longitude WGS84)
                - zoom_level: int (optional, default 20)

        Returns:
            List of SwimmingPoolDetection objects
        """
        self._load_model()

        # Get image path and verify it exists
        image_path = satellite_image["path"]
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create coordinate converter using factory method
        coord_converter = ImageCoordinateConverter.from_satellite_image(
            satellite_image, image_path
        )

        # Run inference
        results = self._model(image_path, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Process detections
        for result in results:
            if not hasattr(result, "obb") or result.obb is None:
                continue

            boxes = result.obb

            if len(boxes) == 0:
                continue

            # Iterate through detections
            for i in range(len(boxes)):
                # Get class name
                class_id = int(boxes.cls[i].item())
                class_name = self._model.names[class_id]

                # Filter to swimming pool only (class 14 in DOTA)
                if class_name.lower() != "swimming pool":
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # Get confidence score
                confidence = float(boxes.conf[i].item())

                # Convert pixel bbox to geographic polygon
                geo_polygon = coord_converter.bbox_to_polygon(x1, y1, x2, y2)

                # Calculate approximate area in square meters
                # Simple approximation using bbox
                width_m = coord_converter.meters_per_pixel * (x2 - x1)
                height_m = coord_converter.meters_per_pixel * (y2 - y1)
                area_sqm = width_m * height_m

                detection = SwimmingPoolDetection(
                    pixel_bbox=(x1, y1, x2, y2),
                    geo_polygon=geo_polygon,
                    confidence=confidence,
                    area_sqm=area_sqm,
                )

                detections.append(detection)

        return detections

    def detect_swimming_pools_geojson(self, satellite_image: Dict) -> Dict:
        """Detect swimming pools and return as GeoJSON FeatureCollection.

        Args:
            satellite_image: Dict with satellite metadata (see detect_swimming_pools)

        Returns:
            GeoJSON FeatureCollection with swimming pool features
        """
        detections = self.detect_swimming_pools(satellite_image)

        features = [detection.to_geojson_feature() for detection in detections]

        return {"type": "FeatureCollection", "features": features}
