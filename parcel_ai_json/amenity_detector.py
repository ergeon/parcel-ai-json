"""Amenity detection service for satellite imagery.

Uses YOLOv8-OBB (DOTA dataset) to detect residential amenities like tennis courts,
basketball courts, baseball diamonds, and soccer fields.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter


@dataclass
class AmenityDetection:
    """Represents a detected amenity with geographic coordinates."""

    # Amenity type (tennis court, basketball court, etc.)
    amenity_type: str

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
            "amenity_type": self.amenity_type,
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
                "feature_type": "amenity",
                "amenity_type": self.amenity_type,
                "confidence": self.confidence,
                "area_sqm": self.area_sqm,
                "pixel_bbox": list(self.pixel_bbox),
            },
        }


class AmenityDetectionService:
    """Service for detecting residential amenities in satellite imagery.

    Uses YOLOv8-OBB trained on DOTA dataset which includes:
    - Tennis courts (class 4)
    - Basketball courts (class 5)
    - Baseball diamonds (class 3)
    - Soccer ball fields (class 13)
    - Ground track fields (class 6) - may detect driveways/paved areas
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ):
        """Initialize amenity detection service.

        Args:
            model_path: Path to YOLO-OBB model (default: yolov8m-obb.pt)
            confidence_threshold: Minimum confidence score (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

        # DOTA classes we want to detect as amenities
        # Class IDs based on DOTA dataset
        self.amenity_classes = {
            "baseball diamond": 3,
            "tennis court": 4,
            "basketball court": 5,
            "ground track field": 6,
            "soccer ball field": 13,
        }

    def _load_model(self):
        """Lazy-load the YOLOv8-OBB model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Amenity detection requires ultralytics. "
                "Install with: pip install parcel-ai-json"
            )

        # Use YOLOv8m-OBB (DOTA dataset includes amenity classes)
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

    def detect_amenities(self, satellite_image: Dict) -> List[AmenityDetection]:
        """Detect amenities in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Optional zoom level (default 20)

        Returns:
            List of AmenityDetection objects
        """
        # Load model on first use
        self._load_model()

        # Get image path and validate
        img_path = Path(satellite_image["path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Get image metadata
        center_lat = satellite_image["center_lat"]
        center_lon = satellite_image["center_lon"]
        zoom_level = satellite_image.get("zoom_level", 20)

        # Create coordinate converter
        from PIL import Image

        with Image.open(img_path) as img:
            width, height = img.size

        coord_converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=width,
            image_height_px=height,
            zoom_level=zoom_level,
        )

        # Run inference
        results = self._model.predict(
            source=str(img_path),
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        # Process results
        for result in results:
            # Check if OBB (oriented bounding boxes) are available
            has_obb = hasattr(result, "obb") and result.obb is not None

            if has_obb:
                boxes = result.obb

                if boxes is None or len(boxes) == 0:
                    continue

                # Process each detection
                for i in range(len(boxes)):
                    # Get class name
                    class_id = int(boxes.cls[i].item())
                    class_name = self._model.names[class_id]

                    # Filter to amenity classes only
                    if class_name not in self.amenity_classes:
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # Get confidence score
                    confidence = float(boxes.conf[i].item())

                    # Convert pixel bbox to geographic polygon
                    geo_polygon = coord_converter.bbox_to_polygon(x1, y1, x2, y2)

                    # Calculate area in square meters (simple approximation using bbox)
                    width_m = coord_converter.meters_per_pixel * (x2 - x1)
                    height_m = coord_converter.meters_per_pixel * (y2 - y1)
                    area_sqm = width_m * height_m

                    detection = AmenityDetection(
                        amenity_type=class_name,
                        pixel_bbox=(x1, y1, x2, y2),
                        geo_polygon=geo_polygon,
                        confidence=confidence,
                        area_sqm=area_sqm,
                    )

                    detections.append(detection)

        return detections

    def detect_amenities_geojson(self, satellite_image: Dict) -> Dict:
        """Detect amenities and return as GeoJSON FeatureCollection.

        Args:
            satellite_image: Dict with satellite metadata

        Returns:
            GeoJSON FeatureCollection with amenity features
        """
        detections = self.detect_amenities(satellite_image)

        features = [detection.to_geojson_feature() for detection in detections]

        return {"type": "FeatureCollection", "features": features}
