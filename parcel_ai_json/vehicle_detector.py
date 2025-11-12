"""Vehicle detection service for satellite imagery.

Uses YOLO-based models (YOLOv8, fine-tuned for overhead/satellite views) to detect
vehicles in satellite images and return GeoJSON.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter


@dataclass
class VehicleDetection:
    """Represents a detected vehicle with both pixel and geographic coordinates."""

    # Pixel coordinates (bounding box)
    pixel_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    # Geographic coordinates (polygon)
    geo_polygon: List[Tuple[float, float]] = field(default_factory=list)  # [(lon, lat), ...]

    # Detection metadata
    confidence: float = 0.0
    class_name: str = ""  # 'car', 'truck', 'vehicle', etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "confidence": float(self.confidence),
            "class_name": self.class_name,
        }

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [self.geo_polygon],  # GeoJSON polygon needs array of rings
            },
            "properties": {
                "feature_type": "vehicle",
                "vehicle_class": self.class_name,
                "confidence": self.confidence,
                "pixel_bbox": list(self.pixel_bbox),
            },
        }


class VehicleDetectionService:
    """Service for detecting vehicles in satellite imagery.

    This is a DDD service that coordinates vehicle detection using
    YOLO-based models and converts pixel coordinates to geographic coordinates.

    Supports:
    - YOLOv8 (standard or fine-tuned for overhead imagery)
    - Custom YOLO models trained on satellite data
    - SpaceNet-trained models
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        device: str = "cpu",
    ):
        """Initialize vehicle detection service.

        Args:
            model_path: Path to YOLO model weights (e.g., 'yolov8m-obb.pt')
                       If None, uses yolov8m-obb.pt (best for aerial imagery)
                       Model will be auto-downloaded on first use to ~/.ultralytics/
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

        # Vehicle class names to detect
        # DOTA dataset (aerial imagery): "small vehicle", "large vehicle"
        # COCO dataset (ground-level): car=2, motorcycle=3, bus=5, truck=7
        self.vehicle_classes = {"car", "truck", "bus", "motorcycle", "vehicle"}

    def _load_model(self):
        """Lazy-load the YOLO model.

        Models are automatically downloaded to ~/.ultralytics/ on first use.

        Raises:
            ImportError: If ultralytics is not installed
            FileNotFoundError: If custom model_path doesn't exist
        """
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Vehicle detection requires ultralytics. "
                "Install with: pip install parcel-ai-json"
            )

        # Use default OBB model if not specified
        if self.model_path is None:
            model_file = "yolov8m-obb.pt"  # Best for aerial imagery
            # Check models/ directory first
            models_dir = Path(__file__).parent.parent / "models"
            if (models_dir / model_file).exists():
                model_file = str(models_dir / model_file)
        else:
            model_file = self.model_path

        # Standard YOLO models that will be auto-downloaded by ultralytics
        auto_download_models = {
            # Regular models
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
            # OBB (Oriented Bounding Box) models for aerial imagery
            "yolov8n-obb.pt",
            "yolov8s-obb.pt",
            "yolov8m-obb.pt",
            "yolov8l-obb.pt",
            "yolov8x-obb.pt",
            "yolov8n-obb",
            "yolov8s-obb",
            "yolov8m-obb",
            "yolov8l-obb",
            "yolov8x-obb",
        }

        # Check if custom model exists (skip check for auto-download models)
        if self.model_path and model_file not in auto_download_models:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please provide a valid YOLO model file."
                )

        # Load model (ultralytics will auto-download to ~/.ultralytics/ if needed)
        print(f"Loading YOLOv8 model: {model_file}")
        if model_file in auto_download_models:
            print("  Model will be downloaded to ~/.ultralytics/ on first use")

        self._model = YOLO(model_file)
        self._model.to(self.device)

    def detect_vehicles(
        self,
        satellite_image: Dict,
    ) -> List[VehicleDetection]:
        """Detect vehicles in satellite image and return with geographic coordinates.

        Args:
            satellite_image: Dict with keys:
                - path: str (image file path)
                - center_lat: float (center latitude WGS84)
                - center_lon: float (center longitude WGS84)
                - width_px: int (optional, will read from image if not provided)
                - height_px: int (optional, will read from image if not provided)
                - zoom_level: int (optional, default 20)

        Returns:
            List of VehicleDetection objects with both pixel and geo coordinates

        Raises:
            ImportError: If ultralytics not installed
            FileNotFoundError: If image_path doesn't exist
        """
        self._load_model()

        # Extract metadata
        image_path = satellite_image["path"]
        center_lat = satellite_image["center_lat"]
        center_lon = satellite_image["center_lon"]
        zoom_level = satellite_image.get("zoom_level", 20)

        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get image dimensions
        width_px = satellite_image.get("width_px")
        height_px = satellite_image.get("height_px")

        if width_px is None or height_px is None:
            # Read from image file
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    width_px, height_px = img.size
            except ImportError:
                raise ImportError(
                    "PIL (Pillow) is required to read image dimensions. "
                    "Install with: pip install pillow"
                )

        # Create coordinate converter
        coord_converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=width_px,
            image_height_px=height_px,
            zoom_level=zoom_level,
        )

        # Run inference
        results = self._model(image_path, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Process detections
        for result in results:
            # Check model type: OBB or regular boxes
            has_obb = hasattr(result, "obb") and result.obb is not None

            # For OBB models (oriented bounding boxes) - best for aerial imagery
            if has_obb:
                boxes = result.obb

                if boxes is None or len(boxes) == 0:
                    continue
                # OBB format: boxes.cls, boxes.conf, boxes.xyxy are all tensors
                for i in range(len(boxes)):
                    # Get class name
                    class_id = int(boxes.cls[i].item())
                    class_name = self._model.names[class_id]

                    # Filter to vehicle classes only
                    # (DOTA dataset uses "small vehicle" and "large vehicle")
                    class_lower = class_name.lower()
                    is_vehicle = (
                        class_lower in self.vehicle_classes
                        or "vehicle" in class_lower
                        or "car" in class_lower
                    )
                    if not is_vehicle:
                        continue

                    # Get bounding box coordinates from xyxy tensor
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # Get confidence score
                    confidence = float(boxes.conf[i].item())

                    # Convert pixel bbox to geographic polygon
                    geo_polygon = coord_converter.bbox_to_polygon(x1, y1, x2, y2)

                    detection = VehicleDetection(
                        pixel_bbox=(x1, y1, x2, y2),
                        geo_polygon=geo_polygon,
                        confidence=confidence,
                        class_name=class_name,
                    )

                    detections.append(detection)

            # For regular bounding box models
            else:
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                # Regular format: iterate through box objects
                for box in boxes:
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self._model.names[class_id]

                    # Filter to vehicle classes only
                    if class_name.lower() not in self.vehicle_classes:
                        continue

                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Get confidence score
                    confidence = float(box.conf[0])

                    # Convert pixel bbox to geographic polygon
                    geo_polygon = coord_converter.bbox_to_polygon(x1, y1, x2, y2)

                    detection = VehicleDetection(
                        pixel_bbox=(x1, y1, x2, y2),
                        geo_polygon=geo_polygon,
                        confidence=confidence,
                        class_name=class_name,
                    )

                    detections.append(detection)

        return detections

    def detect_vehicles_geojson(self, satellite_image: Dict) -> Dict:
        """Detect vehicles and return as GeoJSON FeatureCollection.

        Args:
            satellite_image: Dict with satellite metadata (see detect_vehicles)

        Returns:
            GeoJSON FeatureCollection with vehicle features
        """
        detections = self.detect_vehicles(satellite_image)

        features = [detection.to_geojson_feature() for detection in detections]

        return {"type": "FeatureCollection", "features": features}
