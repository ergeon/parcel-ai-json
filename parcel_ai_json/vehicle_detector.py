"""Vehicle detection service for satellite imagery.

Uses YOLO-based models (YOLOv8, fine-tuned for overhead/satellite views) to detect
vehicles in satellite images and convert detections to geographic coordinates.
"""

from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

if TYPE_CHECKING:
    from parcel_geojson.core.image_coordinates import ImageCoordinateConverter


@dataclass
class VehicleDetection:
    """Represents a detected vehicle with pixel and geographic coordinates."""

    # Pixel coordinates (bounding box)
    pixel_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    # Geographic coordinates (polygon)
    geo_polygon: List[Tuple[float, float]]  # [(lon, lat), ...]

    # Detection metadata
    confidence: float
    class_name: str  # 'car', 'truck', 'vehicle', etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "confidence": float(self.confidence),
            "class_name": self.class_name,
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
            model_path: Path to YOLO model weights (e.g., 'yolov8n.pt', 'spacenet_vehicles.pt')
                       If None, uses default YOLOv8n (requires fine-tuning for satellites)
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

        # Vehicle class names to detect (YOLO COCO classes)
        # Standard YOLO: car=2, motorcycle=3, bus=5, truck=7
        # Custom models may have different class mappings
        self.vehicle_classes = {"car", "truck", "bus", "motorcycle", "vehicle"}

    def _load_model(self):
        """Lazy-load the YOLO model.

        Raises:
            ImportError: If ultralytics is not installed
            FileNotFoundError: If model_path doesn't exist
        """
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Vehicle detection requires ultralytics. "
                "Install with: pip install parcel-geojson[vehicle-detection]"
            )

        # Use default YOLOv8n if no model specified
        model_file = self.model_path or "yolov8n.pt"

        # Standard YOLO models that will be auto-downloaded
        standard_models = {
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
        }

        # Check if custom model exists (skip check for standard models)
        if self.model_path and model_file not in standard_models:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Please provide a valid YOLO model file."
                )

        self._model = YOLO(model_file)
        self._model.to(self.device)

    def detect_vehicles(
        self,
        image_path: str,
        coord_converter: "ImageCoordinateConverter",
    ) -> List[VehicleDetection]:
        """Detect vehicles in satellite image and convert to geographic coordinates.

        Args:
            image_path: Path to satellite image file
            coord_converter: ImageCoordinateConverter instance for pixel→geo conversion

        Returns:
            List of VehicleDetection objects with both pixel and geographic coordinates

        Raises:
            ImportError: If ultralytics not installed
            FileNotFoundError: If image_path or model_path don't exist
        """
        self._load_model()

        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference
        results = self._model(image_path, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Process detections
        for result in results:
            # Check if OBB model (oriented bounding boxes) or regular model
            boxes = (
                result.obb if hasattr(result, "obb") and result.obb is not None else result.boxes
            )

            if boxes is None or len(boxes) == 0:
                continue

            # Get the original image shape that YOLO used
            # result.orig_shape is (height, width) of the original image
            orig_height, orig_width = result.orig_shape

            # Check if YOLO resized the image
            if (
                orig_width != coord_converter.image_width_px
                or orig_height != coord_converter.image_height_px
            ):
                # Need to scale coordinates back to original image size
                scale_x = coord_converter.image_width_px / orig_width
                scale_y = coord_converter.image_height_px / orig_height
            else:
                scale_x = scale_y = 1.0

            # OBB and regular boxes have different iteration patterns
            is_obb = hasattr(result, "obb") and result.obb is not None

            if is_obb:
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

                    # Scale coordinates back to original image size if needed
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y

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
            else:
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

                    # Scale coordinates back to original image size if needed
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y

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

    def detect_vehicles_from_metadata(
        self,
        satellite_image: Dict,
    ) -> List[VehicleDetection]:
        """Detect vehicles using satellite image metadata dict.

        Convenience method that creates ImageCoordinateConverter from metadata.

        Args:
            satellite_image: Dict with keys:
                - path: str (image file path)
                - center_lat: float
                - center_lon: float
                - width_px: int (optional, will read from image if not provided)
                - height_px: int (optional, will read from image if not provided)
                - zoom_level: int (optional, default 20)

        Returns:
            List of VehicleDetection objects
        """
        from parcel_geojson.core.image_coordinates import ImageCoordinateConverter

        # Extract metadata
        image_path = satellite_image["path"]
        center_lat = satellite_image["center_lat"]
        center_lon = satellite_image["center_lon"]
        zoom_level = satellite_image.get("zoom_level", 20)

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

        # Create converter
        converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=width_px,
            image_height_px=height_px,
            zoom_level=zoom_level,
        )

        # Detect vehicles
        return self.detect_vehicles(image_path, converter)
