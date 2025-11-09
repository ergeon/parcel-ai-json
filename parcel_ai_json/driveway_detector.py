"""Driveway detection service for satellite imagery.

Uses semantic segmentation to detect driveways, pathways, and parking areas
in satellite images.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter


@dataclass
class DrivewayDetection:
    """Represents a detected driveway with geographic coordinates."""

    # Geographic coordinates (polygon)
    geo_polygon: List[Tuple[float, float]]  # [(lon, lat), ...]

    # Pixel coordinates (polygon)
    pixel_polygon: List[Tuple[float, float]]  # [(x, y), ...]

    # Detection metadata
    confidence: float = 0.0
    area_sqm: float = 0.0  # Area in square meters

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [self.geo_polygon],
            },
            "properties": {
                "feature_type": "driveway",
                "confidence": self.confidence,
                "area_sqm": self.area_sqm,
                "pixel_polygon": self.pixel_polygon,
            },
        }


class DrivewayDetectionService:
    """Service for detecting driveways in satellite imagery.

    Uses semantic segmentation or custom-trained models to identify
    driveways, pathways, and parking areas.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize driveway detection service.

        Args:
            model_path: Path to segmentation model (custom driveway model or SAM)
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the segmentation model.

        For driveway detection, we need a model trained on:
        - Satellite/aerial imagery
        - Road/pavement/driveway classes

        Options:
        1. Custom YOLOv8-seg model trained on driveway data
        2. DeepLabV3/SegFormer trained on satellite imagery
        3. SAM (Segment Anything Model) with prompts

        Raises:
            NotImplementedError: Driveway detection requires a custom-trained model
        """
        if self._model is not None:
            return

        raise NotImplementedError(
            "Driveway detection requires a custom-trained model.\n"
            "\n"
            "Options:\n"
            "1. Train YOLOv8-seg on satellite imagery with driveway annotations\n"
            "2. Use DeepLabV3/SegFormer trained on satellite datasets\n"
            "3. Use SAM (Segment Anything Model) with manual prompts\n"
            "\n"
            "Standard YOLO models don't include 'driveway' as a class.\n"
            "For now, you can:\n"
            "- Use VehicleDetectionService with model_type='seg' to detect roads\n"
            "- Train a custom model on your driveway dataset\n"
            "- Use external segmentation models (e.g., SegFormer)\n"
        )

    def detect_driveways(
        self,
        satellite_image: Dict,
    ) -> List[DrivewayDetection]:
        """Detect driveways in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: str (image file path)
                - center_lat: float (center latitude WGS84)
                - center_lon: float (center longitude WGS84)
                - zoom_level: int (optional, default 20)

        Returns:
            List of DrivewayDetection objects

        Raises:
            NotImplementedError: Custom model required
        """
        self._load_model()
        # Model loading will raise NotImplementedError
        return []

    def detect_driveways_geojson(self, satellite_image: Dict) -> Dict:
        """Detect driveways and return as GeoJSON FeatureCollection.

        Args:
            satellite_image: Dict with satellite metadata (see detect_driveways)

        Returns:
            GeoJSON FeatureCollection with driveway features
        """
        detections = self.detect_driveways(satellite_image)

        features = [detection.to_geojson_feature() for detection in detections]

        return {"type": "FeatureCollection", "features": features}
