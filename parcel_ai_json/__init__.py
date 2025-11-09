"""
Parcel AI JSON - AI-Powered Detection for Satellite Imagery

Provides detection services for satellite imagery:
- Vehicle detection using YOLOv8-OBB/Segmentation models
- Driveway detection (requires custom-trained model)
- Automatic coordinate conversion from pixels to WGS84
- GeoJSON output format
- Future: Building detection, fence detection
"""

from parcel_ai_json.vehicle_detector import VehicleDetectionService, VehicleDetection
from parcel_ai_json.driveway_detector import DrivewayDetectionService, DrivewayDetection
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

__version__ = "0.1.0"
__all__ = [
    "VehicleDetectionService",
    "VehicleDetection",
    "DrivewayDetectionService",
    "DrivewayDetection",
    "ImageCoordinateConverter",
]
