"""
Parcel AI JSON - Standalone Vehicle Detection with GeoJSON Output

Provides AI-powered vehicle detection for satellite imagery:
- Vehicle detection using YOLOv8-OBB (trained on DOTA aerial dataset)
- Automatic coordinate conversion from pixels to WGS84
- GeoJSON output format
- Future: Building detection
- Future: Fence detection
"""

from parcel_ai_json.vehicle_detector import VehicleDetectionService, VehicleDetection
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

__version__ = "0.1.0"
__all__ = ["VehicleDetectionService", "VehicleDetection", "ImageCoordinateConverter"]
