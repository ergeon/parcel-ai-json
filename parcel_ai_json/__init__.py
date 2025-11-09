"""
Parcel AI JSON - AI-Powered Detection for Satellite Imagery

Provides detection services for satellite imagery:
- Vehicle detection using YOLOv8-OBB models
- Swimming pool detection using YOLOv8-OBB (DOTA dataset)
- Amenity detection (tennis courts, basketball courts, etc.)
- Automatic coordinate conversion from pixels to WGS84
- GeoJSON output format
"""

from parcel_ai_json.vehicle_detector import VehicleDetectionService, VehicleDetection
from parcel_ai_json.swimming_pool_detector import (
    SwimmingPoolDetectionService,
    SwimmingPoolDetection,
)
from parcel_ai_json.amenity_detector import (
    AmenityDetectionService,
    AmenityDetection,
)
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

__version__ = "0.1.0"
__all__ = [
    "VehicleDetectionService",
    "VehicleDetection",
    "SwimmingPoolDetectionService",
    "SwimmingPoolDetection",
    "AmenityDetectionService",
    "AmenityDetection",
    "ImageCoordinateConverter",
]
