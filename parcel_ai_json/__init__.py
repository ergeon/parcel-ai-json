"""
Parcel AI JSON - AI/ML Extensions for Parcel GeoJSON

Provides AI-powered enhancements for parcel GeoJSON generation:
- Vehicle detection using YOLOv8-OBB
- Future: Building detection
- Future: Fence detection
"""

from parcel_ai_json.vehicle_detector import VehicleDetectionService

__version__ = "0.1.0"
__all__ = ["VehicleDetectionService"]
