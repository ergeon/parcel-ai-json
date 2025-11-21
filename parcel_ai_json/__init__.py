"""
Parcel AI JSON - AI-Powered Detection for Satellite Imagery

Provides unified detection service for satellite imagery:
- Vehicles (cars, trucks, etc.)
- Swimming pools
- Amenities (tennis courts, basketball courts, baseball diamonds, soccer fields)
- Automatic coordinate conversion from pixels to WGS84
- GeoJSON output format

Uses YOLOv8-OBB model trained on DOTA aerial dataset.
"""

from parcel_ai_json.property_detector import (
    PropertyDetectionService,
    PropertyDetections,
)
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

# Keep individual detectors available for advanced use cases
from parcel_ai_json.vehicle_detector import VehicleDetectionService, VehicleDetection
from parcel_ai_json.swimming_pool_detector import (
    SwimmingPoolDetectionService,
    SwimmingPoolDetection,
)
from parcel_ai_json.amenity_detector import (
    AmenityDetectionService,
    AmenityDetection,
)
from parcel_ai_json.sam_segmentation import SAMSegmentationService, SAMSegment
from parcel_ai_json.sam3_segmentation import SAM3SegmentationService, SAM3Detection
from parcel_ai_json.sam_labeler import (
    SAMSegmentLabeler,
    LabeledSAMSegment,
    LABEL_SCHEMA,
)
from parcel_ai_json.grounded_sam_detector import (
    GroundedSAMDetector,
    GroundedDetection,
)

__version__ = "0.1.0"
__all__ = [
    # Primary unified API
    "PropertyDetectionService",
    "PropertyDetections",
    # Utilities
    "ImageCoordinateConverter",
    # Individual detectors (advanced usage)
    "VehicleDetectionService",
    "VehicleDetection",
    "SwimmingPoolDetectionService",
    "SwimmingPoolDetection",
    "AmenityDetectionService",
    "AmenityDetection",
    # SAM segmentation (original SAM)
    "SAMSegmentationService",
    "SAMSegment",
    # SAM3 segmentation (open-vocabulary)
    "SAM3SegmentationService",
    "SAM3Detection",
    # SAM labeling
    "SAMSegmentLabeler",
    "LabeledSAMSegment",
    "LABEL_SCHEMA",
    # Grounded-SAM (text-prompted detection + segmentation)
    "GroundedSAMDetector",
    "GroundedDetection",
]
