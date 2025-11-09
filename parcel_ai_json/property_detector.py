"""Unified property detection service for satellite imagery.

Detects all property features in a single call:
- Vehicles (cars, trucks, etc.)
- Swimming pools
- Amenities (tennis courts, basketball courts, etc.)

Uses YOLOv8-OBB model trained on DOTA aerial dataset.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from parcel_ai_json.vehicle_detector import VehicleDetectionService, VehicleDetection
from parcel_ai_json.swimming_pool_detector import (
    SwimmingPoolDetectionService,
    SwimmingPoolDetection,
)
from parcel_ai_json.amenity_detector import AmenityDetectionService, AmenityDetection
from parcel_ai_json.tree_detector import TreeDetectionService, TreeDetection


@dataclass
class PropertyDetections:
    """Container for all property detections."""

    vehicles: List[VehicleDetection]
    swimming_pools: List[SwimmingPoolDetection]
    amenities: List[AmenityDetection]
    trees: TreeDetection

    def to_geojson(self) -> Dict:
        """Convert all detections to GeoJSON FeatureCollection."""
        features = []

        # Add vehicle features
        for vehicle in self.vehicles:
            features.append(vehicle.to_geojson_feature())

        # Add pool features
        for pool in self.swimming_pools:
            features.append(pool.to_geojson_feature())

        # Add amenity features
        for amenity in self.amenities:
            features.append(amenity.to_geojson_feature())

        # Add tree coverage metadata (not a spatial feature, but coverage info)
        tree_coverage = {
            "tree_coverage_percent": self.trees.tree_coverage_percent,
            "tree_pixel_count": self.trees.tree_pixel_count,
            "total_pixels": self.trees.total_pixels,
            "image_width": self.trees.width,
            "image_height": self.trees.height,
        }

        # Include tree mask path if available
        if self.trees.tree_mask_path:
            tree_coverage["tree_mask_path"] = self.trees.tree_mask_path

        return {
            "type": "FeatureCollection",
            "features": features,
            "tree_coverage": tree_coverage,
        }

    def summary(self) -> Dict:
        """Get summary statistics of detections."""
        # Count amenities by type
        amenity_counts = {}
        for amenity in self.amenities:
            amenity_type = amenity.amenity_type
            amenity_counts[amenity_type] = amenity_counts.get(amenity_type, 0) + 1

        return {
            "vehicles": len(self.vehicles),
            "swimming_pools": len(self.swimming_pools),
            "amenities": amenity_counts,
            "total_amenities": len(self.amenities),
            "tree_coverage_percent": self.trees.tree_coverage_percent,
        }


class PropertyDetectionService:
    """Unified service for detecting all property features in satellite imagery.

    Detects vehicles, swimming pools, and amenities (tennis courts, basketball
    courts, baseball diamonds, soccer fields) using YOLOv8-OBB model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vehicle_confidence: float = 0.25,
        pool_confidence: float = 0.3,
        amenity_confidence: float = 0.3,
        device: str = "cpu",
        tree_use_docker: bool = True,
        tree_docker_image: str = "parcel-tree-detector",
    ):
        """Initialize property detection service.

        Args:
            model_path: Path to YOLO-OBB model (default: yolov8m-obb.pt)
            vehicle_confidence: Minimum confidence for vehicles (0.0-1.0)
            pool_confidence: Minimum confidence for pools (0.0-1.0)
            amenity_confidence: Minimum confidence for amenities (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            tree_use_docker: Whether to use Docker for tree detection (False for native)
            tree_docker_image: Docker image for tree detection (when tree_use_docker=True)
        """
        # Initialize individual detectors
        self.vehicle_detector = VehicleDetectionService(
            model_path=model_path,
            confidence_threshold=vehicle_confidence,
            device=device,
        )

        self.pool_detector = SwimmingPoolDetectionService(
            model_path=model_path,
            confidence_threshold=pool_confidence,
            device=device,
        )

        self.amenity_detector = AmenityDetectionService(
            model_path=model_path,
            confidence_threshold=amenity_confidence,
            device=device,
        )

        # Tree detection
        self.tree_detector = TreeDetectionService(
            use_docker=tree_use_docker, docker_image=tree_docker_image
        )

    def detect_all(self, satellite_image: Dict) -> PropertyDetections:
        """Detect all property features in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Optional zoom level (default 20)

        Returns:
            PropertyDetections object with vehicles, pools, amenities, and trees
        """
        # Detect all features
        vehicles = self.vehicle_detector.detect_vehicles(satellite_image)
        pools = self.pool_detector.detect_swimming_pools(satellite_image)
        amenities = self.amenity_detector.detect_amenities(satellite_image)
        trees = self.tree_detector.detect_trees(satellite_image)

        return PropertyDetections(
            vehicles=vehicles, swimming_pools=pools, amenities=amenities, trees=trees
        )

    def detect_all_geojson(self, satellite_image: Dict) -> Dict:
        """Detect all property features and return as GeoJSON.

        Args:
            satellite_image: Dict with satellite metadata

        Returns:
            GeoJSON FeatureCollection with all detections
        """
        detections = self.detect_all(satellite_image)
        return detections.to_geojson()
