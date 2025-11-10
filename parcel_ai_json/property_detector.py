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

        # Add tree bounding box features (DeepForest)
        for tree in self.trees.trees:
            features.append(tree.to_geojson_feature())

        # Add tree coverage polygon features (detectree)
        if self.trees.tree_polygons:
            for polygon in self.trees.tree_polygons:
                features.append(polygon.to_geojson_feature())

        # Add tree detection metadata
        tree_metadata = {
            "tree_count": self.trees.tree_count,
        }

        # Include DeepForest statistics if available
        if self.trees.average_confidence is not None:
            tree_metadata["average_confidence"] = round(
                self.trees.average_confidence, 3
            )
        if self.trees.average_crown_area_sqm is not None:
            tree_metadata["average_crown_area_sqm"] = round(
                self.trees.average_crown_area_sqm, 2
            )

        # Include detectree statistics if available
        if (
            hasattr(self.trees, "tree_coverage_percent")
            and self.trees.tree_coverage_percent is not None
        ):
            tree_metadata["tree_coverage_percent"] = round(
                self.trees.tree_coverage_percent, 2
            )
        if hasattr(self.trees, "tree_pixel_count"):
            tree_metadata["tree_pixel_count"] = self.trees.tree_pixel_count

        return {
            "type": "FeatureCollection",
            "features": features,
            "trees": tree_metadata,
        }

    def summary(self) -> Dict:
        """Get summary statistics of detections."""
        # Count amenities by type
        amenity_counts = {}
        for amenity in self.amenities:
            amenity_type = amenity.amenity_type
            amenity_counts[amenity_type] = amenity_counts.get(amenity_type, 0) + 1

        summary = {
            "vehicles": len(self.vehicles),
            "swimming_pools": len(self.swimming_pools),
            "amenities": amenity_counts,
            "total_amenities": len(self.amenities),
            "tree_count": self.trees.tree_count,
        }

        # Add DeepForest tree statistics if available
        if self.trees.average_confidence is not None:
            summary["average_tree_confidence"] = round(self.trees.average_confidence, 3)
        if self.trees.average_crown_area_sqm is not None:
            summary["average_crown_area_sqm"] = round(
                self.trees.average_crown_area_sqm, 2
            )

        # Add detectree tree statistics if available
        if (
            hasattr(self.trees, "tree_coverage_percent")
            and self.trees.tree_coverage_percent is not None
        ):
            summary["tree_coverage_percent"] = round(
                self.trees.tree_coverage_percent, 2
            )
        if hasattr(self.trees, "tree_polygons") and self.trees.tree_polygons:
            summary["tree_polygon_count"] = len(self.trees.tree_polygons)

        return summary


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
        tree_confidence: float = 0.1,
        tree_model_name: str = "weecology/deepforest-tree",
        detectree_extract_polygons: bool = True,
        detectree_min_tree_area_pixels: int = 50,
        detectree_simplify_tolerance_meters: float = 0.5,
        detectree_use_docker: bool = False,
    ):
        """Initialize property detection service.

        Args:
            model_path: Path to YOLO-OBB model (default: yolov8m-obb.pt)
            vehicle_confidence: Minimum confidence for vehicles (0.0-1.0)
            pool_confidence: Minimum confidence for pools (0.0-1.0)
            amenity_confidence: Minimum confidence for amenities (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            tree_confidence: Minimum confidence for trees (0.0-1.0, default: 0.1)
            tree_model_name: Hugging Face model name for DeepForest
            detectree_extract_polygons: Extract tree cluster polygons from detectree
            detectree_min_tree_area_pixels: Minimum tree area in pixels for detectree
            detectree_simplify_tolerance_meters: Polygon simplification tolerance in meters
            detectree_use_docker: Run detectree in Docker (True) or natively (False)
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

        # Tree detection with DeepForest and detectree (combined service)
        self.tree_detector = TreeDetectionService(
            deepforest_model_name=tree_model_name,
            deepforest_confidence_threshold=tree_confidence,
            detectree_extract_polygons=detectree_extract_polygons,
            detectree_min_tree_area_pixels=detectree_min_tree_area_pixels,
            detectree_simplify_tolerance_meters=detectree_simplify_tolerance_meters,
            detectree_use_docker=detectree_use_docker,
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
