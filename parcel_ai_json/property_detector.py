"""Unified property detection service for satellite imagery.

Detects all property features in a single call:
- Vehicles (cars, trucks, etc.)
- Swimming pools
- Amenities (tennis courts, basketball courts, etc.)
- Fences (HED model with Regrid data)

Uses YOLOv8-OBB model trained on DOTA aerial dataset.
"""

from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from parcel_ai_json.vehicle_detector import VehicleDetectionService, VehicleDetection
from parcel_ai_json.swimming_pool_detector import (
    SwimmingPoolDetectionService,
    SwimmingPoolDetection,
)
from parcel_ai_json.amenity_detector import AmenityDetectionService, AmenityDetection
from parcel_ai_json.tree_detector import TreeDetectionService, TreeDetection
from parcel_ai_json.fence_detector import FenceDetectionService, FenceDetection


@dataclass
class PropertyDetections:
    """Container for all property detections."""

    vehicles: List[VehicleDetection]
    swimming_pools: List[SwimmingPoolDetection]
    amenities: List[AmenityDetection]
    trees: TreeDetection
    fences: Optional[FenceDetection] = None

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

        # Add fence features
        if self.fences:
            for fence_feature in self.fences.to_geojson_features():
                features.append(fence_feature)

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

        # Add fence statistics if available
        if self.fences:
            summary["fence_pixel_count"] = self.fences.fence_pixel_count
            summary["fence_segment_count"] = len(self.fences.geo_polygons)

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
        detectree_use_docker: bool = True,
        fence_threshold: float = 0.1,
    ):
        """Initialize property detection service.

        Args:
            model_path: Path to YOLO-OBB model (default: yolov8m-obb.pt)
            vehicle_confidence: Minimum confidence for vehicles (0.0-1.0)
            pool_confidence: Minimum confidence for pools (0.0-1.0)
            amenity_confidence: Minimum confidence for amenities (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            tree_confidence: Minimum confidence (0.0-1.0, default: 0.1)
            tree_model_name: Hugging Face model name for DeepForest
            detectree_extract_polygons: Extract tree polygons
            detectree_min_tree_area_pixels: Min tree area (pixels)
            detectree_simplify_tolerance_meters: Simplification (meters)
            detectree_use_docker: Run detectree in Docker or natively
            fence_threshold: Probability threshold for fence detection (0.0-1.0)
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

        # Fence detection (HED model)
        self.fence_detector = FenceDetectionService(
            threshold=fence_threshold,
            device=device,
        )

    def detect_all(
        self,
        satellite_image: Dict,
        detect_fences: bool = False,
        regrid_parcel_polygon: Optional[Union[List[Tuple[float, float]], Dict]] = None,
        fence_probability_mask: Optional[np.ndarray] = None,
    ) -> PropertyDetections:
        """Detect all property features in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Optional zoom level (default 20)
            detect_fences: Whether to run fence detection (default: False)
            regrid_parcel_polygon: Optional Regrid parcel polygon to generate
                fence probability mask. Can be either:
                - List of (lon, lat) tuples defining parcel boundary
                - GeoJSON polygon dict with 'coordinates' key
                If provided, takes precedence over fence_probability_mask.
            fence_probability_mask: Optional pre-computed fence probability mask
                Shape: (512, 512), dtype: uint8 (0-255) or float32 (0-1)
                Only used if regrid_parcel_polygon is None.

        Returns:
            PropertyDetections object with vehicles, pools, amenities,
                trees, and optionally fences
        """
        # Detect all features
        vehicles = self.vehicle_detector.detect_vehicles(satellite_image)
        pools = self.pool_detector.detect_swimming_pools(satellite_image)
        amenities = self.amenity_detector.detect_amenities(satellite_image)
        trees = self.tree_detector.detect_trees(satellite_image)

        # Optionally detect fences
        fences = None
        if detect_fences:
            fences = self.fence_detector.detect_fences(
                satellite_image,
                regrid_parcel_polygon=regrid_parcel_polygon,
                fence_probability_mask=fence_probability_mask,
            )

        return PropertyDetections(
            vehicles=vehicles,
            swimming_pools=pools,
            amenities=amenities,
            trees=trees,
            fences=fences,
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
