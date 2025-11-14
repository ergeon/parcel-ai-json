"""SAM segment labeling service for semantic understanding.

Assigns semantic labels to SAM segments using multi-source data fusion:
- Overlap with existing detections (vehicles, pools, trees, amenities)
- OSM building footprints
- Visual features (color, texture)
- Spatial context (neighboring segments)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon


# Label schema with properties for each label type
LABEL_SCHEMA = {
    "vehicle": {
        "color": "#C70039",  # Red
        "priority": 1,  # Highest
        "min_area_sqm": 8,
        "max_area_sqm": 50,
    },
    "pool": {
        "color": "#3498DB",  # Blue
        "priority": 1,
        "min_area_sqm": 10,
        "max_area_sqm": 200,
    },
    "building": {
        "color": "#E74C3C",  # Orange-red
        "priority": 2,
        "min_area_sqm": 20,
        "max_area_sqm": 10000,
    },
    "roof": {
        "color": "#E67E22",  # Orange
        "priority": 2,
        "min_area_sqm": 20,
        "max_area_sqm": 10000,
    },
    "tree": {
        "color": "#27AE60",  # Dark green
        "priority": 3,
        "min_area_sqm": 5,
        "max_area_sqm": 5000,
    },
    "tree_canopy": {
        "color": "#229954",  # Darker green
        "priority": 3,
        "min_area_sqm": 5,
        "max_area_sqm": 5000,
    },
    "vegetation": {
        "color": "#82E0AA",  # Light green
        "priority": 4,
        "min_area_sqm": 5,
        "max_area_sqm": 5000,
    },
    "driveway": {
        "color": "#7F8C8D",  # Gray
        "priority": 5,
        "min_area_sqm": 10,
        "max_area_sqm": 300,
    },
    "pavement": {
        "color": "#95A5A6",  # Light gray
        "priority": 5,
        "min_area_sqm": 2,
        "max_area_sqm": 200,
    },
    "water": {
        "color": "#2E86C1",  # Blue
        "priority": 3,
        "min_area_sqm": 5,
        "max_area_sqm": 10000,
    },
    "ground": {
        "color": "#A0522D",  # Brown
        "priority": 6,
        "min_area_sqm": 5,
        "max_area_sqm": 5000,
    },
    "amenity": {
        "color": "#9B59B6",  # Purple
        "priority": 2,
        "min_area_sqm": 50,
        "max_area_sqm": 1000,
    },
    "unknown": {
        "color": "#BDC3C7",  # Very light gray
        "priority": 10,  # Lowest
        "min_area_sqm": 0,
        "max_area_sqm": 999999,
    },
}


@dataclass
class LabeledSAMSegment:
    """SAM segment with semantic label.

    Attributes:
        segment_id: Unique identifier for this segment
        pixel_mask: Binary mask array (H x W)
        pixel_bbox: Bounding box in pixels (x1, y1, x2, y2)
        geo_polygon: Geographic polygon coordinates [(lon, lat), ...]
        area_pixels: Area in pixels
        area_sqm: Area in square meters
        stability_score: SAM stability score (0-1)
        predicted_iou: Predicted IoU score (0-1)
        primary_label: Semantic label (e.g., 'vehicle', 'pool', 'building')
        label_confidence: Confidence score for label (0-1)
        label_source: Source of label (overlap, osm, visual, context)
        label_subtype: Optional subtype (e.g., 'car', 'house', 'grass')
        secondary_labels: Alternative label interpretations
        related_detections: IDs of related detections
        labeling_reason: Human-readable explanation of labeling
    """

    # Original SAM fields
    segment_id: int
    pixel_mask: np.ndarray
    pixel_bbox: Tuple[int, int, int, int]
    geo_polygon: List[Tuple[float, float]]
    area_pixels: int
    area_sqm: float
    stability_score: float
    predicted_iou: float

    # Labeling fields
    primary_label: str = "unknown"
    label_confidence: float = 0.0
    label_source: str = "none"
    label_subtype: Optional[str] = None

    # Alternative interpretations
    secondary_labels: List[Dict] = field(default_factory=list)

    # Relationships
    related_detections: List[str] = field(default_factory=list)

    # Metadata
    labeling_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excludes pixel_mask)."""
        return {
            "segment_id": self.segment_id,
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "area_pixels": self.area_pixels,
            "area_sqm": self.area_sqm,
            "stability_score": self.stability_score,
            "predicted_iou": self.predicted_iou,
            "primary_label": self.primary_label,
            "label_confidence": self.label_confidence,
            "label_source": self.label_source,
            "label_subtype": self.label_subtype,
            "secondary_labels": self.secondary_labels,
            "related_detections": self.related_detections,
            "labeling_reason": self.labeling_reason,
        }

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON Feature with label properties."""
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [self.geo_polygon]},
            "properties": {
                "feature_type": "labeled_sam_segment",
                "segment_id": self.segment_id,
                "primary_label": self.primary_label,
                "label_confidence": round(self.label_confidence, 3),
                "label_source": self.label_source,
                "label_subtype": self.label_subtype,
                "area_sqm": round(self.area_sqm, 2),
                "area_pixels": self.area_pixels,
                "stability_score": round(self.stability_score, 3),
                "predicted_iou": round(self.predicted_iou, 3),
                "color": LABEL_SCHEMA.get(self.primary_label, LABEL_SCHEMA["unknown"])[
                    "color"
                ],
                "related_detections": self.related_detections,
                "labeling_reason": self.labeling_reason,
            },
        }


class SAMSegmentLabeler:
    """Assigns semantic labels to SAM segments using multi-source fusion.

    Phase 1 implementation: Overlap-based labeling with existing detections
    Phase 2 implementation: OSM buildings and roads integration
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        containment_threshold: float = 0.7,
        osm_overlap_threshold: float = 0.5,
        use_osm: bool = True,
    ):
        """Initialize SAM segment labeler.

        Args:
            overlap_threshold: IoU threshold for overlap-based labeling (default: 0.3)
            containment_threshold: Threshold for containment-based labeling
            osm_overlap_threshold: IoU threshold for OSM feature overlap (default: 0.5)
            use_osm: Whether to fetch and use OSM data (default: True)
        """
        self.overlap_threshold = overlap_threshold
        self.containment_threshold = containment_threshold
        self.osm_overlap_threshold = osm_overlap_threshold
        self.use_osm = use_osm
        self._last_osm_data = None  # Store last fetched OSM data

    @property
    def last_osm_buildings(self) -> List:
        """Get OSM buildings from last labeling run.

        Returns:
            List of OSMBuilding objects, or empty list if none available
        """
        if self._last_osm_data:
            return self._last_osm_data.get("buildings", [])
        return []

    @property
    def last_osm_roads(self) -> List:
        """Get OSM roads from last labeling run.

        Returns:
            List of OSMRoad objects, or empty list if none available
        """
        if self._last_osm_data:
            return self._last_osm_data.get("roads", [])
        return []

    def label_segments(
        self,
        sam_segments: List,
        detections: Dict,
        satellite_image: Optional[Dict] = None,
    ) -> List[LabeledSAMSegment]:
        """Label SAM segments using detection overlap and OSM data.

        Args:
            sam_segments: List of SAMSegment objects
            detections: Dict with keys:
                - 'vehicles': List of vehicle detections
                - 'pools': List of pool detections
                - 'trees': List of tree detections (DeepForest bboxes)
                - 'tree_polygons': List of TreePolygon objects
                - 'amenities': List of amenity detections
            satellite_image: Optional dict with:
                - center_lat: Center latitude
                - center_lon: Center longitude
                - image_width_px: Image width in pixels
                - image_height_px: Image height in pixels
                - zoom_level: Zoom level (default: 20)

        Returns:
            List of LabeledSAMSegment objects
        """
        # Fetch OSM data if enabled and satellite_image provided
        osm_data = None
        if self.use_osm and satellite_image:
            osm_data = self._fetch_osm_data(satellite_image)
            self._last_osm_data = osm_data  # Store for later retrieval

        labeled_segments = []

        for segment in sam_segments:
            # Try overlap-based labeling with existing detections
            label_result = self._label_by_overlap(segment, detections)

            if not label_result and osm_data:
                # Try OSM-based labeling (buildings, roads)
                label_result = self._label_by_osm(segment, osm_data)

            if not label_result:
                # Try containment-based labeling
                label_result = self._label_by_containment(segment, detections)

            if not label_result:
                # No match found - mark as unknown
                label_result = {
                    "label": "unknown",
                    "confidence": 0.0,
                    "source": "none",
                    "reason": "no_match",
                }

            # Create labeled segment
            labeled_segment = LabeledSAMSegment(
                segment_id=segment.segment_id,
                pixel_mask=segment.pixel_mask,
                pixel_bbox=segment.pixel_bbox,
                geo_polygon=segment.geo_polygon,
                area_pixels=segment.area_pixels,
                area_sqm=segment.area_sqm,
                stability_score=segment.stability_score,
                predicted_iou=segment.predicted_iou,
                primary_label=label_result["label"],
                label_confidence=label_result["confidence"],
                label_source=label_result["source"],
                label_subtype=label_result.get("subtype"),
                labeling_reason=label_result.get("reason"),
                related_detections=label_result.get("related_ids", []),
            )

            labeled_segments.append(labeled_segment)

        return labeled_segments

    def _fetch_osm_data(self, satellite_image: Dict) -> Optional[Dict]:
        """Fetch OSM buildings and roads for the image area.

        Args:
            satellite_image: Dict with image metadata

        Returns:
            Dict with 'buildings' and 'roads' lists, or None if fetch fails
        """
        try:
            from parcel_ai_json.osm_data_fetcher import OSMDataFetcher

            fetcher = OSMDataFetcher()
            osm_data = fetcher.fetch_buildings_and_roads(
                center_lat=satellite_image["center_lat"],
                center_lon=satellite_image["center_lon"],
                image_width_px=satellite_image.get("image_width_px", 640),
                image_height_px=satellite_image.get("image_height_px", 640),
                zoom_level=satellite_image.get("zoom_level", 20),
            )
            return osm_data
        except Exception as e:
            print(f"Warning: OSM data fetch failed: {e}")
            return None

    def _label_by_overlap(self, segment, detections: Dict) -> Optional[Dict]:
        """Label segment based on IoU overlap with detections.

        Args:
            segment: SAMSegment object
            detections: Dict of detection lists

        Returns:
            Dict with label info, or None if no match
        """
        segment_poly = Polygon(segment.geo_polygon)

        best_match = None
        best_iou = 0.0

        # Check vehicles
        for i, vehicle in enumerate(detections.get("vehicles", [])):
            iou = self._calculate_iou(segment_poly, self._detection_to_polygon(vehicle))
            if iou > best_iou:
                best_iou = iou
                best_match = {
                    "label": "vehicle",
                    "confidence": iou,
                    "source": "overlap",
                    "reason": f"overlap_iou_{iou:.2f}",
                    "related_ids": [f"vehicle_{i}"],
                }

        # Check pools
        for i, pool in enumerate(detections.get("pools", [])):
            iou = self._calculate_iou(segment_poly, self._detection_to_polygon(pool))
            if iou > best_iou:
                best_iou = iou
                best_match = {
                    "label": "pool",
                    "confidence": iou,
                    "source": "overlap",
                    "reason": f"overlap_iou_{iou:.2f}",
                    "related_ids": [f"pool_{i}"],
                }

        # Check amenities
        for i, amenity in enumerate(detections.get("amenities", [])):
            iou = self._calculate_iou(segment_poly, self._detection_to_polygon(amenity))
            if iou > best_iou:
                best_iou = iou
                best_match = {
                    "label": "amenity",
                    "confidence": iou,
                    "source": "overlap",
                    "subtype": amenity.amenity_type,
                    "reason": f"overlap_iou_{iou:.2f}",
                    "related_ids": [f"amenity_{i}"],
                }

        # Check tree polygons (from detectree)
        for i, tree_poly in enumerate(detections.get("tree_polygons", [])):
            tree_shapely = Polygon(tree_poly.geo_polygon)
            iou = self._calculate_iou(segment_poly, tree_shapely)
            if iou > best_iou:
                best_iou = iou
                best_match = {
                    "label": "tree",
                    "confidence": iou,
                    "source": "overlap",
                    "reason": f"overlap_tree_polygon_iou_{iou:.2f}",
                    "related_ids": [f"tree_polygon_{i}"],
                }

        # Return best match if above threshold
        if best_iou >= self.overlap_threshold:
            return best_match

        return None

    def _label_by_containment(self, segment, detections: Dict) -> Optional[Dict]:
        """Label segment based on containment relationships.

        Args:
            segment: SAMSegment object
            detections: Dict of detection lists

        Returns:
            Dict with label info, or None if no match
        """
        segment_poly = Polygon(segment.geo_polygon)
        segment_centroid = segment_poly.centroid

        # Check if segment contains a vehicle (likely driveway)
        for i, vehicle in enumerate(detections.get("vehicles", [])):
            vehicle_poly = self._detection_to_polygon(vehicle)
            vehicle_center = vehicle_poly.centroid

            if segment_poly.contains(vehicle_center):
                # Segment must be significantly larger to be driveway
                # Calculate vehicle area using pyproj for accurate comparison
                vehicle_area_sqm = self._calculate_polygon_area_sqm(vehicle_poly)
                if (
                    segment.area_sqm
                    and vehicle_area_sqm
                    and segment.area_sqm > vehicle_area_sqm * 2
                ):
                    return {
                        "label": "driveway",
                        "confidence": 0.70,
                        "source": "containment",
                        "reason": "contains_vehicle",
                        "related_ids": [f"vehicle_{i}"],
                    }

        # Check if segment is inside tree coverage
        for i, tree_poly in enumerate(detections.get("tree_polygons", [])):
            tree_shapely = Polygon(tree_poly.geo_polygon)
            if tree_shapely.contains(segment_centroid):
                return {
                    "label": "tree_canopy",
                    "confidence": 0.75,
                    "source": "containment",
                    "reason": "inside_tree_coverage",
                    "related_ids": [f"tree_polygon_{i}"],
                }

        # Check if segment contains pool (likely pool deck)
        for i, pool in enumerate(detections.get("pools", [])):
            pool_poly = self._detection_to_polygon(pool)
            pool_center = pool_poly.centroid

            if segment_poly.contains(pool_center):
                # Check if segment is slightly larger
                pool_area_sqm = self._calculate_polygon_area_sqm(pool_poly)
                if (
                    segment.area_sqm
                    and pool_area_sqm
                    and segment.area_sqm > pool_area_sqm * 1.2
                ):
                    return {
                        "label": "pavement",
                        "subtype": "pool_deck",
                        "confidence": 0.65,
                        "source": "containment",
                        "reason": "contains_pool",
                        "related_ids": [f"pool_{i}"],
                    }

        return None

    def _calculate_iou(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate Intersection over Union for two polygons.

        Args:
            poly1: First polygon (Shapely)
            poly2: Second polygon (Shapely)

        Returns:
            IoU score (0-1)
        """
        try:
            if not poly1.is_valid or not poly2.is_valid:
                return 0.0

            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area

            if union == 0:
                return 0.0

            return intersection / union

        except Exception:
            return 0.0

    def _detection_to_polygon(self, detection) -> Polygon:
        """Convert detection object to Shapely Polygon.

        Handles both oriented bounding boxes and polygons.

        Args:
            detection: Detection object with geo_polygon attribute

        Returns:
            Shapely Polygon
        """
        try:
            return Polygon(detection.geo_polygon)
        except Exception:
            # If conversion fails, return empty polygon
            return Polygon()

    def _calculate_polygon_area_sqm(self, polygon: Polygon) -> float:
        """Calculate polygon area in square meters using geodesic calculations.

        Args:
            polygon: Shapely Polygon in WGS84 coordinates

        Returns:
            Area in square meters
        """
        from pyproj import Geod

        if not polygon.is_valid or polygon.is_empty:
            return 0.0

        geod = Geod(ellps="WGS84")
        exterior_coords = list(polygon.exterior.coords)
        lons, lats = zip(*exterior_coords)

        try:
            area, _ = geod.polygon_area_perimeter(lons, lats)
            return abs(area)
        except Exception:
            return 0.0

    def _label_by_osm(self, segment, osm_data: Dict) -> Optional[Dict]:
        """Label segment based on overlap with OSM features.

        Args:
            segment: SAMSegment object
            osm_data: Dict with 'buildings' and 'roads' from OSMDataFetcher

        Returns:
            Dict with label info, or None if no match
        """
        from shapely.geometry import LineString

        segment_poly = Polygon(segment.geo_polygon)
        best_match = None
        best_iou = 0.0

        # Check OSM buildings first (higher priority)
        for building in osm_data.get("buildings", []):
            building_poly = building.to_shapely()
            iou = self._calculate_iou(segment_poly, building_poly)

            if iou > best_iou and iou >= self.osm_overlap_threshold:
                best_iou = iou
                best_match = {
                    "label": "building",
                    "confidence": min(iou + 0.1, 1.0),  # Boost confidence for OSM data
                    "source": "osm",
                    "subtype": building.building_type,
                    "reason": f"osm_building_iou_{iou:.2f}",
                    "related_ids": [f"osm_building_{building.osm_id}"],
                }

        # Check OSM roads/driveways
        for road in osm_data.get("roads", []):
            # Create buffer around road linestring for overlap check
            road_line = LineString(road.geo_linestring)

            # Estimate buffer width based on highway type
            buffer_widths = {
                "residential": 8.0,  # ~8m
                "service": 5.0,  # ~5m
                "driveway": 3.0,  # ~3m
                "footway": 1.5,  # ~1.5m
                "path": 1.5,
            }
            # Use road width if available, otherwise estimate from highway type
            if road.width_m:
                buffer_m = road.width_m / 2
            else:
                buffer_m = buffer_widths.get(road.highway_type, 4.0) / 2

            # Convert buffer meters to degrees (approximate at this latitude)
            # 1 degree latitude ≈ 111km, adjust for longitude by latitude
            import math

            if segment_poly.centroid:
                lat_rad = math.radians(segment_poly.centroid.y)
                buffer_deg_lat = buffer_m / 111000
                buffer_deg_lon = buffer_m / (111000 * math.cos(lat_rad))
                buffer_deg = (buffer_deg_lat + buffer_deg_lon) / 2
            else:
                buffer_deg = buffer_m / 111000

            road_poly = road_line.buffer(buffer_deg)
            iou = self._calculate_iou(segment_poly, road_poly)

            # Lower threshold for roads (they're linear features)
            road_threshold = max(self.osm_overlap_threshold * 0.6, 0.2)

            if iou > best_iou and iou >= road_threshold:
                best_iou = iou

                # Classify as driveway or pavement based on highway type
                if road.highway_type in ["driveway", "service"]:
                    label = "driveway"
                elif road.highway_type in ["footway", "path", "pedestrian"]:
                    label = "pavement"
                else:
                    label = "pavement"  # Default for other road types

                best_match = {
                    "label": label,
                    "confidence": min(
                        iou + 0.05, 0.85
                    ),  # Slightly lower confidence for roads
                    "source": "osm",
                    "subtype": road.highway_type,
                    "reason": f"osm_road_{road.highway_type}_iou_{iou:.2f}",
                    "related_ids": [f"osm_road_{road.osm_id}"],
                }

        return best_match if best_match else None
