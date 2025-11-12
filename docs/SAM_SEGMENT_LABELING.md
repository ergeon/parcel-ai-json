# SAM Segment Labeling Strategy

## Executive Summary

This document outlines a comprehensive strategy for assigning semantic labels to SAM (Segment Anything Model) segments using multi-source data fusion and heuristic reasoning.

**Goal**: Transform generic "Segment #1, #2, #3" into meaningful labels like "vehicle", "pool", "tree", "building", "driveway", etc.

## Current State

### What We Have
- **SAM segments**: Generic polygons with no semantic meaning
- **YOLOv8-OBB detections**: vehicles, pools, amenities (with labels)
- **DeepForest detections**: tree bounding boxes
- **detectree polygons**: tree coverage areas
- **Potential**: OSM building footprints, road data

### The Problem
SAM produces visually coherent segments but lacks semantic understanding:
```json
{
  "segment_id": 42,
  "label": "sam_segment",  // ❌ Not useful
  "area_sqm": 25.3,
  "stability_score": 0.95
}
```

We want:
```json
{
  "segment_id": 42,
  "primary_label": "vehicle",  // ✅ Meaningful!
  "label_confidence": 0.87,
  "label_source": "overlap",
  "secondary_labels": ["driveway"],
  "area_sqm": 25.3
}
```

## Best Practices in Semantic Segmentation

### Industry Standards

1. **Overlap-based Labeling (IoU)**
   - Calculate Intersection over Union with known detections
   - Threshold typically 0.5-0.7 for "strong" match
   - Most reliable method when ground truth exists

2. **Multi-source Fusion**
   - Combine multiple labeling strategies
   - Weighted confidence scoring
   - Conflict resolution hierarchy

3. **Spatial Reasoning**
   - Use geometric relationships (containment, adjacency)
   - Contextual inference from neighbors
   - Topological analysis

4. **Feature-based Classification**
   - Color/texture analysis (RGB, HSV)
   - Shape features (compactness, elongation)
   - Size constraints (area filtering)

5. **External Knowledge Integration**
   - OpenStreetMap (buildings, roads, landuse)
   - Cadastral data (parcel boundaries)
   - Elevation models (terrain)

## Available Data Sources

### 1. Existing Detections (High Confidence)

**YOLOv8-OBB Detections**:
- Vehicles (cars, trucks)
- Swimming pools
- Amenities (tennis courts, basketball courts, etc.)
- Confidence scores: 0.5-0.99
- Oriented bounding boxes → convert to polygons

**DeepForest/detectree**:
- Tree detections (bounding boxes)
- Tree coverage polygons
- Already geo-referenced

### 2. OpenStreetMap Data (Medium Confidence)

Can fetch via Overpass API:
```python
import overpy
api = overpy.Overpass()

# Get buildings in bounding box
query = f"""
[out:json];
(
  way["building"]({bbox});
  relation["building"]({bbox});
);
out geom;
"""
result = api.query(query)
```

Available OSM features:
- `building=*` (house, garage, shed, commercial, etc.)
- `highway=*` (road types)
- `landuse=*` (residential, commercial, forest, etc.)
- `natural=*` (water, tree, etc.)

### 3. Visual Features (Low-Medium Confidence)

Extract from image pixels within segment mask:
- **Color analysis**: RGB → HSV conversion
  - Green vegetation: H=60-180°, S>30%, V>20%
  - Blue water: H=180-240°, S>30%
  - Gray pavement: S<20%, V=20-80%
- **Texture**: Standard deviation, entropy
- **Shape**: Aspect ratio, compactness, orientation

### 4. Spatial Context (Low-Medium Confidence)

- Adjacent segments (shared boundaries)
- Containment relationships (A inside B)
- Proximity to labeled features
- Distance to parcel boundaries

## Labeling Strategies (Heuristics)

### Strategy 1: IoU-based Overlap (Highest Priority)

**Algorithm**:
```python
def label_by_overlap(sam_segment, all_detections):
    """Label segment based on IoU with known detections."""

    best_match = None
    best_iou = 0.0

    # Check overlap with all detection types
    for detection_type in ['vehicle', 'pool', 'tree', 'amenity']:
        for detection in all_detections[detection_type]:
            iou = calculate_iou(
                sam_segment.geo_polygon,
                detection.geo_polygon
            )

            if iou > best_iou:
                best_iou = iou
                best_match = {
                    'label': detection_type,
                    'confidence': iou,
                    'detection_id': detection.id
                }

    # Strong overlap threshold
    if best_iou >= 0.5:
        return best_match

    return None
```

**Confidence**: 0.5 → 1.0 (scales with IoU)

**Expected coverage**: 30-40% of segments

**Examples**:
- SAM segment overlaps 0.82 IoU with vehicle → label "vehicle" (conf=0.82)
- SAM segment overlaps 0.67 IoU with pool → label "pool" (conf=0.67)
- SAM segment overlaps 0.91 IoU with tree polygon → label "tree" (conf=0.91)

### Strategy 2: Containment Analysis (High Priority)

**Algorithm**:
```python
def label_by_containment(sam_segment, all_detections):
    """Infer label from containment relationships."""

    # Case 1: Segment contains a detection
    # → Segment is likely the surface/context
    for vehicle in all_detections['vehicles']:
        if polygon_contains(sam_segment.geo_polygon, vehicle.center):
            if sam_segment.area_sqm > vehicle.area_sqm * 2:
                return {
                    'label': 'driveway',
                    'confidence': 0.70,
                    'reason': 'contains_vehicle',
                    'related_id': vehicle.id
                }

    for pool in all_detections['pools']:
        if polygon_contains(sam_segment.geo_polygon, pool.center):
            # Check if segment is slightly larger (deck/patio)
            if sam_segment.area_sqm > pool.area_sqm * 1.2:
                return {
                    'label': 'pool_deck',
                    'confidence': 0.65,
                    'reason': 'contains_pool'
                }

    # Case 2: Segment is contained by detection
    # → Segment is part of the detection
    for tree_polygon in all_detections['tree_polygons']:
        if polygon_contains(tree_polygon, sam_segment.centroid):
            return {
                'label': 'tree_canopy',
                'confidence': 0.75,
                'reason': 'inside_tree_coverage'
            }

    return None
```

**Confidence**: 0.65-0.75

**Expected coverage**: 10-15% of segments

**Examples**:
- Large SAM segment contains small vehicle → "driveway"
- SAM segment inside tree coverage polygon → "tree_canopy"
- SAM segment around pool → "pool_deck" or "patio"

### Strategy 3: OSM Building Match (High Priority)

**Algorithm**:
```python
def label_by_osm_buildings(sam_segment, osm_buildings):
    """Match segment to OpenStreetMap building footprints."""

    for building in osm_buildings:
        iou = calculate_iou(
            sam_segment.geo_polygon,
            building.geometry
        )

        if iou >= 0.6:  # Strong building match
            return {
                'label': 'building',
                'subtype': building.tags.get('building', 'yes'),
                'confidence': min(iou, 0.90),  # Cap at 0.90
                'source': 'osm',
                'osm_id': building.id
            }

        # Check if segment is building roof (slightly smaller)
        if iou >= 0.4 and iou < 0.6:
            area_ratio = sam_segment.area_sqm / building.area_sqm
            if 0.8 <= area_ratio <= 1.0:
                return {
                    'label': 'roof',
                    'building_type': building.tags.get('building'),
                    'confidence': 0.70,
                    'source': 'osm'
                }

    return None
```

**Confidence**: 0.60-0.90

**Expected coverage**: 20-30% of segments (in urban/suburban areas)

**Implementation**:
```python
import overpy

def fetch_osm_buildings(bbox):
    """Fetch OSM buildings in bounding box.

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon)
    """
    api = overpy.Overpass()

    query = f"""
    [out:json];
    (
      way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out geom;
    """

    result = api.query(query)

    buildings = []
    for way in result.ways:
        polygon = [(float(node.lon), float(node.lat))
                   for node in way.nodes]
        buildings.append({
            'id': way.id,
            'geometry': polygon,
            'tags': way.tags,
            'area_sqm': calculate_area(polygon)
        })

    return buildings
```

### Strategy 4: Visual Features (Medium Priority)

**Algorithm**:
```python
def label_by_visual_features(sam_segment, image):
    """Classify segment based on color/texture."""

    # Extract pixels from segment mask
    pixels = extract_masked_pixels(image, sam_segment.pixel_mask)

    # Color analysis
    mean_rgb = np.mean(pixels, axis=0)
    hsv = rgb_to_hsv(mean_rgb)
    h, s, v = hsv[0], hsv[1], hsv[2]

    # Green vegetation (grass, shrubs, lawns)
    if 60 <= h <= 180 and s > 0.3 and v > 0.2:
        # Distinguish from trees (already labeled by overlap)
        if sam_segment.area_sqm < 500:  # Smaller vegetation
            return {
                'label': 'vegetation',
                'subtype': 'lawn',
                'confidence': 0.60,
                'reason': 'green_color',
                'hsv': hsv.tolist()
            }

    # Blue water (pools, ponds)
    if 180 <= h <= 240 and s > 0.3:
        return {
            'label': 'water',
            'confidence': 0.65,
            'reason': 'blue_color'
        }

    # Gray/white pavement (driveways, walkways)
    if s < 0.2 and 0.3 <= v <= 0.8:
        if sam_segment.area_sqm < 200:
            return {
                'label': 'pavement',
                'subtype': 'walkway',
                'confidence': 0.55,
                'reason': 'gray_color'
            }
        else:
            return {
                'label': 'pavement',
                'subtype': 'driveway',
                'confidence': 0.55,
                'reason': 'gray_color'
            }

    # Brown/tan bare earth
    if 10 <= h <= 40 and s > 0.2 and v < 0.6:
        return {
            'label': 'ground',
            'subtype': 'bare_earth',
            'confidence': 0.50,
            'reason': 'brown_color'
        }

    return None
```

**Confidence**: 0.50-0.65

**Expected coverage**: 20-30% of remaining segments

**Challenges**:
- Lighting variations (shadows, time of day)
- Image compression artifacts
- Seasonal changes (brown lawns, autumn leaves)

### Strategy 5: Spatial Context (Lowest Priority)

**Algorithm**:
```python
def label_by_spatial_context(sam_segment, labeled_neighbors):
    """Infer label from nearby labeled segments."""

    # Find adjacent segments (shared boundary)
    neighbors = find_adjacent_segments(sam_segment, labeled_neighbors)

    if not neighbors:
        return None

    # Count neighbor label types
    neighbor_labels = [n.primary_label for n in neighbors]
    label_counts = Counter(neighbor_labels)

    # Heuristic 1: Surrounded by same type
    most_common_label = label_counts.most_common(1)[0]
    if most_common_label[1] >= 3:  # 3+ neighbors with same label
        return {
            'label': most_common_label[0],
            'confidence': 0.55,
            'reason': f'surrounded_by_{most_common_label[0]}',
            'neighbor_count': most_common_label[1]
        }

    # Heuristic 2: Between two important features
    if 'vehicle' in neighbor_labels and 'building' in neighbor_labels:
        return {
            'label': 'driveway',
            'confidence': 0.60,
            'reason': 'between_vehicle_and_building'
        }

    if 'pool' in neighbor_labels and 'building' in neighbor_labels:
        return {
            'label': 'patio',
            'confidence': 0.58,
            'reason': 'near_pool_and_building'
        }

    return None
```

**Confidence**: 0.55-0.65

**Expected coverage**: 10-20% of remaining segments

## Label Schema

### Primary Label Categories

```python
LABEL_SCHEMA = {
    'vehicle': {
        'subtypes': ['car', 'truck', 'motorcycle', 'trailer'],
        'color': '#C70039',  # Red
        'priority': 1,  # Highest
        'min_area_sqm': 8,
        'max_area_sqm': 50
    },
    'pool': {
        'subtypes': ['swimming_pool', 'spa', 'hot_tub'],
        'color': '#3498DB',  # Blue
        'priority': 1,
        'min_area_sqm': 10,
        'max_area_sqm': 200
    },
    'building': {
        'subtypes': ['house', 'garage', 'shed', 'commercial'],
        'color': '#E74C3C',  # Orange-red
        'priority': 2,
        'min_area_sqm': 20,
        'max_area_sqm': 10000
    },
    'roof': {
        'subtypes': ['shingle', 'tile', 'metal', 'flat'],
        'color': '#E67E22',  # Orange
        'priority': 2,
        'min_area_sqm': 20,
        'max_area_sqm': 10000
    },
    'tree': {
        'subtypes': ['tree_canopy', 'tree_cluster'],
        'color': '#27AE60',  # Dark green
        'priority': 3,
        'min_area_sqm': 5,
        'max_area_sqm': 5000
    },
    'vegetation': {
        'subtypes': ['lawn', 'grass', 'shrub', 'garden'],
        'color': '#82E0AA',  # Light green
        'priority': 4,
        'min_area_sqm': 5,
        'max_area_sqm': 5000
    },
    'driveway': {
        'subtypes': ['concrete', 'asphalt', 'gravel'],
        'color': '#7F8C8D',  # Gray
        'priority': 5,
        'min_area_sqm': 10,
        'max_area_sqm': 300
    },
    'pavement': {
        'subtypes': ['walkway', 'patio', 'sidewalk'],
        'color': '#95A5A6',  # Light gray
        'priority': 5,
        'min_area_sqm': 2,
        'max_area_sqm': 200
    },
    'water': {
        'subtypes': ['pond', 'lake', 'stream'],
        'color': '#2E86C1',  # Blue
        'priority': 3,
        'min_area_sqm': 5,
        'max_area_sqm': 10000
    },
    'ground': {
        'subtypes': ['bare_earth', 'dirt', 'soil'],
        'color': '#A0522D',  # Brown
        'priority': 6,
        'min_area_sqm': 5,
        'max_area_sqm': 5000
    },
    'amenity': {
        'subtypes': ['tennis_court', 'basketball_court', 'playground'],
        'color': '#9B59B6',  # Purple
        'priority': 2,
        'min_area_sqm': 50,
        'max_area_sqm': 1000
    },
    'unknown': {
        'subtypes': [],
        'color': '#BDC3C7',  # Very light gray
        'priority': 10,  # Lowest
        'min_area_sqm': 0,
        'max_area_sqm': 999999
    }
}
```

### Label Confidence Tiers

- **High confidence (0.75-1.0)**: Overlap with detection, OSM match
- **Medium confidence (0.55-0.74)**: Containment, visual features
- **Low confidence (0.35-0.54)**: Spatial context, weak visual
- **Unknown (<0.35)**: No reliable labeling method

## Conflict Resolution

When multiple strategies produce different labels:

```python
def resolve_label_conflicts(candidate_labels):
    """Pick best label from multiple candidates.

    Args:
        candidate_labels: List of label dicts from different strategies

    Returns:
        Best label dict with updated confidence
    """

    # Priority weights for each source
    SOURCE_WEIGHTS = {
        'overlap': 1.0,      # Highest - direct detection match
        'osm': 0.90,         # Very high - external ground truth
        'containment': 0.80, # High - geometric reasoning
        'visual': 0.60,      # Medium - can be ambiguous
        'context': 0.50      # Lower - inferred from neighbors
    }

    # Score each candidate
    scored = []
    for label in candidate_labels:
        source = label['source']
        confidence = label['confidence']

        # Combined score
        score = confidence * SOURCE_WEIGHTS.get(source, 0.5)

        # Bonus for specific label types (prefer specific over generic)
        if label['label'] in ['vehicle', 'pool', 'building']:
            score *= 1.1

        scored.append((score, label))

    # Sort by score (highest first)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return best label
    best_label = scored[0][1].copy()

    # Store alternatives as secondary labels
    if len(scored) > 1:
        best_label['secondary_labels'] = [s[1] for s in scored[1:3]]

    return best_label
```

## Implementation Architecture

### Enhanced SAMSegment Dataclass

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class LabeledSAMSegment:
    """SAM segment with semantic label."""

    # Original SAM fields
    segment_id: int
    pixel_mask: np.ndarray
    pixel_bbox: Tuple[int, int, int, int]
    geo_polygon: List[Tuple[float, float]]
    area_pixels: int
    area_sqm: float
    stability_score: float
    predicted_iou: float

    # New labeling fields
    primary_label: str = 'unknown'
    label_confidence: float = 0.0
    label_source: str = 'none'  # overlap, osm, visual, context
    label_subtype: Optional[str] = None

    # Alternative interpretations
    secondary_labels: List[Dict] = field(default_factory=list)

    # Relationships to other detections
    related_detections: List[str] = field(default_factory=list)

    # Labeling metadata
    labeling_method: str = 'none'
    labeling_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excludes pixel_mask)."""
        return {
            'segment_id': self.segment_id,
            'pixel_bbox': list(self.pixel_bbox),
            'geo_polygon': self.geo_polygon,
            'area_pixels': self.area_pixels,
            'area_sqm': self.area_sqm,
            'stability_score': self.stability_score,
            'predicted_iou': self.predicted_iou,
            'primary_label': self.primary_label,
            'label_confidence': self.label_confidence,
            'label_source': self.label_source,
            'label_subtype': self.label_subtype,
            'secondary_labels': self.secondary_labels,
            'related_detections': self.related_detections,
            'labeling_reason': self.labeling_reason
        }

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON Feature with label properties."""
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [self.geo_polygon]
            },
            'properties': {
                'feature_type': 'labeled_sam_segment',
                'segment_id': self.segment_id,
                'primary_label': self.primary_label,
                'label_confidence': round(self.label_confidence, 3),
                'label_source': self.label_source,
                'label_subtype': self.label_subtype,
                'area_sqm': round(self.area_sqm, 2),
                'area_pixels': self.area_pixels,
                'stability_score': round(self.stability_score, 3),
                'predicted_iou': round(self.predicted_iou, 3),
                'color': LABEL_SCHEMA[self.primary_label]['color'],
                'related_detections': self.related_detections
            }
        }
```

### SAMSegmentLabeler Service

```python
class SAMSegmentLabeler:
    """Assigns semantic labels to SAM segments."""

    def __init__(
        self,
        use_osm: bool = True,
        use_visual_features: bool = True,
        use_spatial_context: bool = True,
        overlap_threshold: float = 0.5,
        osm_cache_dir: str = '.osm_cache'
    ):
        self.use_osm = use_osm
        self.use_visual_features = use_visual_features
        self.use_spatial_context = use_spatial_context
        self.overlap_threshold = overlap_threshold
        self.osm_cache_dir = osm_cache_dir

        # Cache for OSM data
        self.osm_buildings = {}

    def label_segments(
        self,
        sam_segments: List[SAMSegment],
        detections: Dict,
        image: np.ndarray,
        satellite_image_metadata: Dict
    ) -> List[LabeledSAMSegment]:
        """Label all SAM segments using multi-source fusion.

        Args:
            sam_segments: Raw SAM segments
            detections: Dict with keys 'vehicles', 'pools', 'trees', etc.
            image: Original satellite image (for visual features)
            satellite_image_metadata: Center lat/lon, zoom, etc.

        Returns:
            List of labeled segments
        """

        # Fetch OSM data if enabled
        osm_buildings = []
        if self.use_osm:
            bbox = self._compute_bbox(satellite_image_metadata)
            osm_buildings = self._fetch_osm_buildings(bbox)

        # Label each segment
        labeled_segments = []

        for segment in sam_segments:
            # Try each labeling strategy in priority order
            label_candidates = []

            # Strategy 1: Overlap with detections
            overlap_label = self._label_by_overlap(segment, detections)
            if overlap_label:
                label_candidates.append(overlap_label)

            # Strategy 2: Containment analysis
            containment_label = self._label_by_containment(segment, detections)
            if containment_label:
                label_candidates.append(containment_label)

            # Strategy 3: OSM building match
            if self.use_osm and osm_buildings:
                osm_label = self._label_by_osm(segment, osm_buildings)
                if osm_label:
                    label_candidates.append(osm_label)

            # Strategy 4: Visual features
            if self.use_visual_features:
                visual_label = self._label_by_visual(segment, image)
                if visual_label:
                    label_candidates.append(visual_label)

            # Resolve conflicts if multiple candidates
            if label_candidates:
                best_label = self._resolve_conflicts(label_candidates)
            else:
                best_label = {
                    'label': 'unknown',
                    'confidence': 0.0,
                    'source': 'none'
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
                primary_label=best_label['label'],
                label_confidence=best_label['confidence'],
                label_source=best_label['source'],
                label_subtype=best_label.get('subtype'),
                labeling_reason=best_label.get('reason')
            )

            labeled_segments.append(labeled_segment)

        # Strategy 5: Spatial context (second pass)
        if self.use_spatial_context:
            labeled_segments = self._apply_spatial_context(labeled_segments)

        return labeled_segments

    # ... implementation of _label_by_overlap, etc.
```

## Implementation Phases

### Phase 1: Overlap-based Labeling (Week 1)
**Goal**: Label 30-40% of segments using detection overlap

**Tasks**:
1. Implement IoU calculation (Shapely)
2. Create `_label_by_overlap()` method
3. Add `LabeledSAMSegment` dataclass
4. Update GeoJSON output with labels
5. Test with existing detections
6. Create visualization script (color-coded by label)

**Success Criteria**:
- 30%+ of segments labeled
- Confidence scores > 0.7 for overlap matches
- Zero errors in IoU calculation

### Phase 2: OSM Integration (Week 2)
**Goal**: Label buildings and structures

**Tasks**:
1. Add `overpy` dependency (OSM Overpass API)
2. Implement `_fetch_osm_buildings()` with caching
3. Create `_label_by_osm()` method
4. Handle OSM building types (house, garage, shed)
5. Test with urban/suburban properties
6. Add OSM attribution to outputs

**Success Criteria**:
- 50%+ of segments labeled (cumulative)
- Building detection accuracy > 80%
- OSM API errors handled gracefully

### Phase 3: Visual Features (Week 3)
**Goal**: Classify vegetation, pavement, water

**Tasks**:
1. Implement RGB → HSV conversion
2. Create color-based classification rules
3. Add texture analysis (optional)
4. Implement `_label_by_visual()` method
5. Test across different lighting conditions
6. Tune color thresholds

**Success Criteria**:
- 70%+ of segments labeled (cumulative)
- Vegetation detection accuracy > 70%
- Handles shadows and lighting variations

### Phase 4: Spatial Context (Week 4)
**Goal**: Label remaining segments using neighbors

**Tasks**:
1. Implement adjacency detection (shared boundaries)
2. Create `_label_by_spatial_context()` method
3. Apply contextual reasoning rules
4. Multi-pass labeling (iterate until convergence)
5. Test conflict resolution

**Success Criteria**:
- 80%+ of segments labeled (cumulative)
- Spatial reasoning improves coverage by 10-15%
- No circular dependencies

### Phase 5: Validation & Refinement (Week 5)
**Goal**: Validate and improve labeling accuracy

**Tasks**:
1. Manual validation on 20-30 properties
2. Measure precision/recall for each label type
3. Tune confidence thresholds
4. Add label schema validation
5. Create confusion matrix
6. Document known limitations

**Success Criteria**:
- Overall accuracy > 75%
- Vehicle labeling > 90% precision
- Building labeling > 85% precision
- Documentation complete

## Usage Examples

### Example 1: Label SAM Segments

```python
from parcel_ai_json.sam_segmentation import SAMSegmentationService
from parcel_ai_json.sam_labeler import SAMSegmentLabeler
from parcel_ai_json.property_detector import PropertyDetectionService

# Detect all features
property_detector = PropertyDetectionService()
detections = property_detector.detect_from_image(
    image_path="satellite.jpg",
    center_lat=37.7749,
    center_lon=-122.4194,
    zoom_level=20
)

# Run SAM segmentation
sam_service = SAMSegmentationService()
sam_segments = sam_service.segment_image(satellite_image)

# Label segments
labeler = SAMSegmentLabeler(
    use_osm=True,
    use_visual_features=True,
    use_spatial_context=True
)

labeled_segments = labeler.label_segments(
    sam_segments=sam_segments,
    detections={
        'vehicles': detections.vehicles,
        'pools': detections.pools,
        'trees': detections.trees,
        'tree_polygons': detections.tree_polygons,
        'amenities': detections.amenities
    },
    image=load_image("satellite.jpg"),
    satellite_image_metadata={
        'center_lat': 37.7749,
        'center_lon': -122.4194,
        'zoom_level': 20
    }
)

# Filter by label
vehicles = [s for s in labeled_segments if s.primary_label == 'vehicle']
buildings = [s for s in labeled_segments if s.primary_label == 'building']

print(f"Found {len(vehicles)} vehicle segments")
print(f"Found {len(buildings)} building segments")
```

### Example 2: Generate Labeled GeoJSON

```python
# Create GeoJSON with labeled segments
geojson = {
    'type': 'FeatureCollection',
    'features': []
}

for segment in labeled_segments:
    feature = segment.to_geojson_feature()
    geojson['features'].append(feature)

# Save
with open('labeled_segments.geojson', 'w') as f:
    json.dump(geojson, f, indent=2)
```

### Example 3: Visualize with Folium

```python
import folium

# Create map
center_lat = 37.7749
center_lon = -122.4194
m = folium.Map(location=[center_lat, center_lon], zoom_start=20)

# Add labeled segments
for segment in labeled_segments:
    label = segment.primary_label
    color = LABEL_SCHEMA[label]['color']

    folium.Polygon(
        locations=[(lat, lon) for lon, lat in segment.geo_polygon],
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.4,
        popup=f"{label} (conf={segment.label_confidence:.2f})",
        tooltip=f"Segment #{segment.segment_id}: {label}"
    ).add_to(m)

m.save('labeled_segments_map.html')
```

## Benefits

### 1. Semantic Understanding
- Segments become interpretable ("this is a vehicle", not "segment #42")
- Enable high-level queries: "Show me all buildings", "Count trees"
- Better insights for property analysis

### 2. Improved Visualization
- Color-code by semantic label
- Filter by label type
- Layer controls in folium maps

### 3. Data Enrichment
- Cross-reference with detections
- Measure coverage (% vegetation, % pavement)
- Spatial analytics (distance to nearest tree, etc.)

### 4. Integration with Other Systems
- Easy to combine with parcel boundaries
- Export to GIS tools (QGIS, ArcGIS)
- Feed into downstream ML models

### 5. Quality Assurance
- Compare SAM labels to YOLO detections (validation)
- Identify missed detections
- Improve detection models

## Challenges & Limitations

### 1. OSM Data Quality
- Varies by region (better in urban areas)
- May be outdated (buildings demolished, new construction)
- Incomplete coverage in rural areas

**Mitigation**: Use OSM as one of several sources; don't rely exclusively

### 2. Visual Ambiguity
- Gray pavement vs gray roof
- Green vegetation vs green pool cover
- Shadows distort colors

**Mitigation**: Combine with geometric features (shape, size, context)

### 3. Overlapping Features
- Vehicle on driveway (both are correct)
- Tree shadow on lawn
- Pool deck vs patio

**Mitigation**: Support secondary labels; use confidence scores

### 4. Computational Cost
- OSM API calls (rate-limited)
- Visual feature extraction (pixel-level operations)
- IoU calculations (polygon operations)

**Mitigation**: Cache OSM data; batch processing; optimize with rtree spatial index

### 5. Generalization
- Rules tuned for residential properties
- May not work well for commercial/industrial
- Different climates/regions need different thresholds

**Mitigation**: Make thresholds configurable; document assumptions

## Future Enhancements

### Short-term (1-2 months)
- [ ] Add more label types (fence, deck, carport)
- [ ] Support multi-label segments (primary + secondary)
- [ ] Improve visual feature classification
- [ ] Add elevation data (DEM) integration

### Medium-term (3-6 months)
- [ ] Train ML classifier on labeled SAM segments
- [ ] Active learning (user feedback loop)
- [ ] Temporal analysis (change detection over time)
- [ ] Integration with parcel boundaries

### Long-term (6-12 months)
- [ ] 3D reconstruction from satellite + elevation
- [ ] Semantic scene graph (relationships between objects)
- [ ] Natural language queries ("find properties with pools and large lawns")
- [ ] Automated property valuation features

## References

### Academic Papers
1. Kirillov et al. (2023) - "Segment Anything" (SAM paper)
2. He et al. (2017) - "Mask R-CNN" (instance segmentation)
3. Zhao et al. (2017) - "Pyramid Scene Parsing Network" (semantic segmentation)

### Industry Best Practices
- Mapbox: Satellite imagery analysis
- Planet Labs: Object detection in satellite imagery
- Descartes Labs: Geospatial ML platform

### Tools & Libraries
- Shapely: Polygon operations (IoU, containment)
- overpy: OpenStreetMap Overpass API client
- Rasterio: Geospatial raster operations
- GeoPandas: Geospatial data manipulation

## Conclusion

Labeling SAM segments transforms generic image segmentation into actionable semantic understanding. By fusing multiple data sources (detections, OSM, visual features, spatial context), we can achieve 70-80% labeling coverage with reasonable confidence.

The phased implementation approach allows incremental validation and refinement, ensuring high quality results while managing complexity.

**Next Steps**: Begin Phase 1 implementation with overlap-based labeling.
