# SAM 2 Integration Plan

## Overview

Integrate Meta's Segment Anything Model 2 (SAM 2) into the parcel-ai-json system for automatic image segmentation of satellite imagery. SAM 2 will provide general-purpose segmentation that can be analyzed in relation to our existing semantic detections (vehicles, pools, amenities, trees).

## Goals

### Phase 1: Basic SAM Integration (Current)
1. Add SAM 2 model to the system
2. Create `SAMSegmentationService`
3. Run automatic segmentation on satellite images
4. Return all segments as GeoJSON polygons
5. Test on example satellite images

### Phase 2: Semantic Analysis (Future)
1. Analyze SAM segments for overlap with semantic detections
2. Use GPT-4o/GPT-5 to classify unknown segments
3. Enhance existing detections with SAM refinement
4. Identify new object types not covered by current detectors

## SAM 2 Model Information

### Model Variants
| Model | Parameters | Speed (fps) | Size | Use Case |
|-------|-----------|-------------|------|----------|
| sam2_hiera_tiny | 38.9M | ~47 | Smallest | Fast inference, limited memory |
| sam2_hiera_small | ~84M | ~40 | Small | Balanced |
| sam2_hiera_base_plus | ~150M | ~35 | Medium | Good quality |
| sam2_hiera_large | 224.4M | ~30 | ~900MB | Best quality |

**Recommendation**: Start with `sam2_hiera_small` for balance of speed and quality.

### Requirements
- Python >= 3.10 (we're using 3.12 ✅)
- PyTorch >= 2.5.1
- torchvision >= 0.20.1
- CUDA support (optional, but recommended)

### Model Downloads
- URL pattern: `https://dl.fbaipublicfiles.com/segment_anything_2/072824/{model_name}.pt`
- Models will be downloaded to `models/` directory
- Auto-download on first use (similar to YOLO models)

## Architecture

### Current System
```
PropertyDetectionService
├── VehicleDetectionService (YOLOv8-OBB)
├── SwimmingPoolDetectionService (YOLOv8-OBB)
├── AmenityDetectionService (YOLOv8-OBB)
└── CombinedTreeDetectionService (DeepForest + detectree)
```

### With SAM Integration
```
PropertyDetectionService
├── VehicleDetectionService (YOLOv8-OBB)
├── SwimmingPoolDetectionService (YOLOv8-OBB)
├── AmenityDetectionService (YOLOv8-OBB)
├── CombinedTreeDetectionService (DeepForest + detectree)
└── SAMSegmentationService (NEW) ⭐
    ├── Automatic mask generation
    ├── Polygon extraction from masks
    └── GeoJSON conversion
```

## Implementation Plan

### Step 1: Add Dependencies
**File**: `requirements.txt`

Add:
```txt
# SAM 2 - Segment Anything Model
git+https://github.com/facebookresearch/sam2.git
# OR for more control:
# sam2>=1.0.0  # if available on PyPI
```

### Step 2: Create SAMSegmentationService
**File**: `parcel_ai_json/sam_segmentation.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image

@dataclass
class SAMSegment:
    """Represents a single SAM segment."""
    segment_id: int
    pixel_mask: np.ndarray  # Binary mask
    pixel_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    geo_polygon: List[Tuple[float, float]]  # [(lon, lat), ...]
    area_pixels: int
    area_sqm: Optional[float] = None
    confidence: float = 1.0  # SAM doesn't provide confidence, use 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "segment_id": self.segment_id,
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "area_pixels": self.area_pixels,
            "area_sqm": self.area_sqm,
            "confidence": self.confidence,
        }

    def to_geojson_feature(self) -> Dict:
        """Convert to GeoJSON Feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [self.geo_polygon],
            },
            "properties": {
                "feature_type": "sam_segment",
                "segment_id": self.segment_id,
                "area_pixels": self.area_pixels,
                "area_sqm": self.area_sqm,
                "pixel_bbox": list(self.pixel_bbox),
            },
        }


class SAMSegmentationService:
    """Service for automatic segmentation using SAM 2."""

    def __init__(
        self,
        model_name: str = "sam2_hiera_small",
        device: str = "cpu",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ):
        """Initialize SAM segmentation service.

        Args:
            model_name: SAM2 model variant to use
            device: Device to run inference on ('cpu' or 'cuda')
            points_per_side: Number of points per side for automatic mask generation
            pred_iou_thresh: IOU threshold for filtering masks
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area in pixels
        """
        self.model_name = model_name
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self._model = None
        self._mask_generator = None

    def _load_model(self):
        """Load SAM2 model."""
        if self._model is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM 2 segmentation requires sam2 package. "
                "Install with: pip install git+https://github.com/facebookresearch/sam2.git"
            )

        # Check for model in models/ directory first
        model_file = f"{self.model_name}.pt"
        models_dir = Path(__file__).parent.parent / "models"

        if (models_dir / model_file).exists():
            checkpoint_path = str(models_dir / model_file)
        else:
            # Will auto-download to ~/.cache/sam2/
            checkpoint_path = model_file

        # Build model
        self._model = build_sam2(
            config_file=f"sam2_hiera_{self.model_name.split('_')[-1]}.yaml",
            ckpt_path=checkpoint_path,
            device=self.device,
        )

        # Create automatic mask generator
        self._mask_generator = SAM2AutomaticMaskGenerator(
            self._model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def segment_image(
        self,
        satellite_image: Dict,
    ) -> List[SAMSegment]:
        """Run automatic segmentation on satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Zoom level (optional)

        Returns:
            List of SAMSegment objects
        """
        # Load model if needed
        self._load_model()

        # Load image
        image_path = Path(satellite_image["path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image_array = np.array(img.convert("RGB"))

        # Run automatic mask generation
        masks = self._mask_generator.generate(image_array)

        # Convert masks to SAMSegment objects
        segments = []
        for i, mask_dict in enumerate(masks):
            # Extract mask data
            mask = mask_dict["segmentation"]  # Binary mask
            bbox = mask_dict["bbox"]  # [x, y, w, h]
            area = mask_dict["area"]

            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            pixel_bbox = (x1, y1, x1 + w, y1 + h)

            # Extract polygon from mask (simplified)
            geo_polygon = self._mask_to_geo_polygon(
                mask,
                satellite_image["center_lat"],
                satellite_image["center_lon"],
                image_array.shape[:2],
            )

            # Calculate area in square meters
            area_sqm = self._calculate_area_sqm(geo_polygon)

            segment = SAMSegment(
                segment_id=i,
                pixel_mask=mask,
                pixel_bbox=pixel_bbox,
                geo_polygon=geo_polygon,
                area_pixels=area,
                area_sqm=area_sqm,
            )
            segments.append(segment)

        return segments

    def segment_image_geojson(
        self,
        satellite_image: Dict,
    ) -> Dict:
        """Run segmentation and return GeoJSON FeatureCollection.

        Args:
            satellite_image: Same as segment_image()

        Returns:
            GeoJSON FeatureCollection
        """
        segments = self.segment_image(satellite_image)

        return {
            "type": "FeatureCollection",
            "features": [seg.to_geojson_feature() for seg in segments],
        }

    def _mask_to_geo_polygon(
        self,
        mask: np.ndarray,
        center_lat: float,
        center_lon: float,
        image_shape: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        """Convert binary mask to geographic polygon.

        Uses OpenCV to find contours, then converts to WGS84.
        """
        import cv2
        from pyproj import Geod

        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Convert pixel coordinates to geographic
        geod = Geod(ellps="WGS84")
        image_height, image_width = image_shape

        geo_polygon = []
        for point in simplified:
            px, py = point[0]

            # Calculate offset from center in pixels
            dx_pixels = px - image_width / 2
            dy_pixels = image_height / 2 - py  # Flip Y axis

            # Convert to meters (rough approximation for small areas)
            # TODO: Use proper geodesic calculations
            meters_per_pixel = 0.15  # Approximate for zoom level 20
            dx_meters = dx_pixels * meters_per_pixel
            dy_meters = dy_pixels * meters_per_pixel

            # Calculate new lat/lon
            lon, lat, _ = geod.fwd(center_lon, center_lat, 90, dx_meters)
            lon, lat, _ = geod.fwd(lon, lat, 0, dy_meters)

            geo_polygon.append((lon, lat))

        # Close polygon
        if geo_polygon and geo_polygon[0] != geo_polygon[-1]:
            geo_polygon.append(geo_polygon[0])

        return geo_polygon

    def _calculate_area_sqm(self, geo_polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area in square meters using geodesic calculations."""
        from pyproj import Geod

        if len(geo_polygon) < 3:
            return 0.0

        geod = Geod(ellps="WGS84")
        lons, lats = zip(*geo_polygon)
        area, _ = geod.polygon_area_perimeter(lons, lats)

        return abs(area)
```

### Step 3: Add Tests
**File**: `tests/test_sam_segmentation.py`

Create comprehensive tests for:
- Model loading
- Segmentation on sample images
- Polygon extraction
- GeoJSON conversion

### Step 4: Update Docker Configuration
**File**: `docker/Dockerfile`

Update to include SAM 2:
```dockerfile
# Install SAM 2
RUN pip install git+https://github.com/facebookresearch/sam2.git

# Download SAM 2 model at build time (optional)
RUN python -c "from sam2.build_sam import build_sam2; \
    build_sam2(config_file='sam2_hiera_small.yaml', ckpt_path='sam2_hiera_small.pt')"
```

### Step 5: Add to PropertyDetectionService
**File**: `parcel_ai_json/property_detector.py`

Add optional SAM segmentation:
```python
from .sam_segmentation import SAMSegmentationService, SAMSegment

class PropertyDetections:
    # ... existing fields ...
    sam_segments: List[SAMSegment] = None
```

### Step 6: Create Example Script
**File**: `scripts/test_sam_segmentation.py`

Generate examples showing SAM segments on satellite imagery.

## Testing Strategy

### Unit Tests
1. Model loading and initialization
2. Automatic mask generation
3. Polygon extraction from masks
4. Coordinate transformation accuracy
5. GeoJSON conversion

### Integration Tests
1. Run SAM on actual satellite images
2. Compare segment count across different model sizes
3. Verify memory usage and performance
4. Test Docker container with SAM

### Visual Tests
1. Generate visualization of SAM segments
2. Overlay segments on satellite imagery
3. Compare with semantic detections (vehicles, pools, etc.)

## Performance Considerations

### Memory Usage
- **sam2_hiera_tiny**: ~500MB RAM
- **sam2_hiera_small**: ~1GB RAM
- **sam2_hiera_base_plus**: ~2GB RAM
- **sam2_hiera_large**: ~3GB RAM

### Inference Time (CPU)
- **Estimated**: 10-30 seconds per 512x512 image
- **GPU**: 2-5 seconds (much faster)

### Optimization Options
1. Use smaller model (tiny/small) for faster inference
2. Reduce `points_per_side` (32 → 16 for faster, fewer segments)
3. Increase thresholds to filter out low-quality segments
4. Enable GPU if available
5. Cache results for repeated images

## Future Enhancements (Phase 2)

### Semantic Analysis
1. **Overlap Analysis**: Calculate IoU between SAM segments and semantic detections
2. **Refinement**: Use SAM to refine boundaries of detected objects
3. **Classification**: Use GPT-4o to classify unknown SAM segments
4. **Discovery**: Find new object types not covered by current detectors

### GPT-4o Integration
```python
class SemanticAnalysisService:
    def classify_sam_segment(
        self,
        segment: SAMSegment,
        satellite_image: Image,
        context_detections: Dict,  # Nearby vehicles, pools, etc.
    ) -> str:
        """Use GPT-4o Vision to classify unknown segment."""
        # Crop segment from image
        # Send to GPT-4o with context
        # Return classification
        pass
```

## Deliverables - Phase 1

- [x] Research SAM 2 capabilities and requirements
- [ ] Create `parcel_ai_json/sam_segmentation.py`
- [ ] Add SAM 2 to `requirements.txt`
- [ ] Download and test SAM model locally
- [ ] Create `tests/test_sam_segmentation.py`
- [ ] Update Docker configuration
- [ ] Generate example outputs
- [ ] Update documentation (README.md, ARCHITECTURE.md)

## Timeline

### Session 1 (Current)
- Create plan document ✅
- Download SAM 2 model
- Implement basic `SAMSegmentationService`
- Test on sample satellite image
- Generate visualization

### Session 2
- Add comprehensive tests
- Integrate with PropertyDetectionService
- Update Docker container
- Performance benchmarking

### Session 3+ (Phase 2)
- Implement overlap analysis
- Add GPT-4o classification
- Create semantic enhancement pipeline

## Resources

- **SAM 2 Repository**: https://github.com/facebookresearch/sam2
- **SAM 2 Paper**: Segment Anything 2 (Meta AI Research)
- **segment-geospatial**: Library for SAM on satellite imagery
- **Model Checkpoints**: https://dl.fbaipublicfiles.com/segment_anything_2/072824/

## Notes

- SAM 2 requires Python 3.10+ (we're using 3.12 ✅)
- PyTorch 2.5.1+ required (need to verify our current version)
- Models are large (~900MB for largest), will be git-ignored
- GPU highly recommended but not required
- SAM provides zero-shot segmentation (no training needed)
- Works on any image type (not just aerial/satellite)
