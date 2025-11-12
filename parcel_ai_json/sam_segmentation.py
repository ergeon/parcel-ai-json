"""SAM (Segment Anything Model) segmentation service for satellite imagery."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image


@dataclass
class SAMSegment:
    """Represents a single SAM segment.

    Attributes:
        segment_id: Unique identifier for this segment
        pixel_mask: Binary mask array (H x W)
        pixel_bbox: Bounding box in pixels (x1, y1, x2, y2)
        geo_polygon: Geographic polygon coordinates [(lon, lat), ...]
        area_pixels: Area in pixels
        area_sqm: Area in square meters
        stability_score: SAM stability score (0-1)
        predicted_iou: Predicted IoU score (0-1)
    """

    segment_id: int
    pixel_mask: np.ndarray
    pixel_bbox: Tuple[int, int, int, int]
    geo_polygon: List[Tuple[float, float]]
    area_pixels: int
    area_sqm: Optional[float] = None
    stability_score: float = 1.0
    predicted_iou: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "segment_id": self.segment_id,
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "area_pixels": self.area_pixels,
            "area_sqm": self.area_sqm,
            "stability_score": self.stability_score,
            "predicted_iou": self.predicted_iou,
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
                "stability_score": self.stability_score,
                "predicted_iou": self.predicted_iou,
                "pixel_bbox": list(self.pixel_bbox),
            },
        }


class SAMSegmentationService:
    """Service for automatic segmentation using SAM (Segment Anything Model).

    Uses Meta's original SAM model for zero-shot segmentation of satellite imagery.
    Compatible with PyTorch >= 1.7 (we're using 2.2.2).
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        model_path: Optional[str] = None,
        device: str = "cpu",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ):
        """Initialize SAM segmentation service.

        Args:
            model_type: SAM model type ('vit_b', 'vit_l', or 'vit_h')
                       Default: 'vit_h' for highest accuracy
            model_path: Path to model checkpoint (if None, auto-detects from models/)
            device: Device to run inference on ('cpu' or 'cuda')
            points_per_side: Number of points per side for automatic mask generation
            pred_iou_thresh: IOU threshold for filtering masks
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area in pixels
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self._sam = None
        self._mask_generator = None

    def _load_model(self):
        """Load SAM model and create mask generator."""
        if self._sam is not None:
            return

        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM segmentation requires segment_anything package. "
                "Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        # Determine checkpoint path
        if self.model_path:
            checkpoint_path = self.model_path
        else:
            # Check models/ directory for checkpoint
            models_dir = Path(__file__).parent.parent / "models"
            checkpoint_files = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth",
            }

            checkpoint_file = checkpoint_files.get(self.model_type)
            if not checkpoint_file:
                raise ValueError(
                    f"Invalid model_type: {self.model_type}. " "Must be one of: vit_b, vit_l, vit_h"
                )

            checkpoint_path = models_dir / checkpoint_file

            if not checkpoint_path.exists():
                download_url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_file}"
                raise FileNotFoundError(
                    f"Model checkpoint not found: {checkpoint_path}\n"
                    f"Download from: {download_url}"
                )

        # Load SAM model
        self._sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
        self._sam.to(device=self.device)

        # Create automatic mask generator
        self._mask_generator = SamAutomaticMaskGenerator(
            self._sam,
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
                - zoom_level: Zoom level (optional, default 20)

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

        # Get image dimensions
        image_height, image_width = image_array.shape[:2]

        # Get center coordinates
        center_lat = satellite_image["center_lat"]
        center_lon = satellite_image["center_lon"]
        zoom_level = satellite_image.get("zoom_level", 20)

        # Create coordinate converter for accurate pixel-to-geo conversion
        from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

        coord_converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=image_width,
            image_height_px=image_height,
            zoom_level=zoom_level,
        )

        # Run automatic mask generation
        masks = self._mask_generator.generate(image_array)

        print(f"SAM generated {len(masks)} segments")

        # Convert masks to SAMSegment objects
        segments = []
        for i, mask_dict in enumerate(masks):
            # Extract mask data
            mask = mask_dict["segmentation"]  # Binary mask (H x W)
            bbox = mask_dict["bbox"]  # [x, y, w, h]
            area = mask_dict["area"]  # Number of pixels
            stability_score = mask_dict.get("stability_score", 1.0)
            predicted_iou = mask_dict.get("predicted_iou", 1.0)

            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox
            pixel_bbox = (int(x), int(y), int(x + w), int(y + h))

            # Extract polygon from mask using coordinate converter
            geo_polygon = self._mask_to_geo_polygon(
                mask,
                coord_converter,
            )

            if not geo_polygon or len(geo_polygon) < 3:
                # Skip invalid polygons
                continue

            # Calculate area in square meters
            area_sqm = self._calculate_area_sqm(geo_polygon)

            segment = SAMSegment(
                segment_id=i,
                pixel_mask=mask,
                pixel_bbox=pixel_bbox,
                geo_polygon=geo_polygon,
                area_pixels=int(area),
                area_sqm=area_sqm,
                stability_score=float(stability_score),
                predicted_iou=float(predicted_iou),
            )
            segments.append(segment)

        print(f"Created {len(segments)} valid SAMSegment objects")
        return segments

    def segment_image_geojson(
        self,
        satellite_image: Dict,
    ) -> Dict:
        """Run segmentation and return GeoJSON FeatureCollection.

        Args:
            satellite_image: Same as segment_image()

        Returns:
            GeoJSON FeatureCollection with all segments
        """
        segments = self.segment_image(satellite_image)

        return {
            "type": "FeatureCollection",
            "features": [seg.to_geojson_feature() for seg in segments],
        }

    def _mask_to_geo_polygon(
        self,
        mask: np.ndarray,
        coord_converter,
    ) -> List[Tuple[float, float]]:
        """Convert binary mask to geographic polygon.

        Uses OpenCV to find contours, then converts pixel coordinates to WGS84
        using ImageCoordinateConverter for accurate geodesic calculations.

        Args:
            mask: Binary mask array (H x W)
            coord_converter: ImageCoordinateConverter instance

        Returns:
            List of (lon, lat) coordinates forming a polygon
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for mask-to-polygon conversion")

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

        if len(contour) < 3:
            return []

        # Simplify contour (reduce points)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        if len(simplified) < 3:
            return []

        # Convert pixel coordinates to geographic using coordinate converter
        geo_polygon = []
        for point in simplified:
            px, py = point[0]
            lon, lat = coord_converter.pixel_to_geo(float(px), float(py))
            geo_polygon.append((lon, lat))

        # Close polygon if not already closed
        if geo_polygon and geo_polygon[0] != geo_polygon[-1]:
            geo_polygon.append(geo_polygon[0])

        return geo_polygon

    def _calculate_area_sqm(self, geo_polygon: List[Tuple[float, float]]) -> float:
        """Calculate polygon area in square meters using geodesic calculations.

        Args:
            geo_polygon: List of (lon, lat) coordinates

        Returns:
            Area in square meters
        """
        from pyproj import Geod

        if len(geo_polygon) < 3:
            return 0.0

        geod = Geod(ellps="WGS84")
        lons, lats = zip(*geo_polygon)

        try:
            area, _ = geod.polygon_area_perimeter(lons, lats)
            return abs(area)
        except Exception:
            return 0.0

    def segment_image_labeled(
        self,
        satellite_image: Dict,
        detections: Dict,
        overlap_threshold: float = 0.5
    ) -> List:
        """Run segmentation and label segments using detection overlap.

        Args:
            satellite_image: Dict with path, center_lat, center_lon, zoom_level
            detections: Dict with detection lists:
                - 'vehicles': List of vehicle detections
                - 'pools': List of pool detections
                - 'trees': List of tree detections
                - 'tree_polygons': List of TreePolygon objects
                - 'amenities': List of amenity detections
            overlap_threshold: IoU threshold for overlap labeling

        Returns:
            List of LabeledSAMSegment objects
        """
        from parcel_ai_json.sam_labeler import SAMSegmentLabeler

        # Run standard segmentation
        sam_segments = self.segment_image(satellite_image)

        # Label segments
        labeler = SAMSegmentLabeler(overlap_threshold=overlap_threshold)
        labeled_segments = labeler.label_segments(sam_segments, detections)

        print(
            f"Labeled {len(labeled_segments)} segments: "
            f"{sum(1 for s in labeled_segments if s.primary_label != 'unknown')} "
            f"with semantic labels"
        )

        return labeled_segments

    def segment_image_labeled_geojson(
        self,
        satellite_image: Dict,
        detections: Dict,
        overlap_threshold: float = 0.5
    ) -> Dict:
        """Run segmentation with labeling and return GeoJSON.

        Args:
            satellite_image: Same as segment_image_labeled()
            detections: Same as segment_image_labeled()
            overlap_threshold: IoU threshold for overlap labeling

        Returns:
            GeoJSON FeatureCollection with labeled segments
        """
        labeled_segments = self.segment_image_labeled(
            satellite_image,
            detections,
            overlap_threshold
        )

        return {
            "type": "FeatureCollection",
            "features": [seg.to_geojson_feature() for seg in labeled_segments],
        }
