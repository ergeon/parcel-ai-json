"""SAM3 (Segment Anything Model 3) segmentation service for satellite imagery.

SAM3 is Meta's 848M parameter vision foundation model for open-vocabulary
object detection and segmentation. Unlike the original SAM, SAM3 uses natural
language prompts to detect and segment specific objects.

Features:
- Open-vocabulary detection: Use any text prompt (e.g., "houses", "cars", "trees")
- Instance segmentation: Pixel-level masks for each detected object
- Aerial/satellite imagery optimized
- Multi-class detection in a single model

Performance:
- Model loading: ~11-12 seconds (one-time)
- Per-class detection: ~18-23 seconds (CPU), ~4-8 seconds (GPU)
- Image resolution: 1008x1008 (automatically resized)
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Add sam3 to path
SAM3_PATH = Path(__file__).parent.parent / "models" / "sam3"
if str(SAM3_PATH) not in sys.path:
    sys.path.insert(0, str(SAM3_PATH))

# Import coordinate converter at module level for easier mocking
from parcel_ai_json.coordinate_converter import ImageCoordinateConverter  # noqa: E402


@dataclass
class SAM3Detection:
    """Represents a single SAM3 detection.

    Attributes:
        detection_id: Unique identifier for this detection
        class_name: Text prompt used for detection (e.g., "house", "car")
        confidence: Detection confidence score (0-1)
        pixel_mask: Binary mask array (H x W)
        pixel_bbox: Bounding box in pixels (x1, y1, x2, y2)
        geo_polygon: Geographic polygon coordinates [(lon, lat), ...]
        area_pixels: Area in pixels
        area_sqm: Area in square meters
    """

    detection_id: int
    class_name: str
    confidence: float
    pixel_mask: np.ndarray
    pixel_bbox: Tuple[int, int, int, int]
    geo_polygon: List[Tuple[float, float]]
    area_pixels: int
    area_sqm: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "detection_id": self.detection_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "pixel_bbox": list(self.pixel_bbox),
            "geo_polygon": self.geo_polygon,
            "area_pixels": self.area_pixels,
            "area_sqm": self.area_sqm,
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
                "feature_type": "sam3_detection",
                "detection_id": self.detection_id,
                "class_name": self.class_name,
                "confidence": self.confidence,
                "area_pixels": self.area_pixels,
                "area_sqm": self.area_sqm,
                "pixel_bbox": list(self.pixel_bbox),
            },
        }


class SAM3SegmentationService:
    """Service for open-vocabulary segmentation using SAM3.

    Uses Meta's SAM3 model for text-prompted detection and segmentation
    of satellite imagery. Supports natural language prompts like "houses",
    "cars", "trees", "swimming pool", etc.

    Compatible with PyTorch >= 2.2.2. Requires HuggingFace authentication.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        model_cache_dir: Optional[str] = None,
    ):
        """Initialize SAM3 segmentation service.

        Args:
            device: Device to run inference on ('cuda', 'mps', or 'cpu').
                   Auto-detected if None.
            confidence_threshold: Minimum confidence for detections (0-1)
            model_cache_dir: Optional directory to cache model weights
                           (default: ~/.cache/huggingface/hub/)

        Raises:
            ImportError: If SAM3 dependencies are not installed
            ValueError: If HuggingFace token is not set
        """
        # Set environment for PyTorch
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        # Check for HuggingFace token
        has_hf_token = os.environ.get('HF_TOKEN')
        has_hf_hub_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if not has_hf_token and not has_hf_hub_token:
            raise ValueError(
                "HuggingFace token required for SAM3. "
                "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable. "
                "Get token from: https://huggingface.co/settings/tokens"
            )

        self.confidence_threshold = confidence_threshold
        self.model_cache_dir = model_cache_dir
        self._model = None
        self._processor = None
        self._device = device

    def _load_model(self):
        """Load SAM3 model and create processor."""
        if self._model is not None:
            return

        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise ImportError(
                f"SAM3 dependencies not installed: {e}\n"
                "Install with:\n"
                "  cd models/sam3\n"
                "  pip install -e .\n"
                "  pip install python-dotenv pillow opencv-python"
            )

        # Determine device
        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                # Note: SAM3 may have issues with MPS, default to CPU
                print(
                    "Warning: MPS (Apple Silicon GPU) may have "
                    "compatibility issues with SAM3"
                )
                print("Defaulting to CPU. To force MPS, set device='mps' explicitly.")
                self._device = "cpu"
            else:
                self._device = "cpu"

        print(f"Loading SAM3 model on {self._device}...")
        print("This may take 11-12 seconds and will download ~3.44GB on first run")

        # Load model
        self._model = build_sam3_image_model()
        self._model = self._model.to(self._device)

        # Create processor
        self._processor = Sam3Processor(
            self._model,
            device=self._device,
            confidence_threshold=self.confidence_threshold
        )

        print("SAM3 model loaded successfully")

    def segment_image(
        self,
        satellite_image: Dict,
        prompts: List[str],
    ) -> Dict[str, List[SAM3Detection]]:
        """Run open-vocabulary segmentation on satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: Path to image file
                - center_lat: Center latitude (WGS84)
                - center_lon: Center longitude (WGS84)
                - zoom_level: Zoom level (optional, default 20)
            prompts: List of text prompts for detection
                    (e.g., ["houses", "cars", "trees", "swimming pool"])

        Returns:
            Dictionary mapping prompt to list of SAM3Detection objects

        Example:
            >>> service = SAM3SegmentationService()
            >>> satellite_image = {
            ...     "path": "image.jpg",
            ...     "center_lat": 37.7749,
            ...     "center_lon": -122.4194,
            ...     "zoom_level": 20
            ... }
            >>> results = service.segment_image(
            ...     satellite_image,
            ...     prompts=["houses", "cars", "trees"]
            ... )
            >>> print(f"Found {len(results['houses'])} houses")
        """
        # Load model if needed
        self._load_model()

        # Load image
        image_path = Path(satellite_image["path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            pil_image = img.convert("RGB")

        # Create coordinate converter
        coord_converter = ImageCoordinateConverter.from_satellite_image(
            satellite_image, image_path
        )

        # Run detection for each prompt
        results = {}
        for prompt in prompts:
            print(f"Processing prompt: '{prompt}'...")

            # Set image and run detection
            inference_state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            # Convert to SAM3Detection objects
            detections = []
            detection_id = 0

            for mask, box, score in zip(
                output["masks"],
                output["boxes"],
                output["scores"]
            ):
                if score < self.confidence_threshold:
                    continue

                # Extract mask data
                mask_np = mask.squeeze().cpu().numpy()
                box_np = box.cpu().numpy()

                # Convert box from [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box_np
                pixel_bbox = (int(x1), int(y1), int(x2), int(y2))

                # Calculate area
                area_pixels = int(mask_np.sum())

                # Extract polygon from mask
                geo_polygon = self._mask_to_geo_polygon(
                    mask_np,
                    coord_converter,
                )

                if not geo_polygon or len(geo_polygon) < 3:
                    # Skip invalid polygons
                    continue

                # Calculate area in square meters
                area_sqm = self._calculate_area_sqm(geo_polygon)

                detection = SAM3Detection(
                    detection_id=detection_id,
                    class_name=prompt,
                    confidence=float(score),
                    pixel_mask=mask_np,
                    pixel_bbox=pixel_bbox,
                    geo_polygon=geo_polygon,
                    area_pixels=area_pixels,
                    area_sqm=area_sqm,
                )
                detections.append(detection)
                detection_id += 1

            results[prompt] = detections
            print(f"  Found {len(detections)} {prompt} instances")

        return results

    def segment_image_single(
        self,
        satellite_image: Dict,
        prompt: str,
    ) -> List[SAM3Detection]:
        """Run detection for a single class.

        Convenience method for single-class detection.

        Args:
            satellite_image: Same as segment_image()
            prompt: Single text prompt (e.g., "swimming pool")

        Returns:
            List of SAM3Detection objects
        """
        results = self.segment_image(satellite_image, [prompt])
        return results[prompt]

    def segment_image_geojson(
        self,
        satellite_image: Dict,
        prompts: List[str],
    ) -> Dict:
        """Run segmentation and return GeoJSON FeatureCollection.

        Args:
            satellite_image: Same as segment_image()
            prompts: List of text prompts

        Returns:
            GeoJSON FeatureCollection with all detections
        """
        results = self.segment_image(satellite_image, prompts)

        # Flatten all detections into features
        features = []
        for prompt, detections in results.items():
            for detection in detections:
                features.append(detection.to_geojson_feature())

        # Create metadata
        metadata = {
            "model": "sam3",
            "prompts": prompts,
            "total_detections": sum(len(dets) for dets in results.values()),
            "detections_by_class": {
                prompt: len(dets) for prompt, dets in results.items()
            }
        }

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": metadata,
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


# Example usage
if __name__ == "__main__":
    import sys

    # Check environment
    if not os.environ.get('HF_TOKEN'):
        print("Error: HF_TOKEN environment variable not set")
        print("Set it in .env file or export HF_TOKEN=your_token")
        sys.exit(1)

    # Initialize service
    service = SAM3SegmentationService(confidence_threshold=0.3)

    # Example satellite image
    satellite_image = {
        "path": "test_image.jpg",
        "center_lat": 37.7749,
        "center_lon": -122.4194,
        "zoom_level": 20
    }

    # Run detection
    prompts = ["houses", "cars", "trees", "swimming pool"]
    results = service.segment_image(satellite_image, prompts)

    # Print summary
    print("\n=== Detection Summary ===")
    for prompt, detections in results.items():
        print(f"{prompt}: {len(detections)} detected")
        for det in detections[:3]:  # Show top 3
            print(f"  - Confidence: {det.confidence:.2f}, "
                  f"Area: {det.area_sqm:.1f}m², "
                  f"BBox: {det.pixel_bbox}")
