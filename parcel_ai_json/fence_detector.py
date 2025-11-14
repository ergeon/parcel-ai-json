"""Fence detection service using HED model.

Uses HED (Holistically-Nested Edge Detection) with VGG16 backbone
trained on satellite imagery with Regrid parcel data.
"""

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from parcel_ai_json.coordinate_converter import ImageCoordinateConverter
from parcel_ai_json.hed_model import HED


@dataclass
class FenceDetection:
    """Represents detected fence with geographic coordinates."""

    # Pixel coordinates (probability mask)
    probability_mask: np.ndarray  # Shape: (512, 512), values 0-1

    # Binary mask after threshold
    binary_mask: np.ndarray  # Shape: (512, 512), values 0 or 255

    # Geographic polygon (fence boundaries)
    geo_polygons: List[List[Tuple[float, float]]]  # Multiple fence segments

    # Detection metadata
    max_probability: float = 0.0
    mean_probability: float = 0.0
    fence_pixel_count: int = 0
    threshold: float = 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "fence_pixel_count": int(self.fence_pixel_count),
            "max_probability": float(self.max_probability),
            "mean_probability": float(self.mean_probability),
            "threshold": float(self.threshold),
            "fence_segment_count": len(self.geo_polygons),
        }

    def to_geojson_features(self) -> List[Dict]:
        """Convert to list of GeoJSON features (one per fence segment)."""
        features = []
        for i, polygon in enumerate(self.geo_polygons):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon],
                    },
                    "properties": {
                        "feature_type": "fence",
                        "segment_id": i,
                        "max_probability": self.max_probability,
                        "mean_probability": self.mean_probability,
                        "threshold": self.threshold,
                    },
                }
            )
        return features


class FenceDetectionService:
    """Service for detecting fences in satellite imagery using HED.

    Requires 4-channel input:
    - Channels 0-2: RGB satellite image
    - Channel 3: Fence probability from Regrid parcel data
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize fence detection service.

        Args:
            model_path: Path to HED checkpoint
                (default: models/hed_fence_checkpoint_best.pth)
            threshold: Probability threshold for fence detection
                (0.0-1.0, default: 0.1)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.threshold = threshold
        self.device = device
        self._model = None

        # Set default model path
        if model_path is None:
            models_dir = Path(__file__).parent.parent / "models"
            self.model_path = str(models_dir / "hed_fence_weighted_loss.pth")
        else:
            self.model_path = model_path

    def _load_model(self):
        """Lazy-load the HED fence detection model."""
        if self._model is not None:
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"HED fence model not found at: {self.model_path}\n"
                f"Please ensure the model checkpoint is available."
            )

        print(f"Loading HED fence detection model (weighted loss): {self.model_path}")

        # Load model architecture
        self._model = HED(pretrained=False, input_channels=4)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self.device)
        self._model.eval()

        print(f"✓ HED weighted loss model loaded on {self.device}")
        print(f"✓ Trained for {checkpoint['epoch']} epochs")
        print(f"✓ Best val_loss: {checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))}")
        if "pos_weight" in checkpoint:
            print(f"✓ Pos weight (false negative penalty): {checkpoint['pos_weight']}")

    def generate_fence_probability_mask(
        self,
        parcel_polygon: Union[List[Tuple[float, float]], Dict],
        center_lat: float,
        center_lon: float,
        zoom_level: int = 20,
        line_width: int = 3,
        blur_sigma: float = 2.0,
    ) -> np.ndarray:
        """Generate fence probability mask from Regrid parcel polygon.

        This method rasterizes the parcel boundary and applies Gaussian blur
        to create a probability distribution for fence locations.

        Args:
            parcel_polygon: Either:
                - List of (lon, lat) tuples defining parcel boundary
                - GeoJSON polygon dict with 'coordinates' key
            center_lat: Center latitude of satellite image (WGS84)
            center_lon: Center longitude of satellite image (WGS84)
            zoom_level: Zoom level of satellite image (default: 20)
            line_width: Width of parcel boundary line in pixels (default: 3)
            blur_sigma: Standard deviation for Gaussian blur (default: 2.0)

        Returns:
            np.ndarray of shape (512, 512), dtype float32, values in [0, 1]
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python required for fence probability mask generation. "
                "Install with: pip install opencv-python"
            )

        # Parse parcel polygon
        if isinstance(parcel_polygon, dict):
            # GeoJSON format: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}
            coords = parcel_polygon.get("coordinates", [[]])[0]
        else:
            coords = parcel_polygon

        if not coords or len(coords) < 3:
            print(
                "WARNING: Invalid parcel polygon - using empty fence probability mask"
            )
            return np.zeros((512, 512), dtype=np.float32)

        # Create coordinate converter
        coord_converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=512,
            image_height_px=512,
            zoom_level=zoom_level,
        )

        # Convert geographic coords to pixel coords
        pixel_coords = []
        for lon, lat in coords:
            x, y = coord_converter.geo_to_pixel(lon, lat)
            pixel_coords.append([int(x), int(y)])

        # Create blank mask
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Draw parcel boundary
        if len(pixel_coords) >= 2:
            pts = np.array(pixel_coords, dtype=np.int32)
            cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=line_width)

        # Apply Gaussian blur to create probability distribution
        if blur_sigma > 0:
            blurred = gaussian_filter(mask.astype(np.float32), sigma=blur_sigma)
        else:
            blurred = mask.astype(np.float32)

        # Normalize to [0, 1]
        if blurred.max() > 0:
            blurred = blurred / blurred.max()

        print(
            f"✓ Generated fence probability mask from parcel polygon "
            f"(line_width={line_width}, blur_sigma={blur_sigma})"
        )

        return blurred

    def detect_fences(
        self,
        satellite_image: Dict,
        regrid_parcel_polygon: Optional[Union[List[Tuple[float, float]], Dict]] = None,
        fence_probability_mask: Optional[np.ndarray] = None,
    ) -> FenceDetection:
        """Detect fences in satellite image.

        Args:
            satellite_image: Dict with keys:
                - path: str (image file path)
                - center_lat: float (center latitude WGS84)
                - center_lon: float (center longitude WGS84)
                - zoom_level: int (optional, default 20)
            regrid_parcel_polygon: Optional Regrid parcel polygon to generate
                fence probability mask. Can be either:
                - List of (lon, lat) tuples defining parcel boundary
                - GeoJSON polygon dict with 'coordinates' key
                If provided, takes precedence over fence_probability_mask.
            fence_probability_mask: Optional pre-computed fence probability
                mask. Shape: (512, 512), dtype: float32 (0-1) or uint8 (0-255).
                Only used if regrid_parcel_polygon is None.
                If both are None, will use empty mask (zeros).

        Returns:
            FenceDetection object with probability mask and polygons
        """
        self._load_model()

        # Generate fence probability mask from parcel polygon if provided
        if regrid_parcel_polygon is not None:
            center_lat = satellite_image["center_lat"]
            center_lon = satellite_image["center_lon"]
            zoom_level = satellite_image.get("zoom_level", 20)
            fence_probability_mask = self.generate_fence_probability_mask(
                parcel_polygon=regrid_parcel_polygon,
                center_lat=center_lat,
                center_lon=center_lon,
                zoom_level=zoom_level,
            )

        # Extract metadata
        image_path = satellite_image["path"]
        center_lat = satellite_image["center_lat"]
        center_lon = satellite_image["center_lon"]
        zoom_level = satellite_image.get("zoom_level", 20)

        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Prepare 4-channel input
        input_4ch = self._prepare_4channel_input(image_path, fence_probability_mask)

        # Run inference
        with torch.no_grad():
            input_tensor = (
                torch.from_numpy(input_4ch).permute(2, 0, 1).unsqueeze(0)
            )  # NHWC -> NCHW
            input_tensor = input_tensor.to(self.device)

            outputs = self._model(input_tensor)
            prediction = outputs["fused"].cpu().squeeze().numpy()  # Shape: (512, 512)

        # Apply threshold
        binary_mask = (prediction > self.threshold).astype(np.uint8) * 255

        # Extract fence polygons
        geo_polygons = self._extract_fence_polygons(
            binary_mask, center_lat, center_lon, zoom_level
        )

        # Calculate statistics
        fence_pixels = np.sum(binary_mask > 0)

        return FenceDetection(
            probability_mask=prediction,
            binary_mask=binary_mask,
            geo_polygons=geo_polygons,
            max_probability=float(prediction.max()),
            mean_probability=float(prediction.mean()),
            fence_pixel_count=int(fence_pixels),
            threshold=self.threshold,
        )

    def _prepare_4channel_input(
        self, image_path: str, fence_probability_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Prepare 4-channel input (RGB + fence probability).

        Returns:
            np.ndarray of shape (512, 512, 4) with values in [0, 1]
        """
        from PIL import Image

        # Load RGB image
        rgb_image = Image.open(image_path).convert("RGB")

        # Resize to 512x512 if needed
        if rgb_image.size != (512, 512):
            rgb_image = rgb_image.resize((512, 512), Image.LANCZOS)

        rgb_array = np.array(rgb_image).astype(np.float32) / 255.0

        # Prepare fence probability channel
        if fence_probability_mask is None:
            # Generate empty mask (all zeros)
            fence_prob = np.zeros((512, 512), dtype=np.float32)
            print(
                "WARNING: No fence probability mask provided - using zeros "
                "(lower accuracy expected)"
            )
        else:
            # Normalize to [0, 1]
            if fence_probability_mask.shape != (512, 512):
                fence_prob = np.array(
                    Image.fromarray(fence_probability_mask).resize((512, 512))
                ).astype(np.float32)
                if fence_prob.max() > 1.0:
                    fence_prob = fence_prob / 255.0
            else:
                fence_prob = fence_probability_mask.astype(np.float32)
                if fence_prob.max() > 1.0:
                    fence_prob = fence_prob / 255.0

        # Stack into 4 channels
        input_4ch = np.zeros((512, 512, 4), dtype=np.float32)
        input_4ch[:, :, :3] = rgb_array  # RGB channels
        input_4ch[:, :, 3] = fence_prob  # Fence probability channel

        return input_4ch

    def _extract_fence_polygons(
        self,
        binary_mask: np.ndarray,
        center_lat: float,
        center_lon: float,
        zoom_level: int,
    ) -> List[List[Tuple[float, float]]]:
        """Extract fence polygons from binary mask.

        Uses connected components and contour extraction.
        """
        try:
            import cv2
            from scipy import ndimage
        except ImportError:
            print("WARNING: opencv-python and scipy required for polygon extraction")
            print("Install with: pip install opencv-python scipy")
            return []

        # Find connected components
        labeled, num_features = ndimage.label(binary_mask)

        # Create coordinate converter
        coord_converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=512,
            image_height_px=512,
            zoom_level=zoom_level,
        )

        polygons = []

        # Extract contours for each component
        for i in range(1, num_features + 1):
            component_mask = (labeled == i).astype(np.uint8) * 255

            # Skip very small components (noise)
            if np.sum(component_mask > 0) < 10:
                continue

            # Find contours
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if len(contour) < 3:
                    continue

                # Convert pixel coordinates to geographic
                geo_points = []
                for point in contour.squeeze():
                    if point.ndim == 1 and len(point) == 2:
                        x, y = point
                        lon, lat = coord_converter.pixel_to_geo(int(x), int(y))
                        geo_points.append((lon, lat))

                if len(geo_points) >= 3:
                    # Close the polygon
                    geo_points.append(geo_points[0])
                    polygons.append(geo_points)

        return polygons

    def detect_fences_geojson(
        self, satellite_image: Dict, fence_probability_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """Detect fences and return as GeoJSON FeatureCollection.

        Args:
            satellite_image: Dict with satellite metadata
            fence_probability_mask: Optional Regrid fence probability mask

        Returns:
            GeoJSON FeatureCollection with fence features
        """
        detection = self.detect_fences(satellite_image, fence_probability_mask)

        features = detection.to_geojson_features()

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": detection.to_dict(),
        }
