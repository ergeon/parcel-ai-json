"""Grounded-SAM detector for open-vocabulary object detection with segmentation.

Combines GroundingDINO (text-prompted object detection) with SAM (segmentation)
to detect and segment arbitrary objects using natural language prompts.

Example prompts:
    "driveway", "patio", "deck", "shed", "gazebo", "hot tub",
    "playground equipment", "pergola", "fire pit", "dog house"
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch


@dataclass
class GroundedDetection:
    """Represents a single grounded detection with segmentation.

    Attributes:
        label: Detected object class/label from text prompt
        pixel_bbox: Bounding box in pixels (x1, y1, x2, y2)
        geo_bbox: Geographic bounding box (lon_min, lat_min, lon_max, lat_max)
        geo_polygon: Geographic polygon from SAM segmentation [(lon, lat), ...]
        confidence: Detection confidence score (0-1)
        pixel_mask: Binary segmentation mask (H x W) if SAM enabled
        area_pixels: Area in pixels
        area_sqm: Area in square meters
    """

    label: str
    pixel_bbox: Tuple[float, float, float, float]
    geo_bbox: Tuple[float, float, float, float]
    geo_polygon: List[Tuple[float, float]]
    confidence: float
    pixel_mask: Optional[np.ndarray] = None
    area_pixels: Optional[int] = None
    area_sqm: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "label": self.label,
            "pixel_bbox": list(self.pixel_bbox),
            "geo_bbox": list(self.geo_bbox),
            "geo_polygon": self.geo_polygon,
            "confidence": self.confidence,
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
                "feature_type": "grounded_detection",
                "label": self.label,
                "confidence": self.confidence,
                "area_pixels": self.area_pixels,
                "area_sqm": self.area_sqm,
                "pixel_bbox": list(self.pixel_bbox),
                "geo_bbox": list(self.geo_bbox),
            },
        }


class GroundedSAMDetector:
    """Grounded-SAM detector combining GroundingDINO + SAM.

    Uses GroundingDINO for text-prompted object detection and optionally
    SAM for precise segmentation masks.
    """

    def __init__(
        self,
        grounding_model_path: Optional[str] = None,
        sam_model_path: Optional[str] = None,
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        use_sam: bool = True,
    ):
        """Initialize Grounded-SAM detector.

        Args:
            grounding_model_path: Path to GroundingDINO checkpoint
            sam_model_path: Path to SAM checkpoint (if use_sam=True)
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
            box_threshold: Confidence threshold for detections (0-1)
            text_threshold: Text similarity threshold (0-1)
            use_sam: Whether to use SAM for segmentation masks
        """
        self.grounding_model_path = grounding_model_path
        self.sam_model_path = sam_model_path
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_sam = use_sam

        self._grounding_model = None
        self._sam_predictor = None

    def _load_models(self):
        """Lazy-load GroundingDINO and SAM models."""
        if self._grounding_model is not None:
            return

        # Load GroundingDINO
        try:
            from groundingdino.util.inference import (
                load_model as load_grounding_model
            )
        except ImportError:
            raise ImportError(
                "GroundingDINO requires groundingdino package. "
                "Install with: pip install "
                "git+https://github.com/IDEA-Research/GroundingDINO.git"
            )

        # Determine GroundingDINO model path
        if self.grounding_model_path:
            grounding_path = self.grounding_model_path
        else:
            # Auto-detect in models/ directory
            models_dir = Path(__file__).parent.parent / "models"
            grounding_path = models_dir / "groundingdino_swinb_cogcoor.pth"

            if not grounding_path.exists():
                raise FileNotFoundError(
                    f"GroundingDINO model not found: {grounding_path}\n"
                    "Download from: https://github.com/IDEA-Research/GroundingDINO/releases"
                )

        # GroundingDINO also needs config file
        # Get from installed package
        try:
            import groundingdino
            package_dir = Path(groundingdino.__file__).parent
            config_path = package_dir / "config" / "GroundingDINO_SwinB.cfg.py"

            if not config_path.exists():
                # Try alternative location
                config_path = package_dir / "config" / "GroundingDINO_SwinB_cfg.py"

            print(f"Loading GroundingDINO from: {grounding_path}")
            print(f"Config: {config_path}")

            self._grounding_model = load_grounding_model(
                str(config_path), str(grounding_path), device=self.device
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GroundingDINO model: {e}\n"
                "Make sure you have installed: "
                "git+https://github.com/IDEA-Research/GroundingDINO.git"
            )

        # Load SAM if requested
        if self.use_sam:
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                raise ImportError(
                    "SAM requires segment_anything package. "
                    "Install with: pip install "
                    "git+https://github.com/facebookresearch/segment-anything.git"
                )

            # Determine SAM model path
            if self.sam_model_path:
                sam_path = self.sam_model_path
            else:
                models_dir = Path(__file__).parent.parent / "models"
                sam_path = models_dir / "sam_vit_h_4b8939.pth"

                if not sam_path.exists():
                    raise FileNotFoundError(
                        f"SAM model not found: {sam_path}\n"
                        "Download from: "
                        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                    )

            print(f"Loading SAM from: {sam_path}")

            # SAM doesn't support MPS, force CPU
            sam_device = self.device
            if self.device == "mps":
                print("Warning: SAM doesn't support MPS - using CPU instead")
                sam_device = "cpu"

            sam = sam_model_registry["vit_h"](checkpoint=str(sam_path))
            sam.to(device=sam_device)
            self._sam_predictor = SamPredictor(sam)

    def detect(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompts: Union[str, List[str]],
        coordinate_converter: Optional[object] = None,
    ) -> List[GroundedDetection]:
        """Detect objects using text prompts and optionally segment with SAM.

        Args:
            image: Input image (PIL Image, numpy array, or path)
            prompts: Text prompt(s) describing objects to detect
                    Example: "driveway. patio. deck. shed"
                    Or: ["driveway", "patio", "deck", "shed"]
            coordinate_converter: Optional ImageCoordinateConverter for geo coordinates

        Returns:
            List of GroundedDetection objects with bounding boxes and optional masks
        """
        self._load_models()

        # Load and prepare image
        if isinstance(image, (str, Path)):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image_np = np.array(image_pil)

        # Prepare prompts
        if isinstance(prompts, list):
            # Join with ". " for GroundingDINO
            text_prompt = ". ".join(prompts)
        else:
            text_prompt = prompts

        # Run GroundingDINO detection
        from groundingdino.util.inference import predict

        boxes, logits, phrases = predict(
            model=self._grounding_model,
            image=image_pil,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        # Convert to xyxy format
        h, w = image_np.shape[:2]
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])

        detections = []

        # Process each detection
        for box, logit, phrase in zip(boxes_xyxy, logits, phrases):
            x1, y1, x2, y2 = box.cpu().numpy()
            confidence = logit.item()

            # Get SAM mask if enabled
            if self.use_sam and self._sam_predictor is not None:
                self._sam_predictor.set_image(image_np)
                masks, _, _ = self._sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False,
                )
                pixel_mask = masks[0]

                # Convert mask to polygon
                geo_polygon = self._mask_to_polygon(
                    pixel_mask, coordinate_converter
                )
                area_pixels = int(np.sum(pixel_mask))
            else:
                # Use bounding box as polygon
                pixel_mask = None
                geo_polygon = self._bbox_to_polygon(
                    (x1, y1, x2, y2), coordinate_converter
                )
                area_pixels = int((x2 - x1) * (y2 - y1))

            # Convert to geographic coordinates
            if coordinate_converter:
                lon_min, lat_max = coordinate_converter.pixel_to_geo(x1, y1)
                lon_max, lat_min = coordinate_converter.pixel_to_geo(x2, y2)
                geo_bbox = (lon_min, lat_min, lon_max, lat_max)

                # Calculate area in square meters
                area_sqm = coordinate_converter.calculate_area_sqm(geo_polygon)
            else:
                geo_bbox = (float(x1), float(y1), float(x2), float(y2))
                area_sqm = None

            detection = GroundedDetection(
                label=phrase.strip(),
                pixel_bbox=(float(x1), float(y1), float(x2), float(y2)),
                geo_bbox=geo_bbox,
                geo_polygon=geo_polygon,
                confidence=confidence,
                pixel_mask=pixel_mask,
                area_pixels=area_pixels,
                area_sqm=area_sqm,
            )

            detections.append(detection)

        return detections

    def _box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _bbox_to_polygon(
        self, bbox: Tuple[float, float, float, float], coordinate_converter
    ) -> List[Tuple[float, float]]:
        """Convert bounding box to geographic polygon."""
        x1, y1, x2, y2 = bbox

        if coordinate_converter:
            # Convert corners to geo
            lon1, lat1 = coordinate_converter.pixel_to_geo(x1, y1)
            lon2, lat2 = coordinate_converter.pixel_to_geo(x2, y2)
            return [
                (lon1, lat1),
                (lon2, lat1),
                (lon2, lat2),
                (lon1, lat2),
                (lon1, lat1),  # Close polygon
            ]
        else:
            # Return pixel coordinates
            return [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
                (x1, y1),
            ]

    def _mask_to_polygon(
        self, mask: np.ndarray, coordinate_converter
    ) -> List[Tuple[float, float]]:
        """Convert binary mask to geographic polygon using contours."""
        import cv2

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
        epsilon = 0.005 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to polygon
        polygon = []
        for point in contour:
            x, y = point[0]
            if coordinate_converter:
                lon, lat = coordinate_converter.pixel_to_geo(float(x), float(y))
                polygon.append((lon, lat))
            else:
                polygon.append((float(x), float(y)))

        # Close polygon
        if polygon and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])

        return polygon
