"""FastAPI REST API for parcel property detection service.

Provides HTTP endpoints for detecting vehicles, pools, amenities, and trees
in satellite imagery.

Clean Architecture Implementation:
- Dependency injection for services (testability)
- Separation of concerns (file I/O, validation, business logic)
- Clear boundaries between layers
"""

from typing import Optional, Callable, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import tempfile
import shutil
from pathlib import Path
import logging
import json

from parcel_ai_json.property_detector import PropertyDetectionService
from parcel_ai_json.device_utils import get_best_device
from parcel_ai_json.sam_segmentation import SAMSegmentationService
from parcel_ai_json.sam3_segmentation import SAM3SegmentationService
from parcel_ai_json.sam_labeler import SAMSegmentLabeler
from parcel_ai_json.grounded_sam_detector import GroundedSAMDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Parcel AI Property Detection API",
    description=(
        "Detect vehicles, pools, amenities, trees, fences, SAM segments, "
        "SAM3 open-vocabulary detections (houses, cars, roofs, etc.), "
        "and custom objects (driveways, patios, sheds, etc.) via Grounded-SAM "
        "in satellite imagery"
    ),
    version="0.1.0",
)

# Initialize detection service (lazy-loaded on first request)
_detector = None
_sam_service = None
_sam3_service = None
_grounded_sam_detector = None


# ============================================================================
# DEPENDENCY INJECTION - Clean Architecture Layer
# ============================================================================


def get_detector() -> PropertyDetectionService:
    """Get or create the PropertyDetectionService singleton.

    This acts as a dependency provider for FastAPI's dependency injection.
    Can be overridden in tests using app.dependency_overrides.
    """
    global _detector
    if _detector is None:
        # Auto-detect best device (cuda/mps/cpu)
        device = get_best_device()
        logger.info(
            f"Initializing PropertyDetectionService with "
            f"DeepForest (device: {device})..."
        )

        # Create property detector with parallel DeepForest + detectree tree detection
        _detector = PropertyDetectionService(
            vehicle_confidence=0.25,
            pool_confidence=0.3,
            amenity_confidence=0.3,
            device=device,  # Auto-detected: cuda/mps/cpu
            tree_confidence=0.1,  # Low threshold to detect more trees
            tree_model_name="weecology/deepforest-tree",
            # Enable detectree polygon extraction (runs natively inside container)
            detectree_extract_polygons=True,
            detectree_min_tree_area_pixels=50,
            detectree_simplify_tolerance_meters=0.5,
            detectree_use_docker=False,  # Native mode inside unified container
        )

        logger.info(f"PropertyDetectionService initialized with DeepForest on {device}")
    return _detector


def get_sam_service() -> SAMSegmentationService:
    """Get or create the SAMSegmentationService singleton.

    This acts as a dependency provider for FastAPI's dependency injection.
    Can be overridden in tests using app.dependency_overrides.
    """
    global _sam_service
    if _sam_service is None:
        device = get_best_device()
        logger.info(f"Initializing SAMSegmentationService (device: {device})...")

        _sam_service = SAMSegmentationService(
            model_type="vit_b",  # Using ViT-B model
            device=device,
            points_per_side=32,  # Standard grid sampling
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=100,
        )

        logger.info(f"SAMSegmentationService initialized on {device}")
    return _sam_service


def get_grounded_sam_detector() -> GroundedSAMDetector:
    """Get or create the GroundedSAMDetector singleton.

    This acts as a dependency provider for FastAPI's dependency injection.
    Can be overridden in tests using app.dependency_overrides.
    """
    global _grounded_sam_detector
    if _grounded_sam_detector is None:
        device = get_best_device()
        logger.info(f"Initializing GroundedSAMDetector (device: {device})...")

        _grounded_sam_detector = GroundedSAMDetector(
            box_threshold=0.25,
            text_threshold=0.20,
            use_sam=True,
            device=device,
        )

        logger.info(f"GroundedSAMDetector initialized on {device}")
    return _grounded_sam_detector


def get_sam3_service() -> SAM3SegmentationService:
    """Get or create the SAM3SegmentationService singleton.

    This acts as a dependency provider for FastAPI's dependency injection.
    Can be overridden in tests using app.dependency_overrides.
    """
    global _sam3_service
    if _sam3_service is None:
        device = get_best_device()
        logger.info(f"Initializing SAM3SegmentationService (device: {device})...")

        _sam3_service = SAM3SegmentationService(
            device=device,
            confidence_threshold=0.3,
        )

        logger.info(f"SAM3SegmentationService initialized on {device}")
    return _sam3_service


# ============================================================================
# VALIDATION LAYER - Input Validation Functions
# ============================================================================


def validate_image_content_type(content_type: Optional[str]) -> None:
    """Validate that uploaded file is an image."""
    if not content_type or not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")


def validate_coordinates(center_lat: float, center_lon: float) -> None:
    """Validate latitude and longitude are within valid ranges."""
    if not -90 <= center_lat <= 90:
        raise HTTPException(
            status_code=400, detail="Latitude must be between -90 and 90"
        )
    if not -180 <= center_lon <= 180:
        raise HTTPException(
            status_code=400, detail="Longitude must be between -180 and 180"
        )


def validate_zoom_level(zoom_level: int) -> None:
    """Validate zoom level is within valid range."""
    if not 1 <= zoom_level <= 22:
        raise HTTPException(status_code=400, detail="Zoom level must be between 1-22")


def validate_format(format: str) -> None:
    """Validate output format is supported."""
    if format not in ["geojson", "summary"]:
        raise HTTPException(
            status_code=400, detail="Format must be 'geojson' or 'summary'"
        )


def validate_sam_points_per_side(points_per_side: int) -> None:
    """Validate SAM points_per_side is within valid range."""
    if not 8 <= points_per_side <= 64:
        raise HTTPException(
            status_code=400, detail="points_per_side must be between 8 and 64"
        )


# ============================================================================
# FILE HANDLING LAYER - I/O Operations
# ============================================================================


class TempFileManager:
    """Context manager for temporary file operations.

    Handles file upload, storage, and cleanup following Clean Architecture
    principles by separating I/O concerns from business logic.
    """

    def __init__(self, image: UploadFile):
        self.image = image
        self.temp_dir: Optional[Path] = None
        self.image_path: Optional[Path] = None

    def __enter__(self) -> Dict[str, Any]:
        """Save uploaded file to temporary directory and return metadata."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.image_path = self.temp_dir / self.image.filename

        with open(self.image_path, "wb") as f:
            shutil.copyfileobj(self.image.file, f)

        logger.info(f"Saved uploaded image to {self.image_path}")
        return {"path": str(self.image_path)}

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def save_uploaded_file(image: UploadFile) -> tuple[Path, Path]:
    """Save uploaded file to temporary directory.

    Returns:
        Tuple of (temp_dir, image_path)
    """
    temp_dir = Path(tempfile.mkdtemp())
    image_path = temp_dir / image.filename
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    return temp_dir, image_path


def cleanup_temp_dir(temp_dir: Optional[Path]) -> None:
    """Clean up temporary directory."""
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# BUSINESS LOGIC LAYER - Detection Orchestration
# ============================================================================


def parse_regrid_parcel_polygon(regrid_json: Optional[str]) -> Optional[Any]:
    """Parse Regrid parcel polygon from JSON string.

    Handles multiple formats:
    - GeoJSON Feature (extracts geometry)
    - GeoJSON Geometry (uses as-is)
    - List of coordinates

    Args:
        regrid_json: JSON string containing parcel polygon data

    Returns:
        Parsed parcel polygon or None

    Raises:
        HTTPException: If JSON is invalid
    """
    if regrid_json is None:
        return None

    try:
        parcel_data = json.loads(regrid_json)
        logger.info("Parsed Regrid parcel polygon from JSON")

        # Handle GeoJSON Feature format (extract geometry)
        if isinstance(parcel_data, dict):
            if parcel_data.get("type") == "Feature" and "geometry" in parcel_data:
                # Extract geometry from Feature
                parcel_polygon = parcel_data["geometry"]
                geom_type = parcel_polygon.get("type")
                logger.info(f"Extracted geometry from GeoJSON Feature: {geom_type}")
            elif "coordinates" in parcel_data:
                # Already a geometry object
                parcel_polygon = parcel_data
                geom_type = parcel_data.get("type")
                logger.info(f"Using GeoJSON geometry: {geom_type}")
            else:
                keys = list(parcel_data.keys())
                logger.warning(f"Unknown parcel polygon format with keys: {keys}")
                parcel_polygon = parcel_data
        elif isinstance(parcel_data, list):
            # List of coordinate tuples
            parcel_polygon = parcel_data
            num_points = len(parcel_data)
            logger.info(f"Using coordinate list with {num_points} points")
        else:
            logger.warning(f"Unexpected parcel polygon type: {type(parcel_data)}")
            parcel_polygon = parcel_data

        # Debug logging
        if isinstance(parcel_polygon, dict) and "coordinates" in parcel_polygon:
            coords = parcel_polygon.get("coordinates", [[]])[0]
            logger.info(f"Parcel polygon has {len(coords)} coordinate points")

        return parcel_polygon

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in regrid_parcel_polygon: {e}",
        )


def create_satellite_image_metadata(
    image_path: Path,
    center_lat: float,
    center_lon: float,
    zoom_level: int,
) -> Dict[str, Any]:
    """Create satellite image metadata dictionary.

    Args:
        image_path: Path to saved image file
        center_lat: Center latitude (WGS84)
        center_lon: Center longitude (WGS84)
        zoom_level: Zoom level

    Returns:
        Satellite image metadata dict
    """
    return {
        "path": str(image_path),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom_level": zoom_level,
    }


def add_regrid_parcel_to_geojson(
    geojson: Dict[str, Any],
    parcel_polygon: Any,
) -> None:
    """Add Regrid parcel polygon to GeoJSON output.

    Modifies geojson dict in-place by appending parcel feature.

    Args:
        geojson: GeoJSON FeatureCollection dict
        parcel_polygon: Parcel polygon data (various formats)
    """
    # Handle different formats: Feature, Geometry, or plain coordinate list
    if isinstance(parcel_polygon, dict):
        if parcel_polygon.get("type") == "Feature":
            # Extract geometry from Feature
            geojson_geometry = parcel_polygon.get("geometry", {})
        elif "type" in parcel_polygon and "coordinates" in parcel_polygon:
            # Already a GeoJSON geometry (Polygon, etc.)
            geojson_geometry = parcel_polygon
        else:
            # Unknown dict format, try to use as-is
            geojson_geometry = parcel_polygon
    else:
        # Convert coordinate list to GeoJSON Polygon
        # Ensure coordinates are in nested list format: [[[lon, lat], ...]]
        coords = parcel_polygon if isinstance(parcel_polygon, list) else []
        geojson_geometry = {
            "type": "Polygon",
            "coordinates": [coords] if coords else [[]],
        }

    geojson["features"].append(
        {
            "type": "Feature",
            "geometry": geojson_geometry,
            "properties": {
                "feature_type": "regrid_parcel",
                "detection_type": "regrid_parcel",
                "source": "regrid",
            },
        }
    )
    logger.info("Added Regrid parcel polygon to GeoJSON output")


# ============================================================================
# API ENDPOINTS - Presentation Layer
# ============================================================================


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Parcel AI Property Detection API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():
    """Detailed health check with service status."""
    return {
        "status": "healthy",
        "detector_loaded": _detector is not None,
    }


@app.post("/detect")
async def detect_property(
    image: UploadFile = File(..., description="Satellite image file (JPEG/PNG)"),
    center_lat: float = Form(..., description="Image center latitude (WGS84)"),
    center_lon: float = Form(..., description="Image center longitude (WGS84)"),
    zoom_level: int = Form(20, description="Zoom level (default: 20)"),
    format: str = Form("geojson", description="Output format: 'geojson' or 'summary'"),
    include_sam: bool = Form(
        False, description="Include SAM segmentation (default: False)"
    ),
    sam_points_per_side: int = Form(
        32, description="SAM grid sampling density (default: 32)"
    ),
    label_sam_segments: bool = Form(
        True, description="Label SAM segments with semantic labels (default: True)"
    ),
    detect_fences: bool = Form(
        False, description="Include fence detection (default: False)"
    ),
    include_grounded_sam: bool = Form(
        False, description="Include Grounded-SAM detection (default: False)"
    ),
    grounded_sam_prompts: str = Form(
        "driveway, patio, deck, shed, gazebo, pergola, hot tub, fire pit, "
        "pool house, dog house, playground equipment, trampoline, basketball hoop, "
        "above ground pool, boat, RV, trailer, carport, greenhouse, chicken coop",
        description=(
            "Comma-separated text prompts for Grounded-SAM detection "
            "(default: comprehensive residential property features)"
        ),
    ),
    regrid_parcel_polygon: Optional[str] = Form(
        None,
        description=(
            "Regrid parcel polygon as JSON string "
            "(coordinates array in WGS84 format). Required only for fence detection."
        ),
    ),
    detector: PropertyDetectionService = Depends(get_detector),
    sam_service: SAMSegmentationService = Depends(get_sam_service),
    grounded_sam_detector: GroundedSAMDetector = Depends(get_grounded_sam_detector),
):
    """Detect all property features in satellite image.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        format: Output format - 'geojson' (default) or 'summary'
        include_sam: Include SAM segmentation (default: False)
        sam_points_per_side: SAM grid sampling density (default: 32)
        label_sam_segments: Label SAM segments with semantic labels (default: True)
        detect_fences: Include fence detection (default: False)
        include_grounded_sam: Include Grounded-SAM detection (default: False)
        grounded_sam_prompts: Text prompts for Grounded-SAM detection
        regrid_parcel_polygon: Required Regrid parcel polygon as JSON string
        detector: Injected PropertyDetectionService (for testing)
        sam_service: Injected SAMSegmentationService (for testing)
        grounded_sam_detector: Injected GroundedSAMDetector (for testing)

    Returns:
        JSON with detected features in GeoJSON format or summary statistics
    """
    logger.info(
        f"Processing detection request: lat={center_lat}, lon={center_lon}, "
        f"zoom={zoom_level}, format={format}, include_sam={include_sam}, "
        f"detect_fences={detect_fences}, include_grounded_sam={include_grounded_sam}"
    )

    # Validate inputs (Clean Architecture - Input Validation Layer)
    validate_image_content_type(image.content_type)
    validate_coordinates(center_lat, center_lon)
    validate_zoom_level(zoom_level)
    validate_format(format)
    if include_sam:
        validate_sam_points_per_side(sam_points_per_side)

    # Parse Regrid parcel polygon (Business Logic Layer)
    # Only required for fence detection
    if detect_fences and not regrid_parcel_polygon:
        raise HTTPException(
            status_code=400,
            detail="regrid_parcel_polygon is required when detect_fences=true",
        )
    parcel_polygon = parse_regrid_parcel_polygon(regrid_parcel_polygon)

    # File handling (I/O Layer)
    temp_dir = None
    try:
        temp_dir, image_path = save_uploaded_file(image)

        # Create satellite image metadata (Business Logic Layer)
        satellite_image = create_satellite_image_metadata(
            image_path, center_lat, center_lon, zoom_level
        )

        # Run detection (Business Logic Layer)
        if format == "summary":
            # Return summary statistics only
            detections = detector.detect_all(
                satellite_image,
                detect_fences=detect_fences,
                regrid_parcel_polygon=parcel_polygon,
            )
            summary = detections.summary()

            # Add SAM segment count if requested
            if include_sam:
                if sam_points_per_side != sam_service.points_per_side:
                    sam_service.points_per_side = sam_points_per_side
                sam_segments = sam_service.segment_image(satellite_image)
                summary["sam_segments"] = len(sam_segments)

            logger.info(f"Detection complete: {summary}")
            return JSONResponse(content=summary)
        else:
            # Return full GeoJSON
            detections = detector.detect_all(
                satellite_image,
                detect_fences=detect_fences,
                regrid_parcel_polygon=parcel_polygon,
            )
            geojson = detections.to_geojson()

            # Add SAM segments if requested
            if include_sam:
                logger.info("Running SAM segmentation...")
                if sam_points_per_side != sam_service.points_per_side:
                    logger.info(
                        f"Updating SAM points_per_side to {sam_points_per_side}"
                    )
                    sam_service.points_per_side = sam_points_per_side

                sam_segments = sam_service.segment_image(satellite_image)

                # Label SAM segments if requested
                if label_sam_segments:
                    logger.info("Labeling SAM segments with semantic labels...")

                    # Create detection dictionary
                    detection_dict = {
                        "vehicles": detections.vehicles or [],
                        "pools": detections.swimming_pools or [],
                        "amenities": detections.amenities or [],
                        "trees": detections.trees.trees if detections.trees else [],
                        "tree_polygons": (
                            detections.trees.tree_polygons
                            if detections.trees and detections.trees.tree_polygons
                            else []
                        ),
                    }

                    # Label segments (with OSM buildings support)
                    labeler = SAMSegmentLabeler(
                        overlap_threshold=0.3, osm_overlap_threshold=0.5, use_osm=True
                    )
                    sam_segments = labeler.label_segments(
                        sam_segments,
                        detection_dict,
                        satellite_image,  # Pass satellite_image for OSM fetching
                    )

                    labeled_count = sum(
                        1 for seg in sam_segments if seg.primary_label != "unknown"
                    )
                    logger.info(
                        f"Labeled {len(sam_segments)} segments: "
                        f"{labeled_count} with semantic labels"
                    )

                # Add SAM segments to GeoJSON
                sam_features = [seg.to_geojson_feature() for seg in sam_segments]
                geojson["features"].extend(sam_features)

                logger.info(
                    f"SAM segmentation complete: {len(sam_segments)} segments added"
                )

                # Add OSM buildings to GeoJSON output if they were fetched
                if label_sam_segments and labeler:
                    osm_buildings = labeler.last_osm_buildings
                    if osm_buildings:
                        for building in osm_buildings:
                            geojson["features"].append(building.to_geojson_feature())
                        num_buildings = len(osm_buildings)
                        logger.info(
                            f"Added {num_buildings} OSM buildings to " f"GeoJSON output"
                        )

            # Add Grounded-SAM detections if requested
            if include_grounded_sam:
                logger.info("Running Grounded-SAM detection...")
                logger.info(f"Prompts: {grounded_sam_prompts}")

                # Parse prompts from comma-separated string
                prompt_list = [p.strip() for p in grounded_sam_prompts.split(",")]

                # Run Grounded-SAM detection
                from parcel_ai_json.coordinate_converter import ImageCoordinateConverter
                from PIL import Image as PILImage

                # Create coordinate converter
                with PILImage.open(image_path) as img:
                    img_width, img_height = img.size

                converter = ImageCoordinateConverter(
                    center_lat=center_lat,
                    center_lon=center_lon,
                    image_width_px=img_width,
                    image_height_px=img_height,
                    zoom_level=zoom_level,
                )

                # Run detection
                grounded_detections = grounded_sam_detector.detect(
                    image=image_path,
                    prompts=prompt_list,
                    coordinate_converter=converter,
                )

                # Add to GeoJSON
                grounded_features = [det.to_geojson_feature() for det in grounded_detections]
                geojson["features"].extend(grounded_features)

                logger.info(
                    f"Grounded-SAM detection complete: "
                    f"{len(grounded_detections)} objects detected"
                )

            # Add regrid parcel polygon to GeoJSON output if provided
            if parcel_polygon is not None:
                add_regrid_parcel_to_geojson(geojson, parcel_polygon)

            logger.info(
                f"Detection complete: {len(geojson['features'])} total features"
            )
            return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    finally:
        # Clean up temporary files (I/O Layer)
        cleanup_temp_dir(temp_dir)


async def _handle_detection_request(
    image: UploadFile,
    center_lat: float,
    center_lon: float,
    zoom_level: int,
    detection_func: Callable,
    feature_name: str,
):
    """Generic handler for detection requests with common boilerplate.

    Args:
        image: Uploaded satellite image file
        center_lat: Center latitude of image
        center_lon: Center longitude of image
        zoom_level: Google Maps zoom level
        detection_func: Callable that performs detection
            (takes satellite_image dict, returns list of detections)
        feature_name: Name of feature being detected
            (e.g., "vehicle", "pool", "amenity", "tree")

    Returns:
        JSONResponse with GeoJSON FeatureCollection

    Raises:
        HTTPException: On detection failure
    """
    logger.info(
        f"Processing {feature_name} detection: " f"lat={center_lat}, lon={center_lon}"
    )

    temp_dir = None
    try:
        # Save uploaded file to temporary directory (I/O Layer)
        temp_dir, image_path = save_uploaded_file(image)

        # Create satellite image metadata (Business Logic Layer)
        satellite_image = create_satellite_image_metadata(
            image_path, center_lat, center_lon, zoom_level
        )

        # Run detection (Business Logic Layer)
        detections = detection_func(satellite_image)

        # Convert to GeoJSON (Presentation Layer)
        geojson = {
            "type": "FeatureCollection",
            "features": [d.to_geojson_feature() for d in detections],
        }

        logger.info(f"Found {len(detections)} {feature_name}s")
        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"{feature_name} detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"{feature_name.capitalize()} detection failed: {str(e)}",
        )

    finally:
        # Clean up temporary files (I/O Layer)
        cleanup_temp_dir(temp_dir)


@app.post("/detect/vehicles")
async def detect_vehicles(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
    detector: PropertyDetectionService = Depends(get_detector),
):
    """Detect only vehicles in satellite image."""
    return await _handle_detection_request(
        image=image,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_level=zoom_level,
        detection_func=detector.vehicle_detector.detect_vehicles,
        feature_name="vehicle",
    )


@app.post("/detect/pools")
async def detect_pools(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
    detector: PropertyDetectionService = Depends(get_detector),
):
    """Detect only swimming pools in satellite image."""
    return await _handle_detection_request(
        image=image,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_level=zoom_level,
        detection_func=detector.pool_detector.detect_swimming_pools,
        feature_name="pool",
    )


@app.post("/detect/amenities")
async def detect_amenities(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
    detector: PropertyDetectionService = Depends(get_detector),
):
    """Detect only amenities (tennis/basketball courts, etc.)."""
    return await _handle_detection_request(
        image=image,
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_level=zoom_level,
        detection_func=detector.amenity_detector.detect_amenities,
        feature_name="amenity",
    )


@app.post("/detect/trees")
async def detect_trees(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
    detector: PropertyDetectionService = Depends(get_detector),
):
    """Detect tree coverage in satellite image."""
    logger.info(f"Processing tree detection: lat={center_lat}, lon={center_lon}")

    temp_dir = None
    try:
        temp_dir, image_path = save_uploaded_file(image)

        satellite_image = create_satellite_image_metadata(
            image_path, center_lat, center_lon, zoom_level
        )

        tree_detection = detector.tree_detector.detect_trees(satellite_image)

        logger.info(
            f"Tree coverage: {tree_detection.tree_coverage_percent:.1f}% "
            f"({tree_detection.tree_pixel_count}/{tree_detection.total_pixels} pixels)"
        )
        return JSONResponse(content=tree_detection.to_dict())

    except Exception as e:
        logger.error(f"Tree detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tree detection failed: {str(e)}")

    finally:
        cleanup_temp_dir(temp_dir)


@app.post("/detect/fences")
async def detect_fences(
    image: UploadFile = File(..., description="Satellite image file (JPEG/PNG)"),
    center_lat: float = Form(..., description="Image center latitude (WGS84)"),
    center_lon: float = Form(..., description="Image center longitude (WGS84)"),
    zoom_level: int = Form(20, description="Zoom level (default: 20)"),
    fence_mask: UploadFile = File(
        None,
        description=(
            "Optional fence probability mask from Regrid " "(512x512 PNG/NPY)"
        ),
    ),
    threshold: float = Form(
        0.05,
        description=(
            "Probability threshold " "(default: 0.05 - lowered for better detection)"
        ),
    ),
    detector: PropertyDetectionService = Depends(get_detector),
):
    """Detect fences in satellite image using HED model.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        fence_mask: Optional fence probability mask from Regrid (512x512 PNG/NPY)
        threshold: Probability threshold for fence detection (default: 0.1)
        detector: Injected PropertyDetectionService (for testing)

    Returns:
        GeoJSON FeatureCollection with fence polygon features
    """
    logger.info(f"Processing fence detection: lat={center_lat}, lon={center_lon}")

    temp_dir = None
    try:
        temp_dir, image_path = save_uploaded_file(image)

        # Process fence mask if provided
        fence_probability_mask = None
        if fence_mask is not None:
            import numpy as np
            from PIL import Image

            fence_mask_path = temp_dir / fence_mask.filename
            with open(fence_mask_path, "wb") as f:
                shutil.copyfileobj(fence_mask.file, f)

            # Load fence mask (supports PNG or NPY)
            if fence_mask.filename.endswith(".npy"):
                fence_probability_mask = np.load(fence_mask_path)
            else:
                fence_img = Image.open(fence_mask_path).convert("L")
                fence_probability_mask = np.array(fence_img)

            logger.info(
                f"Loaded fence probability mask: {fence_probability_mask.shape}"
            )

        satellite_image = create_satellite_image_metadata(
            image_path, center_lat, center_lon, zoom_level
        )

        # Update fence detector threshold if different from default
        if threshold != detector.fence_detector.threshold:
            detector.fence_detector.threshold = threshold
            logger.info(f"Updated fence detection threshold to {threshold}")

        fences = detector.fence_detector.detect_fences(
            satellite_image, fence_probability_mask
        )

        # Convert to GeoJSON
        geojson = fences.to_geojson_features() if fences.geo_polygons else []
        result = {
            "type": "FeatureCollection",
            "features": geojson,
            "metadata": fences.to_dict(),
        }

        logger.info(f"Found {len(fences.geo_polygons)} fence segments")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Fence detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fence detection failed: {str(e)}")

    finally:
        cleanup_temp_dir(temp_dir)


@app.post("/detect/grounded-sam")
async def detect_grounded_sam(
    image: UploadFile = File(..., description="Satellite image file (JPEG/PNG)"),
    center_lat: float = Form(..., description="Image center latitude (WGS84)"),
    center_lon: float = Form(..., description="Image center longitude (WGS84)"),
    zoom_level: int = Form(20, description="Zoom level (default: 20)"),
    prompts: str = Form(
        ...,
        description=(
            "Comma-separated text prompts for objects to detect "
            "(e.g., 'driveway, patio, shed, gazebo')"
        ),
    ),
    grounded_sam: GroundedSAMDetector = Depends(get_grounded_sam_detector),
):
    """Detect property features using text prompts with Grounded-SAM.

    Combines GroundingDINO (text-prompted detection) with SAM (segmentation)
    for open-vocabulary object detection.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        prompts: Comma-separated text prompts (e.g., "driveway, patio, shed")
        grounded_sam: Injected GroundedSAMDetector (for testing)

    Returns:
        GeoJSON FeatureCollection with detected objects

    Example:
        curl -X POST http://localhost:8000/detect/grounded-sam \\
          -F "image=@image.jpg" \\
          -F "center_lat=37.7749" \\
          -F "center_lon=-122.4194" \\
          -F "zoom_level=20" \\
          -F "prompts=driveway, patio, deck, shed, gazebo"
    """
    logger.info(
        f"Processing Grounded-SAM detection: lat={center_lat}, lon={center_lon}, "
        f"prompts='{prompts}'"
    )

    # Validate inputs
    validate_image_content_type(image.content_type)
    validate_coordinates(center_lat, center_lon)
    validate_zoom_level(zoom_level)

    temp_dir = None
    try:
        temp_dir, image_path = save_uploaded_file(image)

        # Create coordinate converter
        from PIL import Image as PILImage
        from parcel_ai_json.coordinate_converter import ImageCoordinateConverter

        pil_image = PILImage.open(image_path)
        converter = ImageCoordinateConverter(
            center_lat=center_lat,
            center_lon=center_lon,
            image_width_px=pil_image.width,
            image_height_px=pil_image.height,
            zoom_level=zoom_level,
        )

        # Parse prompts (comma-separated)
        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        logger.info(f"Parsed {len(prompt_list)} prompts: {prompt_list}")

        # Run detection
        detections = grounded_sam.detect(
            image=pil_image,
            prompts=prompt_list,
            coordinate_converter=converter,
        )

        # Convert to GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": [det.to_geojson_feature() for det in detections],
            "metadata": {
                "prompts": prompt_list,
                "num_detections": len(detections),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "zoom_level": zoom_level,
            },
        }

        logger.info(f"Grounded-SAM complete: {len(detections)} detections found")
        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"Grounded-SAM detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Grounded-SAM detection failed: {str(e)}"
        )

    finally:
        cleanup_temp_dir(temp_dir)


@app.post("/segment/sam")
async def segment_sam(
    image: UploadFile = File(..., description="Satellite image file (JPEG/PNG)"),
    center_lat: float = Form(..., description="Image center latitude (WGS84)"),
    center_lon: float = Form(..., description="Image center longitude (WGS84)"),
    zoom_level: int = Form(20, description="Zoom level (default: 20)"),
    points_per_side: int = Form(
        32, description="SAM grid sampling density (default: 32)"
    ),
    sam_service: SAMSegmentationService = Depends(get_sam_service),
):
    """Run SAM segmentation on satellite image.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        points_per_side: SAM grid sampling density (default: 32)
        sam_service: Injected SAMSegmentationService (for testing)

    Returns:
        GeoJSON with SAM segments
    """
    logger.info(
        f"Processing SAM segmentation: lat={center_lat}, lon={center_lon}, "
        f"zoom={zoom_level}, points_per_side={points_per_side}"
    )

    # Validate inputs
    validate_image_content_type(image.content_type)
    validate_coordinates(center_lat, center_lon)
    validate_zoom_level(zoom_level)
    validate_sam_points_per_side(points_per_side)

    # Save uploaded file to temporary location
    temp_dir = None
    try:
        temp_dir, image_path = save_uploaded_file(image)

        logger.info(f"Saved uploaded image to {image_path}")

        # Prepare satellite image metadata
        satellite_image = create_satellite_image_metadata(
            image_path, center_lat, center_lon, zoom_level
        )

        # Update points_per_side if different from default
        if points_per_side != sam_service.points_per_side:
            logger.info(f"Updating SAM points_per_side to {points_per_side}")
            sam_service.points_per_side = points_per_side

        segments = sam_service.segment_image(satellite_image)

        # Convert to GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": [seg.to_geojson_feature() for seg in segments],
        }

        logger.info(f"SAM segmentation complete: {len(segments)} segments found")
        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"SAM segmentation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"SAM segmentation failed: {str(e)}"
        )

    finally:
        # Clean up temporary files
        cleanup_temp_dir(temp_dir)


@app.post("/segment/sam3")
async def segment_sam3(
    image: UploadFile = File(..., description="Satellite image file (JPEG/PNG)"),
    center_lat: float = Form(..., description="Image center latitude (WGS84)"),
    center_lon: float = Form(..., description="Image center longitude (WGS84)"),
    zoom_level: int = Form(20, description="Zoom level (default: 20)"),
    prompts: str = Form(
        ...,
        description="Comma-separated text prompts (e.g., 'houses,cars,trees,swimming pool')",
    ),
    confidence_threshold: float = Form(
        0.3, description="Minimum confidence threshold (0-1, default: 0.3)"
    ),
    sam3_service: SAM3SegmentationService = Depends(get_sam3_service),
):
    """Run SAM3 open-vocabulary segmentation on satellite image.

    SAM3 uses natural language prompts to detect and segment specific objects.
    Unlike the original SAM, SAM3 can detect specific object classes using text prompts.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        prompts: Comma-separated text prompts (e.g., "houses,cars,trees,swimming pool")
        confidence_threshold: Minimum confidence threshold (0-1, default: 0.3)
        sam3_service: Injected SAM3SegmentationService (for testing)

    Returns:
        GeoJSON with SAM3 detections, grouped by class

    Example:
        curl -X POST "http://localhost:8000/segment/sam3" \\
             -F "image=@image.jpg" \\
             -F "center_lat=37.7749" \\
             -F "center_lon=-122.4194" \\
             -F "zoom_level=20" \\
             -F "prompts=houses,cars,trees,swimming pool" \\
             -F "confidence_threshold=0.3"
    """
    logger.info(
        f"Processing SAM3 segmentation: lat={center_lat}, lon={center_lon}, "
        f"zoom={zoom_level}, prompts='{prompts}', confidence={confidence_threshold}"
    )

    # Validate inputs
    validate_image_content_type(image.content_type)
    validate_coordinates(center_lat, center_lon)
    validate_zoom_level(zoom_level)

    # Parse prompts
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
    if not prompt_list:
        raise HTTPException(
            status_code=400,
            detail="At least one prompt is required (e.g., 'houses,cars')",
        )

    # Validate confidence threshold
    if not 0 <= confidence_threshold <= 1:
        raise HTTPException(
            status_code=400, detail="confidence_threshold must be between 0 and 1"
        )

    # Save uploaded file to temporary location
    temp_dir = None
    try:
        temp_dir, image_path = save_uploaded_file(image)

        logger.info(f"Saved uploaded image to {image_path}")

        # Prepare satellite image metadata
        satellite_image = create_satellite_image_metadata(
            image_path, center_lat, center_lon, zoom_level
        )

        # Update confidence threshold if different from default
        if confidence_threshold != sam3_service.confidence_threshold:
            logger.info(f"Updating SAM3 confidence threshold to {confidence_threshold}")
            sam3_service.confidence_threshold = confidence_threshold

        # Run SAM3 segmentation
        results = sam3_service.segment_image(satellite_image, prompt_list)

        # Convert to GeoJSON
        geojson = sam3_service.segment_image_geojson(satellite_image, prompt_list)

        # Add summary statistics
        total_detections = sum(len(dets) for dets in results.values())
        logger.info(
            f"SAM3 segmentation complete: {total_detections} total detections across "
            f"{len(prompt_list)} classes"
        )

        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"SAM3 segmentation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"SAM3 segmentation failed: {str(e)}"
        )

    finally:
        # Clean up temporary files
        cleanup_temp_dir(temp_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
