"""FastAPI REST API for parcel property detection service.

Provides HTTP endpoints for detecting vehicles, pools, amenities, and trees
in satellite imagery.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
from pathlib import Path
import logging

from parcel_ai_json.property_detector import PropertyDetectionService
from parcel_ai_json.device_utils import get_best_device
from parcel_ai_json.sam_segmentation import SAMSegmentationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Parcel AI Property Detection API",
    description="Detect vehicles, pools, amenities, and trees in satellite imagery",
    version="0.1.0",
)

# Initialize detection service (lazy-loaded on first request)
_detector = None
_sam_service = None


def get_detector() -> PropertyDetectionService:
    """Get or create the PropertyDetectionService singleton."""
    global _detector
    if _detector is None:
        # Auto-detect best device (cuda/mps/cpu)
        device = get_best_device()
        logger.info(f"Initializing PropertyDetectionService with DeepForest (device: {device})...")

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
    """Get or create the SAMSegmentationService singleton."""
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
    include_sam: bool = Form(False, description="Include SAM segmentation (default: False)"),
    sam_points_per_side: int = Form(32, description="SAM grid sampling density (default: 32)"),
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

    Returns:
        JSON with detected features in GeoJSON format or summary statistics
    """
    logger.info(
        f"Processing detection request: lat={center_lat}, lon={center_lon}, "
        f"zoom={zoom_level}, format={format}, include_sam={include_sam}"
    )

    # Validate inputs
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    if not -90 <= center_lat <= 90:
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")

    if not -180 <= center_lon <= 180:
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")

    if not 1 <= zoom_level <= 22:
        raise HTTPException(status_code=400, detail="Zoom level must be between 1-22")

    if format not in ["geojson", "summary"]:
        raise HTTPException(status_code=400, detail="Format must be 'geojson' or 'summary'")

    if include_sam and not 8 <= sam_points_per_side <= 64:
        raise HTTPException(
            status_code=400, detail="sam_points_per_side must be between 8 and 64"
        )

    # Save uploaded file to temporary location
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        logger.info(f"Saved uploaded image to {image_path}")

        # Prepare satellite image metadata
        satellite_image = {
            "path": str(image_path),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
        }

        # Get detector and run detection
        detector = get_detector()

        if format == "summary":
            # Return summary statistics only
            detections = detector.detect_all(satellite_image)
            summary = detections.summary()

            # Add SAM segment count if requested
            if include_sam:
                sam_service = get_sam_service()
                if sam_points_per_side != sam_service.points_per_side:
                    sam_service.points_per_side = sam_points_per_side
                sam_segments = sam_service.segment_image(satellite_image)
                summary["sam_segments"] = len(sam_segments)

            logger.info(f"Detection complete: {summary}")
            return JSONResponse(content=summary)
        else:
            # Return full GeoJSON
            geojson = detector.detect_all_geojson(satellite_image)

            # Add SAM segments if requested
            if include_sam:
                logger.info("Running SAM segmentation...")
                sam_service = get_sam_service()
                if sam_points_per_side != sam_service.points_per_side:
                    logger.info(f"Updating SAM points_per_side to {sam_points_per_side}")
                    sam_service.points_per_side = sam_points_per_side

                sam_segments = sam_service.segment_image(satellite_image)

                # Add SAM segments to GeoJSON
                sam_features = [seg.to_geojson_feature() for seg in sam_segments]
                geojson["features"].extend(sam_features)

                logger.info(f"SAM segmentation complete: {len(sam_segments)} segments added")

            logger.info(f"Detection complete: {len(geojson['features'])} total features")
            return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    finally:
        # Clean up temporary files
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/detect/vehicles")
async def detect_vehicles(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
):
    """Detect only vehicles in satellite image."""
    logger.info(f"Processing vehicle detection: lat={center_lat}, lon={center_lon}")

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        satellite_image = {
            "path": str(image_path),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
        }

        detector = get_detector()
        vehicles = detector.vehicle_detector.detect_vehicles(satellite_image)

        geojson = {
            "type": "FeatureCollection",
            "features": [v.to_geojson_feature() for v in vehicles],
        }

        logger.info(f"Found {len(vehicles)} vehicles")
        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"Vehicle detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vehicle detection failed: {str(e)}")

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/detect/pools")
async def detect_pools(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
):
    """Detect only swimming pools in satellite image."""
    logger.info(f"Processing pool detection: lat={center_lat}, lon={center_lon}")

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        satellite_image = {
            "path": str(image_path),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
        }

        detector = get_detector()
        pools = detector.pool_detector.detect_swimming_pools(satellite_image)

        geojson = {
            "type": "FeatureCollection",
            "features": [p.to_geojson_feature() for p in pools],
        }

        logger.info(f"Found {len(pools)} pools")
        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"Pool detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pool detection failed: {str(e)}")

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/detect/amenities")
async def detect_amenities(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
):
    """Detect only amenities (tennis courts, basketball courts, etc.) in satellite image."""
    logger.info(f"Processing amenity detection: lat={center_lat}, lon={center_lon}")

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        satellite_image = {
            "path": str(image_path),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
        }

        detector = get_detector()
        amenities = detector.amenity_detector.detect_amenities(satellite_image)

        geojson = {
            "type": "FeatureCollection",
            "features": [a.to_geojson_feature() for a in amenities],
        }

        logger.info(f"Found {len(amenities)} amenities")
        return JSONResponse(content=geojson)

    except Exception as e:
        logger.error(f"Amenity detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Amenity detection failed: {str(e)}")

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/detect/trees")
async def detect_trees(
    image: UploadFile = File(...),
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    zoom_level: int = Form(20),
):
    """Detect tree coverage in satellite image."""
    logger.info(f"Processing tree detection: lat={center_lat}, lon={center_lon}")

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        satellite_image = {
            "path": str(image_path),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
        }

        detector = get_detector()
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
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/segment/sam")
async def segment_sam(
    image: UploadFile = File(..., description="Satellite image file (JPEG/PNG)"),
    center_lat: float = Form(..., description="Image center latitude (WGS84)"),
    center_lon: float = Form(..., description="Image center longitude (WGS84)"),
    zoom_level: int = Form(20, description="Zoom level (default: 20)"),
    points_per_side: int = Form(32, description="SAM grid sampling density (default: 32)"),
):
    """Run SAM segmentation on satellite image.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        points_per_side: SAM grid sampling density (default: 32)

    Returns:
        GeoJSON with SAM segments
    """
    logger.info(
        f"Processing SAM segmentation: lat={center_lat}, lon={center_lon}, "
        f"zoom={zoom_level}, points_per_side={points_per_side}"
    )

    # Validate inputs
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    if not -90 <= center_lat <= 90:
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")

    if not -180 <= center_lon <= 180:
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")

    if not 1 <= zoom_level <= 22:
        raise HTTPException(status_code=400, detail="Zoom level must be between 1-22")

    if not 8 <= points_per_side <= 64:
        raise HTTPException(
            status_code=400, detail="points_per_side must be between 8 and 64"
        )

    # Save uploaded file to temporary location
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / image.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        logger.info(f"Saved uploaded image to {image_path}")

        # Prepare satellite image metadata
        satellite_image = {
            "path": str(image_path),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom_level": zoom_level,
        }

        # Get SAM service and run segmentation
        sam_service = get_sam_service()

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
        raise HTTPException(status_code=500, detail=f"SAM segmentation failed: {str(e)}")

    finally:
        # Clean up temporary files
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
