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


def get_detector() -> PropertyDetectionService:
    """Get or create the PropertyDetectionService singleton."""
    global _detector
    if _detector is None:
        logger.info(
            "Initializing PropertyDetectionService with tree polygon extraction..."
        )
        # Create tree detector with polygon extraction and simplification
        from parcel_ai_json.tree_detector import TreeDetectionService
        tree_detector = TreeDetectionService(
            use_docker=False,  # Use native detectree in container
            extract_polygons=True,  # Enable tree polygon extraction
            min_tree_area_pixels=50,  # Filter small noise regions
            simplify_tolerance_meters=1.0,  # Simplify polygons (1m tolerance)
        )

        # Create property detector and inject custom tree detector
        _detector = PropertyDetectionService(
            vehicle_confidence=0.25,
            pool_confidence=0.3,
            amenity_confidence=0.3,
            device="cpu",  # Use "cuda" if GPU available
        )

        # Replace tree detector with our configured instance
        _detector.tree_detector = tree_detector

        logger.info(
            "PropertyDetectionService initialized with tree polygon extraction"
        )
    return _detector


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
):
    """Detect all property features in satellite image.

    Args:
        image: Satellite image file (JPEG or PNG)
        center_lat: Center latitude of image (WGS84)
        center_lon: Center longitude of image (WGS84)
        zoom_level: Zoom level (default: 20)
        format: Output format - 'geojson' (default) or 'summary'

    Returns:
        JSON with detected features in GeoJSON format or summary statistics
    """
    logger.info(
        f"Processing detection request: lat={center_lat}, lon={center_lon}, "
        f"zoom={zoom_level}, format={format}"
    )

    # Validate inputs
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    if not -90 <= center_lat <= 90:
        raise HTTPException(
            status_code=400, detail="Latitude must be between -90 and 90"
        )

    if not -180 <= center_lon <= 180:
        raise HTTPException(
            status_code=400, detail="Longitude must be between -180 and 180"
        )

    if not 1 <= zoom_level <= 22:
        raise HTTPException(status_code=400, detail="Zoom level must be between 1-22")

    if format not in ["geojson", "summary"]:
        raise HTTPException(
            status_code=400, detail="Format must be 'geojson' or 'summary'"
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
            logger.info(f"Detection complete: {summary}")
            return JSONResponse(content=summary)
        else:
            # Return full GeoJSON
            geojson = detector.detect_all_geojson(satellite_image)
            logger.info(
                f"Detection complete: {len(geojson['features'])} features found"
            )
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
        raise HTTPException(
            status_code=500, detail=f"Vehicle detection failed: {str(e)}"
        )

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
        raise HTTPException(
            status_code=500, detail=f"Amenity detection failed: {str(e)}"
        )

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
