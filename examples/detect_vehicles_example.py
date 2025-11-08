"""
Example: Vehicle Detection in Satellite Imagery

This script demonstrates how to use the vehicle detection feature
to detect cars, trucks, and other vehicles in satellite imagery and
add them as GeoJSON features.

Requirements:
    pip install parcel-geojson[vehicle-detection]

This will install ultralytics (YOLOv8) for object detection.
"""

import json
from pathlib import Path
from parcel_geojson import ParcelGeoJSONGenerator

# Example usage with a test address
if __name__ == "__main__":
    # Paths to test data (adjust as needed)
    DET_ROOT = Path(__file__).parent.parent.parent.parent / "det-state-visualizer"

    # Example address with satellite imagery
    address_key = "1440_tanager_ln_petaluma_ca_94954_usa"

    regrid_path = DET_ROOT / f"data/raw/regrid_parcels_data/{address_key}.json"
    building_path = DET_ROOT / f"data/raw/buildings_footprints/{address_key}.json"
    satellite_path = DET_ROOT / f"data/raw/satellite_images/{address_key}.jpg"

    # Satellite image metadata
    # Note: You need to provide center coordinates for accurate pixel→geo conversion
    satellite_image = {
        "path": str(satellite_path),
        "center_lat": 38.244194,  # Center latitude
        "center_lon": -122.612183,  # Center longitude
        "zoom_level": 20,  # Satellite image zoom level (default: 20)
    }

    # Initialize generator (vehicle detection enabled by default)
    generator = ParcelGeoJSONGenerator()

    # Optional: Customize vehicle detection settings
    # generator = ParcelGeoJSONGenerator(
    #     enable_vehicle_detection=True,  # Default: True
    #     vehicle_confidence=0.3,  # Minimum confidence (default: 0.3)
    #     vehicle_model_path="path/to/custom/model.pt",  # Optional custom YOLO model
    # )

    print(f"Generating GeoJSON with vehicle detection for {address_key}...")
    print(f"  Satellite image: {satellite_path}")

    # Generate GeoJSON - vehicles automatically detected from satellite image!
    geojson = generator.generate(
        regrid_json=str(regrid_path),
        building_json=str(building_path),
        satellite_image=satellite_image,  # Vehicles detected automatically
    )

    # Count vehicles detected
    vehicle_features = [
        f for f in geojson["features"] if f["properties"].get("feature_type") == "vehicle"
    ]

    print("\n✓ GeoJSON generated successfully!")
    print(f"  Total features: {len(geojson['features'])}")
    print(f"  Vehicles detected: {len(vehicle_features)}")

    if vehicle_features:
        print("\nDetected vehicles:")
        for i, vehicle in enumerate(vehicle_features, 1):
            props = vehicle["properties"]
            print(f"  {i}. {props['vehicle_class']}")
            print(f"     Confidence: {props['confidence']:.1%}")
            print(f"     Pixel bbox: {props['pixel_bbox']}")

    # Save output
    output_path = Path("output") / "test_geojson" / f"{address_key}_with_vehicles.geojson"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")

    # Optional: Visualize with Folium
    try:
        import folium
        from folium import GeoJson

        # Create map centered on parcel
        center_lat = satellite_image["center_lat"]
        center_lon = satellite_image["center_lon"]

        m = folium.Map(location=[center_lat, center_lon], zoom_start=20)

        # Add GeoJSON with custom styling
        def style_function(feature):
            feature_type = feature["properties"].get("feature_type", "")

            if feature_type == "vehicle":
                return {
                    "fillColor": "red",
                    "color": "darkred",
                    "weight": 2,
                    "fillOpacity": 0.6,
                }
            elif feature_type == "building":
                return {"fillColor": "brown", "color": "black", "weight": 2, "fillOpacity": 0.5}
            elif feature_type == "parcel":
                return {"fillColor": "blue", "color": "darkblue", "weight": 2, "fillOpacity": 0.2}
            else:
                return {}

        GeoJson(geojson, style_function=style_function, name="Parcel with Vehicles").add_to(m)

        # Save map
        map_path = Path("output") / "folium_maps" / f"{address_key}_with_vehicles.html"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(map_path))

        print(f"✓ Map saved to: {map_path}")

    except ImportError:
        print("\n⚠️  Install folium to generate interactive maps: pip install folium")


"""
Notes on Vehicle Detection Models:

1. Default Model (yolov8n.pt):
   - Fast and lightweight
   - Trained on COCO dataset (regular street-level photos)
   - May not work well on overhead satellite imagery
   - Good for initial testing

2. Fine-tuned Models for Overhead Imagery:
   - YOLOv8 can be fine-tuned on satellite/aerial datasets
   - SpaceNet vehicle detection datasets available
   - Consider training custom model for better accuracy

3. Improving Detection Accuracy:
   - Lower confidence threshold (0.2-0.3) for satellite imagery
   - Use higher resolution satellite images
   - Fine-tune model on similar aerial imagery
   - Adjust image preprocessing if needed

4. Performance:
   - YOLOv8n (nano): ~Fast, lower accuracy
   - YOLOv8s (small): ~Medium speed, medium accuracy
   - YOLOv8m (medium): ~Slower, higher accuracy
   - Choose based on your speed/accuracy requirements
"""
