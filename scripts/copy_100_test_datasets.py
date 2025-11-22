"""
Copy 100 test datasets from det-state-visualizer to parcel-ai-json.

Each dataset includes:
- Satellite image (512x512 JPG)
- Regrid parcel polygon (JSON)
- Coordinates (lat, lon)
"""

import shutil
import pandas as pd
from pathlib import Path

# Source paths
SOURCE_BASE = Path("../../det-state-visualizer/data/raw")
SOURCE_INDEX = SOURCE_BASE / "training_index_clean.csv"
SOURCE_IMAGES = SOURCE_BASE / "satellite_images"
SOURCE_REGRID = SOURCE_BASE / "regrid_parcels_data"

# Destination paths
DEST_BASE = Path("output/test_datasets")
DEST_IMAGES = DEST_BASE / "satellite_images"
DEST_REGRID = DEST_BASE / "regrid_parcels"
DEST_INDEX = DEST_BASE / "test_index.csv"

# Number of datasets to copy
NUM_DATASETS = 100


def main():
    """Copy 100 test datasets with all required files."""
    print(f"{'='*80}")
    print("Copying 100 Test Datasets from det-state-visualizer")
    print(f"{'='*80}\n")

    # Create destination directories
    print("1. Creating destination directories...")
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_REGRID.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {DEST_IMAGES}")
    print(f"   ✓ {DEST_REGRID}\n")

    # Load index
    print("2. Loading training index...")
    if not SOURCE_INDEX.exists():
        print(f"   ✗ Error: {SOURCE_INDEX} not found!")
        return False

    df = pd.read_csv(SOURCE_INDEX)
    print(f"   ✓ Loaded {len(df)} records\n")

    # Select first 100 records
    print(f"3. Selecting {NUM_DATASETS} datasets...")
    df_subset = df.head(NUM_DATASETS).copy()
    print(f"   ✓ Selected {len(df_subset)} records\n")

    # Copy files
    print("4. Copying files...")
    success_count = 0
    missing_images = []
    missing_regrid = []

    for idx, row in df_subset.iterrows():
        image_filename = row['image_filename']

        # Derive regrid filename from image filename
        # Example: "2117_ildica_ct_spring_valley_ca_91977.jpg"
        # -> "2117_ildica_ct_spring_valley_ca_91977_usa.json"
        regrid_filename = image_filename.replace('.jpg', '_usa.json')

        # Source paths
        src_image = SOURCE_IMAGES / image_filename
        src_regrid = SOURCE_REGRID / regrid_filename

        # Destination paths
        dst_image = DEST_IMAGES / image_filename
        dst_regrid = DEST_REGRID / regrid_filename

        # Check if source files exist
        image_exists = src_image.exists()
        regrid_exists = src_regrid.exists()

        if not image_exists:
            missing_images.append(image_filename)
        if not regrid_exists:
            missing_regrid.append(regrid_filename)

        # Copy files if they exist
        if image_exists:
            shutil.copy2(src_image, dst_image)
        if regrid_exists:
            shutil.copy2(src_regrid, dst_regrid)

        if image_exists and regrid_exists:
            success_count += 1
            if success_count % 10 == 0:
                print(f"   Copied {success_count}/{NUM_DATASETS} datasets...")

    print(f"\n   ✓ Copied {success_count}/{NUM_DATASETS} complete datasets")

    if missing_images:
        print(f"   ⚠ Missing {len(missing_images)} satellite images")
    if missing_regrid:
        print(f"   ⚠ Missing {len(missing_regrid)} regrid parcels")

    # Filter index to only include records with both files
    print("\n5. Creating filtered index...")
    df_complete = df_subset.copy()
    df_complete['has_image'] = df_complete['image_filename'].apply(
        lambda x: (DEST_IMAGES / x).exists()
    )
    df_complete['has_regrid'] = df_complete['image_filename'].apply(
        lambda x: (DEST_REGRID / x.replace('.jpg', '_usa.json')).exists()
    )
    df_complete = df_complete[
        df_complete['has_image'] & df_complete['has_regrid']
    ].copy()
    df_complete = df_complete[['quote_id', 'image_filename', 'latitude', 'longitude']]

    # Save filtered index
    df_complete.to_csv(DEST_INDEX, index=False)
    print(f"   ✓ Saved index with {len(df_complete)} complete records")
    print(f"   ✓ {DEST_INDEX}\n")

    # Summary
    print(f"{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    print(f"Complete datasets:     {len(df_complete)}")
    print(f"Satellite images:      {len(list(DEST_IMAGES.glob('*.jpg')))}")
    print(f"Regrid parcels:        {len(list(DEST_REGRID.glob('*.json')))}")
    print(f"Index file:            {DEST_INDEX}")
    print(f"{'='*80}\n")

    # Show sample records
    if len(df_complete) > 0:
        print("Sample records:")
        print(df_complete.head(5).to_string(index=False))
        print()

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Test datasets copied successfully!")
    else:
        print("✗ Failed to copy test datasets")
