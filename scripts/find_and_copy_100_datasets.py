"""
Find and copy exactly 100 datasets where both satellite image and regrid parcel exist.
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

# Target number
TARGET = 100


def main():
    """Find and copy exactly 100 complete datasets."""
    print(f"{'='*80}")
    print(f"Finding {TARGET} Complete Test Datasets")
    print(f"{'='*80}\n")

    # Create destination directories
    print("1. Creating destination directories...")
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_REGRID.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {DEST_IMAGES}")
    print(f"   ✓ {DEST_REGRID}\n")

    # Load full index
    print("2. Loading training index...")
    df = pd.read_csv(SOURCE_INDEX)
    print(f"   ✓ Loaded {len(df)} records\n")

    # Find complete datasets
    print(f"3. Scanning for complete datasets (target: {TARGET})...")
    complete_records = []

    for idx, row in df.iterrows():
        if len(complete_records) >= TARGET:
            break

        image_filename = row['image_filename']

        # Try multiple regrid naming patterns
        base_name = image_filename.replace('.jpg', '')
        possible_regrid_names = [
            f"{base_name}_usa.json",
            f"{base_name}.json",
        ]

        src_image = SOURCE_IMAGES / image_filename
        regrid_file = None

        # Check if image exists
        if not src_image.exists():
            continue

        # Find matching regrid file
        for regrid_name in possible_regrid_names:
            src_regrid = SOURCE_REGRID / regrid_name
            if src_regrid.exists():
                regrid_file = regrid_name
                break

        if regrid_file:
            complete_records.append({
                'quote_id': row['quote_id'],
                'image_filename': image_filename,
                'regrid_filename': regrid_file,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
            })

            if len(complete_records) % 10 == 0:
                print(f"   Found {len(complete_records)}/{TARGET} complete datasets...")

    print(f"\n   ✓ Found {len(complete_records)} complete datasets\n")

    if len(complete_records) < TARGET:
        print(f"   ⚠ Only found {len(complete_records)}/{TARGET} complete datasets")

    # Copy files
    print(f"4. Copying {len(complete_records)} datasets...")
    for i, record in enumerate(complete_records):
        # Source paths
        src_image = SOURCE_IMAGES / record['image_filename']
        src_regrid = SOURCE_REGRID / record['regrid_filename']

        # Destination paths
        dst_image = DEST_IMAGES / record['image_filename']
        dst_regrid = DEST_REGRID / record['regrid_filename']

        # Copy files
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_regrid, dst_regrid)

        if (i + 1) % 10 == 0:
            print(f"   Copied {i + 1}/{len(complete_records)} datasets...")

    print("\n   ✓ All files copied successfully\n")

    # Save index
    print("5. Creating index file...")
    df_index = pd.DataFrame(complete_records)
    df_index = df_index[['quote_id', 'image_filename', 'latitude', 'longitude']]
    df_index.to_csv(DEST_INDEX, index=False)
    print(f"   ✓ Saved index with {len(df_index)} records")
    print(f"   ✓ {DEST_INDEX}\n")

    # Summary
    print(f"{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    print(f"Complete datasets:     {len(df_index)}")
    print(f"Satellite images:      {len(list(DEST_IMAGES.glob('*.jpg')))}")
    print(f"Regrid parcels:        {len(list(DEST_REGRID.glob('*.json')))}")
    print(f"Index file:            {DEST_INDEX}")
    print(f"{'='*80}\n")

    # Show sample records
    if len(df_index) > 0:
        print("Sample records:")
        print(df_index.head(10).to_string(index=False))
        print()

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Test datasets prepared successfully!")
    else:
        print("✗ Failed to prepare test datasets")
