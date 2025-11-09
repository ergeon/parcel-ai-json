"""Tree detection service using detectree library via Docker.

Due to macOS compatibility issues with detectree's C extensions, this service
runs detectree in a Docker container with Linux.
"""

from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import subprocess
import json
import tempfile


@dataclass
class TreeDetection:
    """Tree detection results."""

    # Tree coverage statistics
    tree_pixel_count: int
    total_pixels: int
    tree_coverage_percent: float

    # Image dimensions
    width: int
    height: int

    # Tree mask image path (optional)
    tree_mask_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "tree_pixel_count": self.tree_pixel_count,
            "total_pixels": self.total_pixels,
            "tree_coverage_percent": self.tree_coverage_percent,
            "width": self.width,
            "height": self.height,
        }
        if self.tree_mask_path:
            result["tree_mask_path"] = self.tree_mask_path
        return result


class TreeDetectionService:
    """Tree detection service using detectree in Docker.

    NOTE: Requires Docker to be installed and running.
    Detectree has macOS compatibility issues, so we run it in a Linux container.
    """

    def __init__(
        self, docker_image: str = "parcel-tree-detector", save_mask: bool = True
    ):
        """Initialize tree detection service.

        Args:
            docker_image: Name of Docker image to use (default: parcel-tree-detector)
            save_mask: Whether to save tree mask as PNG for visualization
        """
        self.docker_image = docker_image
        self.save_mask = save_mask

    def detect_trees(self, satellite_image: Dict) -> TreeDetection:
        """Detect trees in satellite image using Docker.

        Args:
            satellite_image: Dict with keys:
                - path: Path to satellite image file

        Returns:
            TreeDetection with tree coverage statistics

        Raises:
            RuntimeError: If Docker is not available or detection fails
        """
        img_path = Path(satellite_image["path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Prepare output path for tree mask
        tree_mask_path = None
        if self.save_mask:
            tree_mask_path = str(img_path.parent / f"{img_path.stem}_tree_mask.png")

        # Run detectree in Docker container
        try:
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{img_path.parent}:/images",
                self.docker_image,
                "python",
                "-c",
                f"""
import detectree as dtr
import numpy as np
from PIL import Image
import json

# Load and convert to RGB (images are indexed color)
img = Image.open('/images/{img_path.name}')
img_rgb = img.convert('RGB')

# Save temporarily
img_rgb.save('/tmp/temp_rgb.jpg')

# Run detectree
clf = dtr.Classifier()
y_pred = clf.predict_img('/tmp/temp_rgb.jpg')

# Save tree mask as PNG with green trees on transparent background
# Create RGBA image - green (0,255,0) where trees, transparent elsewhere
mask_rgba = np.zeros((y_pred.shape[0], y_pred.shape[1], 4), dtype=np.uint8)
mask_rgba[y_pred == 1] = [0, 255, 0, 180]  # Green with 70% opacity
mask_img = Image.fromarray(mask_rgba, 'RGBA')
mask_img.save('/images/{img_path.stem}_tree_mask.png')

# Output results as JSON
result = {{
    'tree_pixels': int(np.sum(y_pred)),
    'total_pixels': int(y_pred.size),
    'width': int(y_pred.shape[1]),
    'height': int(y_pred.shape[0]),
    'coverage_percent': float(100 * np.sum(y_pred) / y_pred.size)
}}
print(json.dumps(result))
""",
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Tree detection failed: {result.stderr}")

            # Parse JSON output from last line
            output_lines = result.stdout.strip().split("\n")
            json_output = output_lines[-1]
            data = json.loads(json_output)

            return TreeDetection(
                tree_pixel_count=data["tree_pixels"],
                total_pixels=data["total_pixels"],
                tree_coverage_percent=data["coverage_percent"],
                width=data["width"],
                height=data["height"],
                tree_mask_path=tree_mask_path if self.save_mask else None,
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError("Tree detection timed out after 60 seconds")
        except FileNotFoundError:
            raise RuntimeError(
                "Docker not found. Please install Docker to use tree detection."
            )
        except Exception as e:
            raise RuntimeError(f"Tree detection failed: {e}")
