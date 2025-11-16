"""Setup configuration for parcel-ai-json package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="parcel-ai-json",
    version="0.1.0",
    author="Ergeon Engineers",
    author_email="engineers@ergeon.com",
    description="AI/ML extensions for parcel-geojson - vehicle detection and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ergeon/parcel-ai-json",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "scripts", "parcel_ai_json.models"]),
    license="Other/Proprietary License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",      # YOLOv8
        "pillow>=9.0.0",           # Image processing
        "numpy>=1.20.0",
        "torch>=2.0.0",            # PyTorch for YOLO
        "torchvision>=0.15.0",
        "pyproj>=3.0.0",           # Geodesic coordinate transformations
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "deploy": [
            "twine>=6.0.1",
            "wheel>=0.37.1",
        ],
    },
    # Models are auto-downloaded by ultralytics on first use to ~/.ultralytics/
    # Do not bundle models in package to keep package size small
    include_package_data=False,
)
