"""Setup configuration for parcel-ai-json package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="parcel-ai-json",
    version="0.1.0",
    author="Ergeon",
    author_email="dev@ergeon.com",
    description="AI/ML extensions for parcel-geojson - vehicle detection and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ergeon/parcel-ai-json",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "parcel-geojson @ git+https://github.com/ergeon/parcel-geojson.git",  # Core package
        "ultralytics>=8.0.0",      # YOLOv8
        "pillow>=9.0.0",           # Image processing
        "numpy>=1.20.0",
        "torch>=2.0.0",            # PyTorch for YOLO
        "torchvision>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    package_data={
        "parcel_ai_json": [
            "models/*.pt",  # Include YOLO model files
        ],
    },
    include_package_data=True,
)
