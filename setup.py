"""
Setup script for Pose Estimation Module
Professional installation and configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pose-estimation-module",
    version="1.0.0",
    author="Shakeel Rana",
    author_email="shakeelrana6240@gmail.com",
    description="Real-time multi-person pose estimation with human activity detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShakeelRana624/RealTime-Pose-Estimation-Engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "pose-estimation=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/ShakeelRana624/RealTime-Pose-Estimation-Engine/issues",
        "Source": "https://github.com/ShakeelRana624/RealTime-Pose-Estimation-Engine",
        "Documentation": "https://github.com/ShakeelRana624/RealTime-Pose-Estimation-Engine/wiki",
    },
    keywords=[
        "pose estimation",
        "human activity detection", 
        "computer vision",
        "yolov8",
        "machine learning",
        "artificial intelligence",
        "opencv",
        "pytorch",
    ],
)
