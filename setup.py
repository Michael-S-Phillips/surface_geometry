# setup.py
from setuptools import setup, find_packages

setup(
    name="surface_geometry",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "rasterio",
        "geopandas",
        "pillow",
        "statsmodels"
    ],
    author="Python adaptation",
    author_email="",
    description="A Python adaptation of surface geometry for analyzing habitat complexity",
    keywords="surface, geometry, fractal, rugosity, biodiversity",
    url="",
)