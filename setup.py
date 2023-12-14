from setuptools import setup, find_packages

# Get the package name from the current directory name
package_name = "yolov4_tiny_pytorch"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(),
)