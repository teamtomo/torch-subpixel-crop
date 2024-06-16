"""Extract 2D/3D subimages with subpixel precision in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-subpixel-crop")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"
