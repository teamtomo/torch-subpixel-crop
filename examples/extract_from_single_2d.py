import numpy as np
import torch

from torch_subpixel_crop import subpixel_crop_2d
from skimage import data

image = torch.tensor(data.binary_blobs(length=512, n_dim=2)).float()
positions = torch.tensor(np.random.uniform(low=0, high=511, size=(100, 2))).float()

crops = subpixel_crop_2d(
    image=image,
    positions=positions,
    sidelength=32
)

# (100, 32, 32)
print(crops.shape)