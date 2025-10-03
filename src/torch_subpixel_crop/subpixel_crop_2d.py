import einops
import torch
from torch.nn import functional as F
from torch_fourier_shift import fourier_shift_image_2d
from torch_grid_utils import coordinate_grid

from torch_subpixel_crop.dft_utils import dft_center
from torch_subpixel_crop.grid_sample_utils import array_to_grid_sample


def subpixel_crop_2d(
        image: torch.Tensor,
        positions: torch.Tensor,
        sidelength: int,
):
    """Extract square patches from 2D images with subpixel precision.

    Parameters
    ----------
    image : torch.Tensor
        `(b, h, w)` or `(h, w)` array of 2D images.
    positions : torch.Tensor
        `(..., b, 2)` or `(..., 2)` array of coordinates for patch centers.
    sidelength : int
        Sidelength of square patches extracted from `images`.

    Returns
    -------
    patches : torch.Tensor
        `(..., b, sidelength, sidelength)` or `(..., sidelength, sidelength)`
        array of patches.
    """
    # Handle unbatched input
    if image.ndim == 2:
        input_images_are_batched = False
        image = einops.rearrange(image, 'h w -> 1 h w')
        positions = einops.rearrange(positions, '... yx -> ... 1 yx')
    else:
        input_images_are_batched = True

    # Flatten batch dimensions
    positions, ps = einops.pack([positions], pattern='* batch yx')

    # Process ALL images at once (no loop!)
    patches = _extract_patches_batched(
        images=image,  # (batch, h, w)
        positions=positions,  # (n_positions, batch, 2)
        output_image_sidelength=sidelength
    )

    # Restore original shape
    [patches] = einops.unpack(patches, pattern='* batch h w', packed_shapes=ps)

    if not input_images_are_batched:
        patches = einops.rearrange(patches, '... 1 h w -> ... h w')

    return patches


def _extract_patches_batched(
        images: torch.Tensor,  # (batch, h, w)
        positions: torch.Tensor,  # (n_positions, batch, 2) yx
        output_image_sidelength: int,
) -> torch.Tensor:  # (n_positions, batch, ph, pw)
    batch, h, w = images.shape
    n_positions, batch_check, _ = positions.shape
    assert batch == batch_check

    # Find integer positions and shifts
    integer_positions = torch.round(positions)
    shifts = integer_positions - positions  # (n_positions, batch, 2)

    # Generate coordinate grid
    ph = pw = output_image_sidelength
    center = dft_center((ph, pw), rfft=False, fftshifted=True,
                        device=images.device)
    grid = coordinate_grid(
        image_shape=(ph, pw),
        center=center,
        device=images.device
    )  # (ph, pw, 2)

    # Broadcast grid for all positions and batches
    grid = einops.rearrange(grid, 'ph pw yx -> 1 1 ph pw yx')
    integer_positions = einops.rearrange(
        integer_positions, 'n_pos batch yx -> n_pos batch 1 1 yx'
    )
    grid = grid + integer_positions  # (n_pos, batch, ph, pw, 2)

    # Flatten for grid_sample
    grid_flat = einops.rearrange(grid,
                                 'n_pos batch ph pw yx -> (n_pos batch) ph pw yx')
    images_repeated = einops.repeat(
        images, 'batch h w -> (n_pos batch) 1 h w', n_pos=n_positions
    )

    # Extract all patches at once
    patches = F.grid_sample(
        input=images_repeated,
        grid=array_to_grid_sample(grid_flat, array_shape=(h, w)),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )  # (n_pos * batch, 1, ph, pw)

    patches = einops.rearrange(
        patches, '(n_pos batch) 1 ph pw -> (n_pos batch) ph pw',
        n_pos=n_positions, batch=batch
    )

    # Phase shift all at once
    shifts_flat = einops.rearrange(shifts,
                                   'n_pos batch yx -> (n_pos batch) yx')
    patches = fourier_shift_image_2d(image=patches, shifts=shifts_flat)

    # Reshape to output
    patches = einops.rearrange(
        patches, '(n_pos batch) ph pw -> n_pos batch ph pw',
        n_pos=n_positions, batch=batch
    )

    return patches