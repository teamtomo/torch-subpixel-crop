import torch
import einops
import pytest

from torch_subpixel_crop import subpixel_crop_2d


def test_subpixel_crop_single_2d():
    image = torch.zeros((10, 10))
    image[4:6, 4:6] = 1

    cropped_image = subpixel_crop_2d(
        image=image,
        positions=torch.tensor([5, 5]).float(),
        sidelength=4
    )
    assert cropped_image.shape == (4, 4)

    expected = torch.zeros((4, 4))
    expected[1:3, 1:3] = 1
    assert torch.allclose(cropped_image, expected)


def test_subpixel_crop_multi_2d():
    image = torch.zeros((10, 10))
    image[4:6, 4:6] = 1

    cropped_image = subpixel_crop_2d(
        image=image,
        positions=torch.tensor([[4, 4], [5, 5]]).float(),
        sidelength=4
    )
    assert cropped_image.shape == (2, 4, 4)

    expected_0 = torch.zeros((4, 4))
    expected_0[2:4, 2:4] = 1
    assert torch.allclose(cropped_image[0], expected_0)

    expected_1 = torch.zeros((4, 4))
    expected_1[1:3, 1:3] = 1
    assert torch.allclose(cropped_image[1], expected_1)


def test_subpixel_crop_multi_batch_2d():
    batch = 3
    positions = einops.repeat(
        torch.tensor([[4, 4], [5, 5]]).float(),
        'n yx -> n b yx', b=batch
    ).contiguous()  # clone so that we don't get a view
    positions[0, 0, 0] = 1  # change one of them to test if it works

    image = torch.zeros((2, 10, 10))
    with pytest.raises(ValueError):
        # mismatch in image and position batch should raise error
        cropped_image = subpixel_crop_2d(
            image=image,
            positions=positions,
            sidelength=4
        )

    image = torch.zeros((batch, 10, 10))
    image[..., 4:6, 4:6] = 1

    cropped_image = subpixel_crop_2d(
        image=image,
        positions=positions,
        sidelength=4
    )
    assert cropped_image.shape == (2, 3, 4, 4)

    expected_0 = torch.zeros((3, 4, 4))
    expected_0[1:, 2:4, 2:4] = 1
    assert torch.allclose(cropped_image[0], expected_0)

    expected_1 = torch.zeros((3, 4, 4))
    expected_1[:, 1:3, 1:3] = 1
    assert torch.allclose(cropped_image[1], expected_1)
