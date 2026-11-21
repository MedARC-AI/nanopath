# interpret/utils.py: helpers of the interpretability rig, shared by the methods and the runner: strided_indices picks evenly spaced indices, tissue_masks finds the tissue pixels of a batch of tiles.

import torch

from prepare import tissue_mask


def strided_indices(total_count, sample_count):
    stride = total_count // sample_count
    return list(range(0, sample_count * stride, stride))


# The tissue mask of a batch of tiles with the repo's rule (tissue_mask in prepare.py, which takes one tile at a time). The rule thresholds the saturation of RGB values in [0, 1], so a batch that is normalized, in 0-255
# or without the batch dimension would give a meaningless mask without any error: the preconditions are asserted here, every use of the tissue rule in the rig goes through this function.
def tissue_masks(images):  # (B, 3, H, W) floats in [0, 1] -> (B, H, W) booleans
    assert images.ndim == 4 and images.shape[1] == 3, f"expected tiles of shape (B, 3, H, W), got {tuple(images.shape)}"
    assert images.dtype.is_floating_point, f"expected float tiles in [0, 1], got {images.dtype}"
    assert images.min() >= 0.0 and images.max() <= 1.0, f"the tissue rule needs RGB in [0, 1] (not normalized, not 0-255), got the range [{images.min().item():.3g}, {images.max().item():.3g}]"
    return torch.stack([tissue_mask(tile) for tile in images])
