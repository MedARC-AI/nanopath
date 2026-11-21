# interpret/methods/attention_map.py: plots raw CLS-to-patch attention maps for each head of the last block.

import math

import torch
from matplotlib.figure import Figure
from tqdm import tqdm

from interpret.utils import strided_indices


class AttentionMap:
    def __init__(self, dataset, transform, model, device, cfg):
        self._dataset = dataset
        self._transform = transform
        self._model = model
        self._device = device

        self._tile_indices = strided_indices(len(dataset), cfg["num_tiles"])
        self.results = None

    # assert the hook contract for 'attn' (see interpret/README.md): one tile, batch dim first, a square number of patches
    def _assert_attn(self, attn):
        assert attn.ndim == 3 and attn.shape[0] == 1 and math.isqrt(attn.shape[2]) ** 2 == attn.shape[2], f"'attn' must be (1, heads, N_patches) with N_patches a square, got {tuple(attn.shape)}"

    def run(self):
        storage = {}

        # find pytorch name of last block
        layer_names = self._model.register_interpretability_hooks(storage)
        block_names = [k for k in layer_names if k.startswith("block_")]
        last_block_name = block_names[-1]

        tiles_results = {}
        with torch.inference_mode():
            for idx in tqdm(self._tile_indices, desc="Attention maps"):
                # get and preprocess input image
                image, _ = self._dataset[idx]
                transformed = self._transform(image)
                normalized_tile = transformed[1].unsqueeze(0)  # (1, 3, H, W), add batch_dim for model forward pass

                # forward pass
                storage.clear()
                self._model(normalized_tile)

                if "attn" not in storage[last_block_name]:
                    print(f"AttentionMap: '{last_block_name}' has no 'attn' in storage; skipping.", flush=True)
                    return  # user has not implement attention capturing in hook

                attn = storage[last_block_name]["attn"]
                self._assert_attn(attn)
                attn = attn.squeeze(0)  # remove singleton batch_dim again

                rgb_image = transformed[0]  # (3, H, W) in [0, 1]
                tiles_results[idx] = {"rgb": rgb_image, "attn": attn}

        self.results = {"tiles": tiles_results, "tile_indices": self._tile_indices}

    def plot(self):
        if not self.results:
            return {}

        figures = {}
        for idx in self.results["tile_indices"]:
            data = self.results["tiles"][idx]
            rgb = data["rgb"].cpu().permute(1, 2, 0).numpy()
            attn = data["attn"].cpu()  # (heads, num_patches)
            heads, num_patches = attn.shape
            grid_size = int(num_patches ** 0.5)
            attn_grid = attn.reshape(heads, grid_size, grid_size).numpy()

            figure = Figure(figsize=(max(10, (heads + 1) * 2), 2.5))
            axes = figure.subplots(1, heads + 1)

            # Input tile
            axes[0].imshow(rgb)
            axes[0].set_title("Input")

            # Attention map per head with shared color limits
            vmin, vmax = attn_grid.min(), attn_grid.max()
            for h in range(heads):
                im = axes[h + 1].imshow(attn_grid[h], cmap="Blues", vmin=vmin, vmax=vmax)
                axes[h + 1].set_title(f"Head {h}")
                if h == heads - 1:
                    figure.colorbar(im, ax=axes[h + 1], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.axis("off")

            figure.suptitle(f"Tile {idx}: CLS attention, last block (not a measure of importance)", fontsize=11, y=1.02)
            figure.tight_layout()
            figures[f"tile_{idx}"] = figure

        return figures