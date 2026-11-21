# interpret/methods/layer_inversion.py: layer inversion: a decoder per layer rebuilds the tile from the tokens of that layer (the question: README.md).
# It is a Probe (probe_base.py) with decoders as models, the image as target and the pixel MSE as metric: the same folds, grid, training loop and aggregation as the linear probe.

import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from timm.layers import build_sincos2d_pos_embed

from interpret.data import DeviceLoader
from interpret.methods.probe_base import Probe
from model import Block


# Decoder A: one linear head, shared by all patch tokens, turns each patch token alone into the pixels of its own patch (CLS is ignored); the patches are then put back into the image.
class LinearDecoder(torch.nn.Module):
    def __init__(self, token_channel_dim, num_patches, image_side_length, num_image_channels):
        super().__init__()
        self.num_patches = num_patches
        self.image_side_length = image_side_length
        
        self.patches_per_side = math.isqrt(num_patches)  # patches per row and per column
        assert self.patches_per_side ** 2 == num_patches, f"{num_patches} patches do not form a square grid"  # isqrt rounds down silently

        self.patch_side_length = image_side_length // self.patches_per_side  # ensured to divide cleanly
        assert image_side_length % self.patches_per_side == 0, f"the image side length {image_side_length} is not a multiple of the {self.patches_per_side} patches per side"  # a patch is a whole number of pixels
        
        self.num_image_channels = num_image_channels
        self.head = torch.nn.Linear(token_channel_dim, num_image_channels * self.patch_side_length * self.patch_side_length)  # turns one token into one patch

    def forward(self, tokens):  # (batch, 1 + num_patches, C) -> (batch, num_image_channels, H, W)
        assert tokens.shape[1:] == (1 + self.num_patches, self.head.in_features)
        patch_tokens = tokens[:, 1:, :]
        patch_pixels = self.head(patch_tokens)  # (batch, num_patches, p * p * num_image_channels)
        batch_size = tokens.shape[0]

        # arrange the fully flat 3rd dimension of patch_pixels into the required structure
        values_per_patch = patch_pixels.shape[2]  # p * p * num_image_channels
        patch_grid = patch_pixels.reshape(batch_size, self.patches_per_side, self.patches_per_side, values_per_patch)  # (batch, patch row, patch column, p * p * num_image_channels): the patches into a 2D grid
        pixel_rows = patch_grid.reshape(batch_size, self.patches_per_side, self.patches_per_side, self.patch_side_length, self.patch_side_length * self.num_image_channels)  # (batch, patch row, patch column, pixel row, p * num_image_channels): the pixels of a patch into a 2D grid
        patches = pixel_rows.reshape(batch_size, self.patches_per_side, self.patches_per_side, self.patch_side_length, self.patch_side_length, self.num_image_channels)  # (batch, patch row, patch column, pixel row, pixel column, channel): pull the channels out of the pixels and into its own dimension

        # permute the dimensions into the required order (s.t. the fuse in the next line is simple to implement)
        image = patches.permute(0, 5, 1, 3, 2, 4)  # (batch, channel, patch row, pixel row, patch column, pixel column)

        # fuse dimensions (patch row, pixel row) and dimensions (patch column, pixel column) together to obtain an image 
        return image.reshape(batch_size, self.num_image_channels, self.image_side_length, self.image_side_length)


# Decoder B: a fixed 2D sin-cos position embedding is added (CLS gets a zero row, as in MAE), then `depth` encoder-type blocks exchange information between all tokens (CLS included),
# then the head of decoder A is applied to the patch tokens.
class TransformerDecoder(LinearDecoder):
    def __init__(self, token_channel_dim, num_patches, image_side_length, num_image_channels, depth, head_dim, mlp_ratio, drop_path):
        super().__init__(token_channel_dim, num_patches, image_side_length, num_image_channels)
        assert token_channel_dim % head_dim == 0, f"the token size {token_channel_dim} is not a multiple of the head dimension {head_dim}"  # a whole number of attention heads
        num_heads = token_channel_dim // head_dim  # the C channels of a token are cut into num_heads slices of head_dim channels, one per attention head (exact, see the assert)
        blocks = []
        for _ in range(depth):
            blocks.append(Block(token_channel_dim, num_heads, mlp_ratio, drop_path))  # the encoder's blocks: head dimension 64 (6 heads at C = 384), MLP ratio 4, no drop-path
        self.blocks = torch.nn.Sequential(*blocks)

        # the position embedding (MAE): half of the channels encode the patch row, half the patch column, each as sines and cosines of geometrically spaced frequencies (from timm)
        patch_positions = build_sincos2d_pos_embed([self.patches_per_side, self.patches_per_side], token_channel_dim)  # (num_patches, C), row-major like the tokens
        cls_position = torch.zeros(1, token_channel_dim)  # (1, C) - zero vector for positional encoding of CLS token
        position_embedding = torch.cat([cls_position, patch_positions], dim=0)  # (1 + num_patches, C)
        self.register_buffer("position_embedding", position_embedding)

    def forward(self, tokens):  # (batch, 1 + num_patches, C) -> (batch, num_image_channels, H, W)
        positioned_tokens = tokens + self.position_embedding  # (batch, 1 + num_patches, C)
        mixed_tokens = self.blocks(positioned_tokens)  # apply transformer blocks
        return super().forward(mixed_tokens)  # apply linear head to tokens 


class LayerInversion(Probe):
    NAME = "layer inversion"
    METRICS = ("mse",)  # the pixel MSE over all pixels of the validation tiles
    BEST = min
    VARIANTS_KEY = "decoders"

    def __init__(self, activation_datasets, device, seed, method_cfg):
        super().__init__(activation_datasets, device, seed, method_cfg, variants=method_cfg["decoders"])
        
        first_dataset = list(activation_datasets.values())[0]
        example = first_dataset[0]  # one example as CPU tensors, only to read the shapes
        self.num_patches = example["patch_tokens"].shape[0]
        self.token_channel_dim = example["patch_tokens"].shape[1]
        self.num_image_channels = example["image"].shape[0]
        self.image_side_length = example["image"].shape[1]  # the images are square
        
        # the collage, written into the results by collect_evaluation_batch() during the run: originals (the collage rows, num_image_channels, H, W), the same tiles for every decoder and layer,
        # positions (the collage rows,), the random positions of the tiles within that batch, and reconstructions {(decoder, layer name): {setting name: (the collage rows, num_image_channels, H, W)}}
        self.results["collage"] = {"originals": None, "positions": None, "reconstructions": {}}
        
        self.results["briefing"] = self.print_briefing(first_dataset)

    # -------------------------------------------------------------------------------------------------
    # HOOKS
    # -------------------------------------------------------------------------------------------------

    # the variant of the layer inversion is the decoder. All decoders are trained on the same folds, stratified on one label
    def stratify_labels(self, variant):
        return self.labels[self._cfg["stratify"]]  # no target to stratify on (unlike in probe) so we stratify on a meaningful configured label

    # The statistics of the training fold: the mean and std of every token position and channel (the standardization: the scales differ strongly across layers, so one set of
    # learning rates only works for all of them after it). The tokens of the training fold of one layer (about 3 GB) are collected from the loader and held in memory at once.
    def fold_stats(self, dataset, variant, train_indices):
        loader = DeviceLoader(dataset, self.device, train_indices, self._cfg["batch_size"], num_workers=self._cfg["num_workers"])
        batches = []
        for batch in loader:
            batches.append(self._tokens_of(batch))  # (batch, 1 + num_patches, C)
        train_tokens = torch.cat(batches, dim=0)  # (num_train_tiles, 1 + num_patches, C)
        # one mean and std (over input samples) per token-position and within each token per channel (we will standardize each channel of each token individually)
        mean = train_tokens.mean(dim=0)  # (1 + num_patches, C)
        std = train_tokens.std(dim=0)  # (1 + num_patches, C)
        # DATA LEAKAGE PREVENTION: prepare_batch() standardizes every batch of both folds with these training statistics (no leakage of validation statistics into training set)
        return {"mean": mean, "std": std, "class_weights": None}  # there are no classes

    def make_model(self, variant, fold_statistics):
        if variant == "linear":
            return LinearDecoder(self.token_channel_dim, self.num_patches, self.image_side_length, self.num_image_channels)
        if variant == "transformer":
            transformer_cfg = self._cfg["transformer"]
            return TransformerDecoder(self.token_channel_dim, self.num_patches, self.image_side_length, self.num_image_channels, transformer_cfg["depth"], transformer_cfg["head_dim"], transformer_cfg["mlp_ratio"], transformer_cfg["drop_path"])
        raise KeyError(f"Unknown decoder '{variant}'; add it to make_model() of interpret/methods/layer_inversion.py")

    # The tokens of one batch of an ActivationDataset, CLS first: (batch, 1 + num_patches, C).
    def _tokens_of(self, batch):
        cls_token = batch["cls"].unsqueeze(1)  # (batch, 1, C)
        patch_tokens = batch["patch_tokens"]  # (batch, num_patches, C)
        return torch.cat([cls_token, patch_tokens], dim=1)

    # The standardization with the statistics of the training fold: the mean and std of every token position and channel of the (batch, 1 + num_patches, C) tokens.
    def _standardize(self, tokens, fold_statistics):
        assert tokens.shape[1:] == fold_statistics["mean"].shape == fold_statistics["std"].shape  # (1 + num_patches, C)
        return (tokens - fold_statistics["mean"]) / fold_statistics["std"]  # mean and std grids are broadcast along batch dimension

    # The standardized tokens and the target image of one batch of an ActivationDataset.
    def prepare_batch(self, batch, variant, fold_statistics):
        tokens = self._tokens_of(batch)
        standardized_tokens = self._standardize(tokens, fold_statistics)
        return standardized_tokens, batch["image"]  # (batch, 1 + num_patches, C), (batch, num_image_channels, H, W)

    # the same loss in training and validation
    def loss(self, outputs, targets, fold_statistics, train):
        assert outputs.shape == targets.shape  # (batch, num_image_channels, H, W)
        return F.mse_loss(outputs, targets)

    # The extra data of the final evaluation (the base tracks only the loss curves): the error of every validation tile for every model (one float per tile, the MSE over its pixels). The state of the pass of the last epoch is
    # the final one. The collage data (random validation tiles and their reconstructions, too big to keep for all validation tiles) is written straight into self.results, once, in the last epoch of fold 0.
    def start_evaluation_collection(self, num_models, fold_statistics):
        return {"tile_errors": [[] for _ in range(num_models)]}  # per model, the tile errors of every batch

    def collect_evaluation_batch(self, evaluation_state, outputs_per_model, targets, variant, layer_name, fold, setting_names, last_epoch):
        # collage: from the first validation batch of the last epoch of fold 0, once per decoder and layer
        if last_epoch and fold == 0 and (variant, layer_name) not in self.results["collage"]["reconstructions"]:
            # fetch random input images for collage
            num_collage_rows = self._cfg["num_collage_rows"]
            # random positions within the batch, drawn once and reused for every decoder and layer, so that all columns of the collage show the same tiles (the folds are shared, see Probe)
            if self.results["collage"]["positions"] is None:
                permutation = torch.randperm(targets.shape[0])
                self.results["collage"]["positions"] = permutation[:num_collage_rows]
            collage_positions = self.results["collage"]["positions"].to(targets.device)
            selected_images = targets[collage_positions, :, :, :]  # (collage rows, num_image_channels, H, W)
            self.results["collage"]["originals"] = selected_images.cpu()  # to the host, the plot needs numpy
            # fetch same random indices for reconstructed images
            reconstructions = {}
            for model_index, setting_name in enumerate(setting_names):
                selected_outputs = outputs_per_model[model_index][collage_positions, :, :, :]  # (collage rows, num_image_channels, H, W)
                reconstructions[setting_name] = selected_outputs.cpu()
            self.results["collage"]["reconstructions"][(variant, layer_name)] = reconstructions

        # MSE plot
        for model_index, outputs in enumerate(outputs_per_model):
            assert outputs.shape == targets.shape  # (batch, num_image_channels, H, W)
            # the MSE is recomputed here instead of reusing the loss computed in the base class: that one is a scalar per batch (for the loss curves), while the exact final metric needs one error per tile
            squared_error = F.mse_loss(outputs, targets, reduction="none")  # (batch, num_image_channels, H, W), the squared error of every pixel value (the same function as the loss)
            # the batch dimension is kept (one error per tile, not per batch): the mean over all collected tiles then weights every tile equally, so the smaller last batch counts for its real size
            # and no count has to be tracked (a mean of the batch means would give it the weight of a full batch)
            tile_error = squared_error.mean(dim=(1, 2, 3))  # (batch,), the MSE of every tile
            evaluation_state["tile_errors"][model_index].append(tile_error)

    def final_evaluation(self, evaluation_state, variant, layer_name, fold, setting_names, fold_statistics):
        results = {}
        for model_index, setting_name in enumerate(setting_names):
            tile_errors = torch.cat(evaluation_state["tile_errors"][model_index])  # (n_val,)
            results[setting_name] = {"mse": tile_errors.mean().item()}
        return results

    # -------------------------------------------------------------------------------------------------
    # BRIEFING + PLOTTING
    # -------------------------------------------------------------------------------------------------

    def _format_counts(self, counts, class_values):
        total = sum(counts.values())
        parts = []
        for class_value in class_values:
            parts.append(f"{class_value}: {counts[class_value]} ({100 * counts[class_value] / total:.0f}%)")
        return ", ".join(parts)

    # One printout in an ASCII box, once, before the training starts: the plan and the folds. Returns the same numbers as a dict, which is saved in the results.
    def print_briefing(self, first_dataset):
        num_grid_points = len(self._cfg["lrs"]) * len(self._cfg["weight_decays"])
        group_field = self._cfg["leakage_group"]
        stratify_field = self._cfg["stratify"]
        group_names = self.label_names[group_field]

        num_trainings = len(self.variants) * len(self.layer_names) * self._cfg["folds"] * num_grid_points
        total_steps = 0  # optimizer steps of all trainings
        for decoder in self.variants:
            for train_indices, _ in self.folds[decoder]:
                total_steps += len(self.layer_names) * num_grid_points * self._cfg["epochs"] * math.ceil(len(train_indices) / self._cfg["batch_size"])

        lines = ["LAYER INVERSION: run briefing", "", "PLAN"]
        lines.append(f"decoders: {', '.join(self.variants)} (transformer depth {self._cfg['transformer']['depth']}) | layers: {', '.join(self.layer_names)} | input: [CLS, patch tokens], {self.num_patches + 1} tokens of {self.token_channel_dim} values | target: the tile, {self.num_image_channels} x {self.image_side_length} x {self.image_side_length}, in [0, 1]")
        lines.append(f"examples: {len(first_dataset)} tiles in {len(group_names)} groups of {group_field} | folds: {self._cfg['folds']} (StratifiedGroupKFold on {group_field}, stratified on {stratify_field}, seed {self.seed})")
        lines.append(f"grid: {len(self._cfg['lrs'])} learning rates x {len(self._cfg['weight_decays'])} weight decays = {num_grid_points} settings | {self._cfg['epochs']} epochs, batch size {self._cfg['batch_size']}")
        lines.append(f"total: {num_trainings} trainings, about {total_steps / 1e6:.2f} M optimizer steps")
        lines.append("")
        lines.append("DATA")
        fold_stats = []  # the folds are the same for every decoder
        for fold, (train_indices, val_indices) in enumerate(self.folds[self.variants[0]]):
            train_counts = Counter(self.label_names[stratify_field][class_index] for class_index in self.labels[stratify_field][train_indices].tolist())
            val_counts = Counter(self.label_names[stratify_field][class_index] for class_index in self.labels[stratify_field][val_indices].tolist())
            fold_stats.append({"classes": {"train": dict(train_counts), "val": dict(val_counts)}})
            lines.append(
                f"fold {fold} | train {len(train_indices)} tiles: {self._format_counts(train_counts, self.label_names[stratify_field])}"
                f" | val {len(val_indices)} tiles: {self._format_counts(val_counts, self.label_names[stratify_field])}"
            )

        width = max(len(line) for line in lines)
        print("+" + "-" * (width + 2) + "+")
        for line in lines:
            print("| " + line.ljust(width) + " |")
        print("+" + "-" * (width + 2) + "+")

        return {"folds": fold_stats, "num_trainings": num_trainings, "total_optimizer_steps": total_steps}

    # Headline plot: the pixel MSE at the best grid point of every layer, one line per decoder. The collage: random validation tiles of the first fold next to their reconstructions from every
    # layer (best grid point). The debug_ plots are the loss curves. Returns {file name: figure}.
    def plot(self):
        figures = {}

        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        for decoder in self.variants:
            means = []
            stds = []
            for layer_name in self.layer_names:
                layer_results = self.results[self.VARIANTS_KEY][decoder][layer_name]
                best_setting = layer_results["best_setting"]
                best = layer_results["grid"][best_setting]
                means.append(best["mean"]["mse"])
                stds.append(best["std"]["mse"])
            axis.errorbar(range(len(self.layer_names)), means, yerr=stds, marker="o", capsize=3, label=f"{decoder} decoder")
        axis.set_xticks(range(len(self.layer_names)), self.layer_names)
        axis.set_xlabel("layer (increasing depth)")
        axis.set_ylabel("pixel MSE on the validation tiles (mean and std over folds)")
        axis.set_title("Layer inversion: error per layer")
        axis.legend(fontsize=8)
        figures["error"] = figure

        columns = 1 + len(self.variants) * len(self.layer_names)  # the original, then every decoder at every layer
        num_rows = self.results["collage"]["originals"].shape[0]  # the num_collage_rows of the config, chosen in collect_evaluation_batch()
        figure = Figure(figsize=(1.6 * columns, 1.6 * num_rows))
        axes = figure.subplots(num_rows, columns, squeeze=False)
        for row in range(num_rows):
            original = self.results["collage"]["originals"][row, :, :, :]  # (3, H, W)
            original_channels_last = original.permute(1, 2, 0)  # (H, W, 3)
            axes[row][0].imshow(original_channels_last.clamp(0, 1).numpy())  # clipped to [0, 1] for display
            column = 1
            for decoder in self.variants:
                for layer_name in self.layer_names:
                    best_setting = self.results[self.VARIANTS_KEY][decoder][layer_name]["best_setting"]
                    reconstructions = self.results["collage"]["reconstructions"][(decoder, layer_name)]  # {setting name: (collage rows, 3, H, W)}
                    reconstruction = reconstructions[best_setting][row, :, :, :]  # (3, H, W)
                    reconstruction_channels_last = reconstruction.permute(1, 2, 0)  # (H, W, 3)
                    axes[row][column].imshow(reconstruction_channels_last.clamp(0, 1).numpy())  # clipped to [0, 1] for display
                    column += 1
        axes[0][0].set_title("original", fontsize=7)
        column = 1
        for decoder in self.variants:
            for layer_name in self.layer_names:
                axes[0][column].set_title(f"{decoder}\n{layer_name}", fontsize=7)
                column += 1
        for axis in axes.flat:
            axis.axis("off")
        figures["collage"] = figure

        for decoder in self.variants:
            for layer_name in self.layer_names:
                layer_results = self.results[self.VARIANTS_KEY][decoder][layer_name]

                # the curve grid: one small plot per grid point with the train and validation loss per epoch (mean over the folds, band from the min to the max fold)
                figure = Figure(figsize=(9, 7))
                axes = figure.subplots(len(self._cfg["lrs"]), len(self._cfg["weight_decays"]), sharex=True, sharey=True, squeeze=False)
                for row_index, lr in enumerate(self._cfg["lrs"]):
                    for column_index, weight_decay in enumerate(self._cfg["weight_decays"]):
                        setting_name = self._setting_name(lr, weight_decay)
                        curves = layer_results["grid"][setting_name]["curves"]
                        axis = axes[row_index][column_index]
                        for curve_name, label in (("train_loss", "train"), ("val_loss", "validation")):
                            per_fold = np.array(curves[curve_name])  # (folds, epochs)
                            epochs = range(1, per_fold.shape[1] + 1)
                            axis.plot(epochs, per_fold.mean(axis=0), label=label)
                            axis.fill_between(epochs, per_fold.min(axis=0), per_fold.max(axis=0), alpha=0.2)
                        axis.set_title(setting_name, fontsize=7)
                        axis.tick_params(labelsize=6)
                axes[0][0].legend(fontsize=6)
                figure.supxlabel("epoch")
                figure.supylabel("pixel MSE (train and validation)")
                figure.suptitle(f"{decoder}, {layer_name}: loss curves per grid point")
                figures[f"debug_{decoder}_{layer_name}_curves"] = figure

        return figures
