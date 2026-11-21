# interpret/methods/relax.py: RELAX: which regions of an image its final embedding depends on, measured by how similar the embeddings of randomly masked copies of the image stay (the question: README.md).
# Implements the equations of Wickstrøm et al. (RELAX, Eq. 2 and 9) with the masks of RISE (Petsiuk et al.), and makes one figure per analysed image: the input, the importance, the uncertainty and the thresholded importance.

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torchvision import transforms
from tqdm import tqdm

from interpret.utils import strided_indices


class RELAX:
    def __init__(self, dataset, transform, model, device, method_cfg):
        self._dataset = dataset  # the dataset the extraction samples from (data.py): RELAX runs the model itself, so it reads the images, not the extraction
        self._transform = transform  # the extraction's transform: an image -> [the tile in [0, 1], the normalised model input], both on the device
        self._model = model  # in eval mode, on the device
        self._device = device
        self._cfg = method_cfg

        # the images to analyse: num_images tiles evenly spaced through the dataset (a fixed stride, no randomness), so that every run and every checkpoint is analysed on the same tiles
        num_images = method_cfg["num_images"]
        assert num_images <= len(dataset), f"relax.num_images is {num_images}, but the dataset has only {len(dataset)} images"
        self._image_indices = strided_indices(len(dataset), num_images)

        self.results = None

    # The masks of RISE as RELAX describes them (Sec. 3.1): a random grid of 0s and 1s (1 = visible), upscaled bilinearly to one grid square more than the image (soft values in [0, 1]),
    # then an image-sized window is cut out at a random position, which shifts the grid randomly against the image.
    def _draw_masks(self, num_masks, image_height, image_width):  # -> (num_masks, 1, H, W)
        grid_side_length = self._cfg["grid_side_length"]
        visible_probability = self._cfg["visible_probability"]
        # bernoulli draw for every position 
        # add 1 as second dimension s.t. later the mask is broadcast over the image channel dimension
        random_values = torch.rand(num_masks, 1, grid_side_length, grid_side_length, device=self._device)
        is_visible = random_values < visible_probability  # (num_masks, 1, grid side length, grid side length) booleans
        grid = is_visible.float()  # turn bool into 0.0 or 1.0

        # upscale: a cell of the upscaled mask is the image size divided by the grid size (the paper takes the floor, the assert makes it exact), and the upscaled mask has one cell more than the grid in each direction
        assert image_height % grid_side_length == 0 and image_width % grid_side_length == 0, f"the image ({image_height} x {image_width}) is not a multiple of relax.grid_side_length ({grid_side_length}) in both directions"
        cell_height = image_height // grid_side_length  # integer division not because of rest (no rest asserted above) but to obtain an integer height
        cell_width = image_width // grid_side_length
        upscaled_height = (grid_side_length + 1) * cell_height
        upscaled_width = (grid_side_length + 1) * cell_width
        upscaled = F.interpolate(grid, size=(upscaled_height, upscaled_width), mode="bilinear")  # (num_masks, 1, upscaled H, upscaled W), soft values in [0, 1]

        # crop: an image-sized window at a random position where it fits, drawn per mask (torchvision's RandomCrop, which draws from torch's global generator, seeded in main())
        random_crop = transforms.RandomCrop((image_height, image_width))
        masks = torch.empty(num_masks, 1, image_height, image_width, device=self._device)
        for mask_index in range(num_masks):  # apply crop serially s.t. we get a different random crop for each mask
            masks[mask_index, :, :, :] = random_crop(upscaled[mask_index, :, :, :])  # (1, H, W)
        return masks

    # The final embedding of a batch of normalised images: the CLS token and all patch tokens of the model output, concatenated into one vector per image, so that a strongly masked patch
    # shows up in the similarity. The registers are left out.
    def _embed(self, input_images):  # (batch, 3, H, W) -> (batch, C + num_patches * C)
        outputs = self._model(input_images)
        if not isinstance(outputs, dict) or "cls" not in outputs or "patches" not in outputs:
            raise AttributeError("model forward() must return a dict with 'cls' and 'patches' (see interpret/README.md)")
        cls_token = outputs["cls"]  # (batch, C)
        patch_tokens = outputs["patches"]  # (batch, num_patches, C)
        batch_size = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        channel_dim = patch_tokens.shape[2]
        flat_patch_tokens = patch_tokens.reshape(batch_size, num_patches * channel_dim)  # (batch, num_patches * C), the token and channel dimensions are collapsed here
        return torch.cat([cls_token, flat_patch_tokens], dim=1)

    # For every image: draw the masks, embed the unmasked and the masked images, score every mask (pass 1), then compute the three maps from the stored masks and scores (pass 2).
    def run(self):
        num_masks = self._cfg["num_masks"]
        mask_batch_size = self._cfg["mask_batch_size"]
        uncertainty_threshold_scale = self._cfg["uncertainty_threshold_scale"]  # gamma of U-RELAX

        maps_per_image = {}  # {image index: {"image", "importance", "uncertainty", "thresholded_importance"}}, (3, H, W) and (H, W) CPU tensors
        with torch.inference_mode():
            for image_index in tqdm(self._image_indices, desc="RELAX"):
                # the image as the extraction prepares it: the tile in [0, 1] (shown in the figure) and the normalised model input, the last tensor before the model, which the masks act on
                # so that the shift of the input distribution stays minimal
                image, _ = self._dataset[image_index]  # (PIL image, metadata), the metadata is not needed
                transformed = self._transform(image)
                # rgb image is only used for the final plots (but not for the RELAX computation)
                rgb_image = transformed[0]  # (3, H, W) in [0, 1]
                normalized_tile = transformed[1]  # (3, H, W)
                normalized_image = normalized_tile.unsqueeze(0)  # (1, 3, H, W), a batch of one
                image_height = normalized_image.shape[2]  
                image_width = normalized_image.shape[3]

                # masks: drawn from the generators seeded once in main() (the seeding policy in interpret.py)
                masks = self._draw_masks(num_masks, image_height, image_width)
                assert masks.shape == (num_masks, 1, image_height, image_width)

                # the score of every mask, the cosine similarity between the embeddings of the unmasked and the masked image; the masks go through the model in chunks of mask_batch_size to bound the GPU memory
                unmasked_embedding = self._embed(normalized_image)  # (1, D)
                scores_per_batch = []
                for chunk_start in range(0, num_masks, mask_batch_size):
                    chunk_masks = masks[chunk_start:chunk_start + mask_batch_size, :, :, :]  # (chunk, 1, H, W)
                    # * is element wise multiplication, on the NORMALISED image: in the space of the normlaized images 0 represents the dataset-mean colour of the training config (about 0.485, 0.456, 0.406), 
                    # not black, so in pixel space a mask value m turns a pixel x into mean + m * (x - mean), a cross-fade from the pixel (m = 1) to the mean colour (m = 0)
                    masked_images = normalized_image * chunk_masks  # (chunk, 3, H, W): one mask for the three channels, a hidden pixel becomes 0, the dataset-mean colour
                    masked_embeddings = self._embed(masked_images)  # (chunk, D)
                    assert masked_embeddings.shape == (chunk_masks.shape[0], unmasked_embedding.shape[1])
                    chunk_scores = F.cosine_similarity(unmasked_embedding, masked_embeddings, dim=1)  # (chunk,), the unmasked embedding is broadcast over the chunk
                    scores_per_batch.append(chunk_scores)
                scores = torch.cat(scores_per_batch, dim=0)  # (num_masks,)

                # importance (RELAX Eq. 2): per pixel, the mean over the masks of the score times the mask value
                # remove the single channel of the masks
                mask_values = masks.squeeze(1)  # (num_masks, H, W), the singleton channel dimension removed
                scores_per_mask = scores.view(num_masks, 1, 1)  # (num_masks, 1, 1), reshape for the broadcast in the next line
                weighted_scores = scores_per_mask * mask_values  # (num_masks, H, W)
                importance = weighted_scores.sum(dim=0) / num_masks  # (H, W), the masks are collapsed here

                # uncertainty (RELAX Eq. 9): per pixel, the mean over the masks of the squared deviation of the score from the importance, times the mask value
                deviations = scores_per_mask - importance  # (num_masks, H, W)
                squared_deviations = deviations.square()
                weighted_squared_deviations = squared_deviations * mask_values
                uncertainty = weighted_squared_deviations.sum(dim=0) / num_masks  # (H, W), the masks are collapsed here

                # thresholded importance (U-RELAX): the importance only where the uncertainty is below gamma times its median, 0 elsewhere
                uncertainty_threshold = uncertainty_threshold_scale * uncertainty.median()
                is_certain = uncertainty < uncertainty_threshold  # (H, W) booleans
                thresholded_importance = importance * is_certain

                if image_index == self._image_indices[0]:
                    example_masks = masks[:20, 0].cpu()

                maps_per_image[image_index] = {
                    "image": rgb_image.cpu(),
                    "importance": importance.cpu(),
                    "uncertainty": uncertainty.cpu(),
                    "thresholded_importance": thresholded_importance.cpu(),
                }

        self.results = {"image_indices": self._image_indices, "maps": maps_per_image, "example_masks": example_masks}  # the outputs of the method (the settings are in the config.json of the run)

    # One figure per analysed image: the input, then the importance, the uncertainty and the thresholded importance as heat maps (cold = low, warm = high) over the input at half opacity,
    # each with its own colour bar. In the thresholded map only the certain pixels are coloured. Returns {file name: figure}.
    def plot(self):
        figures = {}
        for image_index in self.results["image_indices"]:
            maps = self.results["maps"][image_index]
            image_channels_last = maps["image"].permute(1, 2, 0).numpy()  # (H, W, 3)
            figure = Figure(figsize=(18, 4))
            axes = figure.subplots(1, 4, gridspec_kw={"wspace": 0.35})
            axes[0].imshow(image_channels_last)
            axes[0].set_title(f"image {image_index}")
            for axis, map_name in zip(axes[1:], ("importance", "uncertainty", "thresholded_importance")):
                values = maps[map_name].numpy()
                if map_name == "thresholded_importance":
                    values = np.ma.masked_equal(values, 0.0)  # the uncertain pixels (set to 0) stay transparent
                axis.imshow(image_channels_last)
                shown = axis.imshow(values, cmap="coolwarm", alpha=0.5)
                colorbar_axis = axis.inset_axes([1.03, 0.0, 0.05, 1.0])  # beside the panel, not taken out of it, so all four panels keep the same size
                figure.colorbar(shown, cax=colorbar_axis)
                axis.set_title(map_name.replace("_", " "))
            for axis in axes:
                axis.axis("off")
            figures[f"image_{image_index}"] = figure

        figure = Figure(figsize=(10, 8))
        axes = figure.subplots(4, 5)
        for axis, mask in zip(axes.flat, self.results["example_masks"]):
            axis.imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
        figure.suptitle("20 example RELAX masks (white = visible)")
        figures["debug_masks"] = figure
        return figures
