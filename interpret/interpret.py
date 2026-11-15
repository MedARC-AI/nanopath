# interpret/interpret.py: extracts hooked activations to files, then runs the configured methods on them (see README.md).
import os
import sys
import math
import random
import shutil
import json
import hashlib
import subprocess
import yaml
import numpy as np
import torch
from sklearn.utils import resample
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from model import ViT
from probe import worker_probe_transforms
from dataloader import HED_FROM_RGB, rgb_to_hed
from interpret.data import load_dataset, ActivationDataset

# Headless plotting (figures are only saved to files): Colab exports MPLBACKEND=<notebook backend>, which is not installed in the venv.
os.environ["MPLBACKEND"] = "agg"
from interpret.utils import write_dashboard, tissue_masks
from interpret.methods.effective_rank import EffectiveRank
from interpret.methods.cka import CKA
from interpret.methods.linear_probe import LinearProbe
from interpret.methods.layer_inversion import LayerInversion
from interpret.methods.relax import RELAX
from interpret.methods.tsne import TSNEMaps


# -------------------------------------------------------------------------------------------------
# REPRESENTATION EXTRACTION
# -------------------------------------------------------------------------------------------------

# Token groups every hook must provide; only these are validated and stored, any other group is ignored.
REQUIRED_GROUPS = ("patch", "cls")


# Validates the hook contract in interpret/README.md; add new checks here and document them there.
def assert_hook_interface(storage_dict, layer_names, batch_size, global_state):
    assert len(layer_names) == len(set(layer_names)), "Duplicate layer keys in returned list."

    for layer in layer_names:
        where_layer = f"Layer '{layer}'"
        assert layer in storage_dict, f"{where_layer} missing from hook output."
        for group_name in REQUIRED_GROUPS:
            assert group_name in storage_dict[layer], f"{where_layer} missing required group '{group_name}'."
        group_dict = {group_name: storage_dict[layer][group_name] for group_name in REQUIRED_GROUPS}

        for group_name, tensor in group_dict.items():
            where = f"{where_layer} group '{group_name}'"
            assert tensor.ndim == 3, f"{where} not 3D."
            assert tensor.shape[0] == batch_size, f"{where} batch {tensor.shape[0]} != {batch_size}."
            assert tensor.shape[1] > 0, f"{where} is empty (0 tokens); omit empty groups."
            assert not tensor.requires_grad, f"{where} not detached."
            assert torch.isfinite(tensor).all(), f"{where} contains non-finite values."

        patch = group_dict["patch"]
        cls = group_dict["cls"]
        assert cls.shape[1] == 1, f"{where_layer} 'cls' group has {cls.shape[1]} tokens, expected exactly 1."
        assert math.isqrt(patch.shape[1]) ** 2 == patch.shape[1], f"{where_layer} 'patch' length {patch.shape[1]} not a perfect square."
        assert patch.shape[2] == cls.shape[2], f"{where_layer} hidden dims differ: patch {patch.shape[2]} vs cls {cls.shape[2]}."

        # Token counts must match across all layers and batches (the first one sets them).
        tokens = {group_name: tensor.shape[1] for group_name, tensor in group_dict.items()}
        if "tokens" not in global_state:
            global_state["tokens"] = tokens
        assert tokens == global_state["tokens"], f"{where_layer} token counts {tokens} mismatch the first layer's {global_state['tokens']}."


# Labels derived from the pixels of a batch of tiles (`rgb`: (B, 3, H, W) in [0, 1], before the normalization of the model input): one value per tile and measure. The extraction cuts every measure into quantile bins,
# so each becomes a classification target like the metadata fields.
def derive_labels(rgb):
    # compute the tissue fractions
    tissue = tissue_masks(rgb)  # (B, H, W) - the masks of the batch (the helper asserts that rgb is a batch of tiles in [0, 1])
    num_tissue_pixels = tissue.sum(dim=(1, 2))  # (B,) - sum the 1s of the mask
    num_pixels = tissue.shape[1] * tissue.shape[2]
    tissue_fraction = num_tissue_pixels / num_pixels  # (B,)

    # compute and unpack the stain information
    hed = rgb_to_hed(rgb, HED_FROM_RGB.to(rgb.device))  # (B, H, W, 3): hematoxylin, eosin, DAB
    hematoxylin = hed[:, :, :, 0]  # (B, H, W)
    eosin = hed[:, :, :, 1]  # (B, H, W)

    # compute hematoxylin and eosin target
    # keep each stain information on the tissue pixels only and average it there 
    num_tissue_pixels = num_tissue_pixels.clamp_min(1)  # (B,) avoid division by 0
    hematoxylin_on_tissue = (hematoxylin * tissue)  # (B, H, W) - use mask to ignore hematoxylin information outside of tissue
    hematoxylin_target = hematoxylin_on_tissue.sum(dim=(1, 2)) / num_tissue_pixels  # (B,) - collapse image dimensions into a scalar
    eosin_on_tissue = (eosin * tissue)
    eosin_target = eosin_on_tissue.sum(dim=(1, 2)) / num_tissue_pixels  

    # compute brightness (the mean over the channels of every pixel), once over all pixels and once over the tissue pixels only
    brightness = rgb.mean(dim=1)  # (B, H, W)
    brightness_target = brightness.mean(dim=(1, 2))  # (B,) - collapse image dimensions into a scalar, background included
    brightness_on_tissue = brightness * tissue  # (B, H, W) - use mask to ignore the brightness of the background
    tissue_brightness_target = brightness_on_tissue.sum(dim=(1, 2)) / num_tissue_pixels  # (B,)

    return {"brightness": brightness_target, "tissue_brightness": tissue_brightness_target, "tissue_fraction": tissue_fraction, "hematoxylin": hematoxylin_target, "eosin": eosin_target}


# Samples tiles (stratified, from the generators seeded in main()) and writes the extraction (layout in README.md): the labels (metadata fields, plus labels derived from the pixels during the pass), then one pass of the model over the tiles that collects the activations and the images.
# info.json is written last, so it marks a complete run. `settings` determines the files (hashed in main, saved in info.json).
def extract_activations(model, dataset, transform, settings, extract_cfg, activation_dir):
    # Seeded subsample that keeps the proportions of the dataset's stratification dimension (a metadata field).
    dimension = settings["dataset_config"]["stratification_dimension"]
    values = [metadata[dimension] for _, metadata in dataset.records]
    sampled_indices = resample(list(range(len(dataset))), n_samples=settings["num_samples"], replace=False, stratify=values)  # draws from numpy's generator, seeded in main()
    subset = torch.utils.data.Subset(dataset, sampled_indices)
    # "fork" as in probe.py: Python 3.14 defaults to "forkserver", which pickles the dataset and the collate lambda to every worker.
    loader = torch.utils.data.DataLoader(
        subset, batch_size=extract_cfg["batch_size"], shuffle=False, num_workers=extract_cfg["num_workers"],
        multiprocessing_context="fork", collate_fn=lambda batch: tuple(zip(*batch)),
    )
    num_samples = len(sampled_indices)

    (activation_dir / "activations").mkdir(parents=True)
    (activation_dir / "precomputed").mkdir()

    # LABELS: every metadata field as integers (its sorted unique values become 0, 1, 2, ...) plus the names to map them back. Row i is the i-th sampled tile, like in every other file.
    sampled_metadata = [dataset.records[sampled_index][1] for sampled_index in sampled_indices]
    labels = {}  # e.g. {"medical_center": tensor([0, 1, 1, 0, ...])} - each tensor holds one integer per datapoint (int representation of metadata strings)
    label_names = {}  # e.g. {"medical_center": ["center_a", "center_b"]} - this maps the integers back to human readable strings
    for field in sampled_metadata[0]:
        label_names[field] = sorted({metadata[field] for metadata in sampled_metadata})
        value_to_integer = {value: integer for integer, value in enumerate(label_names[field])}
        labels[field] = torch.tensor([value_to_integer[metadata[field]] for metadata in sampled_metadata])

    storage_dict = {}
    layer_names = model.register_interpretability_hooks(storage_dict)
    # Keep every layer_stride-th layer, counting from the last one (so the last is always kept).
    kept_layer_names = []
    for steps_from_last, layer in enumerate(reversed(layer_names)):
        if steps_from_last % settings["layer_stride"] == 0:
            kept_layer_names.append(layer)
    kept_layer_names.reverse()  # back to forward-pass order

    interface_state = {}  # remembers the token counts of the first layer of the first batch, which all later ones must match

    # EXTRACTION PASS: the tensors of every kept layer and the images are allocated at full size in RAM after the first batch (no double peak from concatenating), each batch fills in its rows, and everything is saved at the end.
    # The patch mean is computed here from the float32 tokens, so the methods never average tokens.
    tensors = {}
    derived_labels = []  # per batch, the labels derived from its pixels
    start = 0  # the first row of the current batch
    with torch.inference_mode():
        for batch_images, _ in tqdm(loader, desc="Extracting"):
            batch_size = len(batch_images)
            transformed = [transform(image) for image in batch_images]  # per image: (rgb, normalized)
            rgb_images = torch.stack([rgb for rgb, _ in transformed])  # (batch, 3, H, W) in [0, 1]
            processed_images = torch.stack([normalized for _, normalized in transformed])  # (batch, 3, H, W) the model input
            derived_labels.append(derive_labels(rgb_images))
            storage_dict.clear()  # the hooks hold this dict, so clear it in place
            model(processed_images)

            assert_hook_interface(storage_dict, layer_names, batch_size, interface_state)  # before the stride

            rows = slice(start, start + batch_size)
            # store images: the decoder target of the layer inversion is the tile as the model saw it (resized to 224 px) in [0, 1], not the normalized model input. The normalization is an invertible per-channel shift and scale
            # that a decoder's final linear head can absorb exactly, so leaving it out costs the decoder nothing and keeps the target readable (collage, MSE in intensity units).
            if start == 0:  # allocate the image tensor with the first batch (we only know dimensions at runtime)
                num_channels = rgb_images.shape[1]
                height = rgb_images.shape[2]
                width = rgb_images.shape[3]
                images = torch.empty(num_samples, num_channels, height, width)
            images[rows] = rgb_images
            
            for layer in kept_layer_names:
                patch = storage_dict[layer]["patch"]  # (batch, seq_len, channel_dim)
                if layer not in tensors:  # initialize tensors for layer if not yet initialized
                    seq_len = patch.shape[1]
                    channel_dim = patch.shape[2]
                    tensors[layer] = {
                        "patch_tokens": torch.empty(num_samples, seq_len, channel_dim),
                        "cls": torch.empty(num_samples, channel_dim),
                        "patch_mean": torch.empty(num_samples, channel_dim),
                    }
                # store patch data
                tensors[layer]["patch_tokens"][rows] = patch
                tensors[layer]["patch_mean"][rows] = patch.mean(dim=1)  # mean over the patch tokens
                # store CLS data
                tensors[layer]["cls"][rows] = storage_dict[layer]["cls"][:, 0, :]  # the CLS group is (batch, 1, C), indexing out position 0 gives plain C-vectors
            start += batch_size
    assert start == num_samples

    # DERIVED LABELS: every measure from the pixels (brightness, stain, tissue) is cut into 4 quantile bins by rank (0 = lowest 25%) and saved as classes, like the metadata labels
    for measure_name in derived_labels[0]:  # iterate over the different derived label types
        # collect all labels of the type selected by the loop iteration 
        values = torch.cat([derived[measure_name] for derived in derived_labels]).cpu()  # (N,) the tiles are on the GPU (the transform's last step), so the derived values are too - so we need to send them back to CPU
        quartile_edges = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75]))  # (3,) the values that split the tiles into four equally big groups
        labels[measure_name] = torch.bucketize(values, quartile_edges)  # (N,) 0 = lowest quarter, 3 = highest quarter
        label_names[measure_name] = ["0-25%", "25-50%", "50-75%", "75-100%"]  # human readable version of the ints in labels
    
    torch.save(labels, activation_dir / "labels.pt")
    (activation_dir / "label_names.json").write_text(json.dumps(label_names))

    for layer, layer_tensors in tensors.items():
        torch.save(layer_tensors["patch_tokens"], activation_dir / "activations" / f"{layer}_patch_tokens.pt")
        torch.save(layer_tensors["cls"], activation_dir / "activations" / f"{layer}_cls.pt")
        torch.save(layer_tensors["patch_mean"], activation_dir / "precomputed" / f"{layer}_patch_mean.pt")
    torch.save(images, activation_dir / "images.pt")

    extraction_info = {
        **settings,
        # Exact code version; -dirty means uncommitted changes.
        "git_commit": subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent, text=True).strip(),
        "kept_layers": kept_layer_names,
        "sampled_dataset_indices": sampled_indices,
    }
    (activation_dir / "info.json").write_text(json.dumps(extraction_info))


# -------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------

# probe.py's eval transform plus the normalization it applies at its model-call sites; duplicated here because probe.py is locked, so keep the two in sync.
# Returns both the image after the eval transform (rgb, in [0, 1], for the labels derived from the pixels) and the normalized one (the model input).
def evaluation_preprocess(train_cfg):
    transform, _ = worker_probe_transforms(train_cfg)
    mean = torch.tensor(train_cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(train_cfg["data"]["std"]).view(3, 1, 1)
    return transforms.Compose([transform, transforms.Lambda(lambda tile: (tile, (tile - mean) / std))])


# Loads a Nanopath checkpoint and returns the model (eval mode, on the CPU), the training config its preprocessing comes
# from, and its training step (part of the cache hash). To interpret another model, swap this function.
def load_model(checkpoint_path, model_weights):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ViT(variant=checkpoint["config"]["model"]["type"])
    model.load_state_dict(checkpoint[model_weights], strict=True)
    return model.eval(), checkpoint["config"], checkpoint["step"]


# seeds every random number generator and switches off nondeterministic GPU behaviour, so extractions and results are reproducible (same as the probe
# job in probe.py, plus numpy, cuDNN and TF32). bitwise equality still needs the same GPU, driver and library versions.
def set_seeds(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for deterministic cuBLAS matmuls; must be set before the first CUDA operation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # raises an error if an operation has no deterministic implementation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark mode picks kernels by timing, which can differ between runs
    torch.backends.cuda.matmul.allow_tf32 = False  # full float32 matmuls, no TF32 rounding
    torch.backends.cudnn.allow_tf32 = False


def run_pipeline(checkpoint_path, results_dir, activation_cache_dir, cfg, icfg, device, dataset, transform, extract_cfg):
    model, train_cfg, checkpoint_step = load_model(checkpoint_path, icfg["model_weights"])
    model = model.to(device)
    if not hasattr(model, "register_interpretability_hooks"):
        print(
            "\n============================================================\n"
            "INTERPRETABILITY HOOK NOT IMPLEMENTED\n"
            "Implement register_interpretability_hooks(self, storage_dict) for this model.\n"
            "See the hook contract in interpret/README.md.\n"
            "============================================================",
            file=sys.stderr,
            flush=True,
        )
        raise AttributeError("model is missing register_interpretability_hooks")
    print(f"Loaded checkpoint from {checkpoint_path}")

    dataset_cfg = icfg["datasets"][icfg["dataset"]]
    # Extractions are cached in activation_cache_dir/data/<hash of settings> (layout in README.md; results go to results_dir); if a run died, info.json is
    # missing and loading crashes below: delete that directory and rerun.
    settings = {
        "checkpoint_path": checkpoint_path, "checkpoint_step": checkpoint_step, "model_weights": icfg["model_weights"],
        "dataset": icfg["dataset"], "dataset_root": str(dataset.root), "dataset_config": dataset_cfg,
        "seed": icfg["seed"], "num_samples": extract_cfg["num_samples"], "layer_stride": extract_cfg["layer_stride"],
    }
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    data_dir = activation_cache_dir / "data" / settings_hash

    # ---- EXTRACTION ----
    if data_dir.exists():
        print(f"Reusing extraction in {data_dir}")
    else:
        print(f"Extracting features into {data_dir}...")
        extract_activations(model, dataset, transform, settings, extract_cfg, data_dir)
        (data_dir / "config.json").write_text(json.dumps(cfg, indent=2))  # the full config that produced these files (a reused extraction keeps its own)

    kept_layers = json.loads((data_dir / "info.json").read_text())["kept_layers"]  # in forward-pass order
    (results_dir / "config.json").write_text(json.dumps(cfg, indent=2))  # the full config of this run

    # ---- ANALYSIS ----
    results = {}
    for method_name in icfg["methods"]:
        print(f"Running {method_name}...")
        # every method builds its own fresh activation datasets (one per kept layer) with exactly the blocks it needs: all data starts out on the CPU and moves to the device only when the method asks for it,
        # so methods never depend on each other. 
        if method_name == "effective_rank":
            activation_datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = EffectiveRank(activation_datasets)
        elif method_name == "cka":
            activation_datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = CKA(activation_datasets, icfg[method_name])
        elif method_name == "linear_probe":
            activation_datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = LinearProbe(activation_datasets, device, icfg["seed"], icfg[method_name])
        elif method_name == "layer_inversion":
            activation_datasets = {layer: ActivationDataset(data_dir, layer, device, patch_tokens=True, image=True) for layer in kept_layers}  # the decoders read the patch tokens and the target images
            method = LayerInversion(activation_datasets, device, icfg["seed"], icfg[method_name])
        elif method_name == "relax":
            method = RELAX(dataset, transform, model, device, icfg[method_name])  # RELAX runs the model itself on masked copies of the images, so it gets the source dataset and the extraction's transform instead of the extractions
        elif method_name == "tsne":
            activation_datasets = {layer: ActivationDataset(data_dir, layer, device) for layer in kept_layers}
            method = TSNEMaps(activation_datasets, icfg[method_name])
        else:
            raise KeyError(f"Unknown method '{method_name}'; add it to the method dispatch in main() of interpret/interpret.py")
        method.run()
        # THE RESULTS CONTRACT: method.results is the one container of the method's outputs, the final numbers, curves, images and maps it produces, i.e. all the data its plots show
        # (plain Python values, dicts, lists and CPU tensors, so that pixel data is kept losslessly). It does not hold the intermediates that are only needed to compute the outputs
        # (e.g. CKA's similarity tables, RELAX's masks, trained weights). plot() takes its data from it alone. Every method must follow this.
        results[method_name] = method.results

        # Each method saves the plots it defines into its own subfolder.
        plot_dir = results_dir / method_name
        plot_dir.mkdir(exist_ok=True)
        for plot_name, figure in method.plot().items():
            figure.savefig(plot_dir / f"{plot_name}.png", bbox_inches="tight")

    # the results of all methods, saved together: the complete paper trail of the run (read it back with torch.load(path, weights_only=False), since the results also hold numpy floats)
    torch.save(results, results_dir / "results.pt")
    # remove activations as otherwise we risk piling up GBs of activation data on cluster in case people run interpretability with many different checkpoints
    shutil.rmtree(data_dir)


# Everything computes on ONE device (the GPU). main() picks it, moves the model and gives it to the activation datasets, which hold their tensors on it,
# so the methods never move data between devices.
def main():

    # ---- SETUP ----
    device = "cuda"
    # Usage: python -m interpret.interpret CONFIG.yaml checkpoint_path=... activation_cache_dir=... results_dir=... [baseline_checkpoint_path=...]
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    icfg = cfg["interpret"]
    for arg in sys.argv[2:]:
        key, value = arg.split("=", 1)
        icfg[key] = value
    results_dir = Path(icfg["results_dir"])
    activation_cache_dir = Path(icfg["activation_cache_dir"])
    if results_dir.exists():  # if output files exits at this path wipe them rather then writing new files next to them
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    # SEEDING POLICY: the one seed of the config is set here, once, for every random number generator (python, numpy, torch on the CPU and the GPU), and nothing else in the rig seeds.
    # The extraction sampler, the methods and the libraries they call (sklearn draws from numpy's) all draw from these generators, so a run is reproduced by its config.json (the same config, so the same order of methods).
    # only exception: the fold splitter of Probe (probe_base.py) gets the seed as random_state, so that all variants of a method share their folds (the decoders of the layer inversion are compared
    # on the same splits, and its collage shows the same tiles in every column).
    set_seeds(icfg["seed"])  # before the first CUDA operation
    # Load model config for eval transform from primary checkpoint
    _, train_cfg, _ = load_model(icfg["checkpoint_path"], icfg["model_weights"])
    # The final Lambda puts each image on `device`, so the extractor never moves anything itself.
    transform = transforms.Compose([evaluation_preprocess(train_cfg), transforms.Lambda(lambda images: [tile.to(device) for tile in images])])

    dataset = load_dataset(icfg["dataset"], cfg["probe"]["dataset_roots"])
    extract_cfg = icfg["extract"]

    baseline_path = icfg.get("baseline_checkpoint_path")
    if baseline_path:
        baseline_results_dir = results_dir / "baseline"
        trained_results_dir = results_dir / "trained"
        baseline_results_dir.mkdir(parents=True, exist_ok=True)
        trained_results_dir.mkdir(parents=True, exist_ok=True)

        print("=== running baseline analysis ===")
        run_pipeline(baseline_path, baseline_results_dir, activation_cache_dir, cfg, icfg, device, dataset, transform, extract_cfg)

        print("=== running trained model analysis ===")
        run_pipeline(icfg["checkpoint_path"], trained_results_dir, activation_cache_dir, cfg, icfg, device, dataset, transform, extract_cfg)

        write_dashboard(results_dir, cfg, baseline_dir=baseline_results_dir, trained_dir=trained_results_dir)
    else:
        run_pipeline(icfg["checkpoint_path"], results_dir, activation_cache_dir, cfg, icfg, device, dataset, transform, extract_cfg)
        write_dashboard(results_dir, cfg)

    print(f"dashboard saved to: {results_dir / 'dashboard.html'}")
