# interpret/data.py: the data of the interpretability rig, in two sections.
# SOURCE DATASETS: one class per analysis dataset, exposing `records` (image, metadata) for the sampler of the extraction.
# ACTIVATION DATASETS: ActivationDataset serves one layer of an extraction (whole small blocks with get_block, examples with __getitem__) and DeviceLoader batches its examples and moves every batch to the device.

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image


# -------------------------------------------------------------------------------------------------
# SOURCE DATASETS
# -------------------------------------------------------------------------------------------------

# PathoROB Camelyon tiles, loaded as in probe.py's PathoROB probe. Reimplemented because that code is inline in a larger
# function and returns only the transformed image; we also expose the metadata: slide_id, biological_class, medical_center.
class PathoRobCamelyonDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        table = pa.concat_tables([pq.read_table(f) for f in sorted((self.root / "camelyon").glob("data/*.parquet"))])
        images = [r["bytes"] for r in table.column("image").to_pylist()]
        metadata = table.select(["slide_id", "biological_class", "medical_center"]).to_pylist()
        self.records = list(zip(images, metadata))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_bytes, metadata = self.records[index]
        return Image.open(io.BytesIO(image_bytes)).convert("RGB"), metadata


# Builds the dataset named by `identifier`; `dataset_roots` is probe.dataset_roots from the config (name -> folder).
def load_dataset(identifier, dataset_roots):
    if identifier == "pathorob_camelyon":
        return PathoRobCamelyonDataset(Path(dataset_roots["pathorob"]))
    raise KeyError(f"Unknown dataset '{identifier}'; add it to load_dataset in interpret/data.py")


# -------------------------------------------------------------------------------------------------
# ACTIVATION DATASETS
# -------------------------------------------------------------------------------------------------

# One layer of an extraction (layout in README.md): row i of every file is the i-th sampled tile. The constructor sends nothing to `device`: the tensors are CPU tensors in memory
# (the big files `patch_tokens` and `image` are memory-mapped and only opened when enabled), all of them are private, and the two ways out, get_block (a whole small block) and DeviceLoader (batches, below), are the only
# places that move data to the device; __getitem__ returns one example as CPU tensors. A part that is not enabled is an empty tensor, so items always have the same keys.
# Only two facts are public: label_names and vector_types.
class ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, layer_name, device, patch_tokens=False, image=False):
        data_dir = Path(data_dir)
        self._device = device
        # load data that the dataset spans
        # the activations of this layer for every example: the vectors, (N, C) each, are in memory, the big patch tokens, (N, 256, C), are memory-mapped and None unless enabled
        self._activations = {
            "cls": torch.load(data_dir / "activations" / f"{layer_name}_cls.pt"),
            "patch_tokens": torch.load(data_dir / "activations" / f"{layer_name}_patch_tokens.pt", mmap=True) if patch_tokens else None,
            "patch_mean": torch.load(data_dir / "precomputed" / f"{layer_name}_patch_mean.pt"),
        }
        self._images = torch.load(data_dir / "images.pt", mmap=True) if image else None  # (N, 3, H, W)
        self._labels = torch.load(data_dir / "labels.pt")  # e.g. {"medical_center": tensor([0, 1, 1, 0, ...])} - each tensor holds one integer per datapoint (int representation of metadata strings)
        # the two public facts
        self.label_names = json.loads((data_dir / "label_names.json").read_text())  # e.g. {"medical_center": ["center_a", "center_b"]} - this maps the integers back to human readable strings
        self.vector_types = {"cls": "CLS token", "patch_mean": "mean over the patch tokens"}  # the vectors every item holds, with a description for titles

    def __len__(self):
        return len(self._activations["cls"])

    # a whole small block of this layer on the device: `field` names a dict of this class ("activations" or "labels") and `key` picks the block in it, e.g. get_block("activations", "cls") or get_block("labels", "medical_center");
    # without a key the whole dict is returned, which only exists for the labels (get_block("labels", to_device=False), a dict has no device move).
    # The big blocks (`patch_tokens`, `images`) are deliberately not available here: they are memory-mapped so that they are never loaded whole, and are only read in batches through DeviceLoader.
    # to_device=False skips the device move, for the few callers that work on the host (sklearn and printouts).
    def get_block(self, field, key=None, to_device=True):
        assert field != "images" and key != "patch_tokens" and (field, key) != ("activations", None), f"'{field}' / '{key}' includes a big block: it is memory-mapped so that it is never loaded whole, so it can only be read in batches through DeviceLoader"
        block = getattr(self, f"_{field}")
        if key is not None:
            block = block[key]
        return block.to(self._device) if to_device else block

    # one example as CPU tensors (this is what DataLoader and Subset fetch); the device move happens per batch in DeviceLoader, not per example here
    def __getitem__(self, i):
        not_loaded = torch.empty(0)
        image = self._images[i] if self._images is not None else not_loaded
        patch_tokens = self._activations["patch_tokens"][i] if self._activations["patch_tokens"] is not None else not_loaded
        return {
            "image": image,
            "cls": self._activations["cls"][i],
            "patch_tokens": patch_tokens,
            "patch_mean": self._activations["patch_mean"][i],
            "labels": {field: values[i] for field, values in self._labels.items()},
        }


# A DataLoader over the examples of a dataset at `indices` that moves every batch it yields to the device. This is the one place that moves batches: workers cannot use the GPU, so they stack the examples on the CPU
# and each batch is moved here, in the main process, with one copy per tensor (instead of per example, which is slow).
class DeviceLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, device, indices, batch_size, shuffle=False, num_workers=0):
        self._device = device
        # a Subset is a lazy view of the examples: it only holds the dataset and the indices (no data is copied or loaded), and reads an example from the dataset when it is fetched
        # "fork" as in the extraction: Python 3.14 defaults to "forkserver", which pickles the dataset to every worker; persistent workers survive between epochs
        super().__init__(  
            torch.utils.data.Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, persistent_workers=num_workers > 0, multiprocessing_context="fork" if num_workers > 0 else None,
        )

    def __iter__(self):
        # STEP 1 (in the base DataLoader, in its workers): 
        # the workers call __getitem__ for the examples of a batch and collate them, 
        # so every pass of this loop receives one collated CPU batch (the workers keep preparing the next batches in the background)
        for cpu_batch in super().__iter__():
            # STEP 2 (here, in the main process): 
            # send the collated CPU batch to the device, which the workers cannot do. The batch is a nested dict 
            # (labels is a dict inside it), and tree_map applies the function to every tensor at any depth, so one call moves all of it
            device_batch = torch.utils._pytree.tree_map(lambda tensor: tensor.to(self._device), cpu_batch)
            yield device_batch
