# interpret/methods/tsne.py: t-SNE maps of every layer: the CLS and the patch-mean vectors of a fixed set of tiles embedded in 2-D, one figure per label (the question: README.md).
# Uses sklearn's t-SNE (van der Maaten & Hinton 2008) on the raw embeddings, one run per vector type and layer; the labels only colour the maps.

import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from tqdm import tqdm

from interpret.utils import strided_indices


class TSNEMaps:
    def __init__(self, activation_datasets, method_cfg):
        self._activation_datasets = activation_datasets  # {layer name: ActivationDataset}, see data.py
        self._cfg = method_cfg
        first_dataset = list(activation_datasets.values())[0]  # the activation datasets of all layers advertise the same examples, vector types and labels
        self._vector_types = first_dataset.vector_types  # {vector type: description}: CLS and the patch mean, mapped separately

        # the tiles to map: num_tiles rows of the extraction, evenly spaced (a fixed stride, no randomness), so that every run and every checkpoint maps the same tiles
        num_tiles = method_cfg["num_tiles"]
        num_examples = len(first_dataset)
        assert num_tiles <= num_examples, f"tsne.num_tiles is {num_tiles}, but the extraction has only {num_examples} tiles"
        self._tile_indices = strided_indices(num_examples, num_tiles)

        # the labels of the selected tiles (the same in every layer), which the plots colour the maps by
        all_labels = first_dataset.get_block("labels", to_device=False)  # {field: (N,) integers}
        self._labels = {field: values[self._tile_indices] for field, values in all_labels.items()}  # {field: (num_tiles,) integers}
        self._label_names = first_dataset.label_names  # {field: class names}

        self.results = None

    # One t-SNE map per vector type and layer, of the selected tiles. The labels of those tiles go into the results too, since the plots colour the maps by them.
    def run(self):
        maps = {}  # {vector type: {layer name: (num_tiles, 2) CPU tensor of 2-D coordinates}}
        for vector_type in self._vector_types:  # iterate over vector types
            maps[vector_type] = {}
            for layer_name, dataset in tqdm(self._activation_datasets.items(), desc=f"t-SNE ({vector_type})"):  # iterate over layers, first to last
                all_features = dataset.get_block("activations", vector_type, to_device=False)  # (N, C), the whole layer on the CPU, where sklearn runs
                features = all_features[self._tile_indices, :]  # (num_tiles, C), the raw embeddings: t-SNE's per-point widths adapt to their scale

                # the 2-D map: sklearn's defaults (2 components, a PCA start layout, Barnes-Hut) with the perplexity of the config; its random draws come from numpy's generator, seeded in main()
                tsne = TSNE(perplexity=self._cfg["perplexity"])
                coordinates = tsne.fit_transform(features.numpy())  # (num_tiles, 2) numpy array
                maps[vector_type][layer_name] = torch.from_numpy(coordinates)

        self.results = {"tile_indices": self._tile_indices, "maps": maps, "labels": self._labels, "label_names": self._label_names}  # the outputs of the method (the settings are in the config.json of the run)

    # One figure per label: a row per vector type, a column per layer, every 2-D map coloured by that label of its tiles (one colour per class, a legend when there are
    # at most 10 classes). Returns {file name: figure}.
    def plot(self):
        figures = {}
        maps = self.results["maps"]
        vector_types = list(maps)
        layer_names = list(maps[vector_types[0]])
        for field, values in self.results["labels"].items():
            class_names = self.results["label_names"][field]
            num_classes = len(class_names)
            figure = Figure(figsize=(4 * len(layer_names), 4 * len(vector_types)))
            axes = figure.subplots(len(vector_types), len(layer_names), squeeze=False)
            for row_index, vector_type in enumerate(vector_types):
                for column_index, layer_name in enumerate(layer_names):
                    coordinates = maps[vector_type][layer_name]
                    axis = axes[row_index][column_index]
                    shown = axis.scatter(coordinates[:, 0].numpy(), coordinates[:, 1].numpy(), c=values.numpy(), cmap="viridis", vmin=0, vmax=num_classes - 1, s=2, linewidths=0)
                    axis.set_title(f"{vector_type}, {layer_name}", fontsize=9)
                    axis.set_xticks([])
                    axis.set_yticks([])
            if num_classes <= 10:  # more classes (the slides) cannot be told apart by colour, so they get no legend
                handles, _ = shown.legend_elements(num=None)  # one marker per class present, in increasing order of the class integers
                present_classes = values.unique().tolist()  # sorted, like the handles
                legend_labels = []
                for class_index in present_classes:
                    legend_labels.append(class_names[class_index])
                figure.legend(handles, legend_labels, loc="upper right", fontsize=8)
            figure.suptitle(f"t-SNE maps coloured by {field} (rows: vector types, columns: layers)")
            figures[f"{field}_tsne"] = figure
        return figures
