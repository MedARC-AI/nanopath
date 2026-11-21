# interpret/methods/effective_rank.py: per-layer effective rank of the centered mean-over-patch-tokens and CLS vectors.

from matplotlib.figure import Figure
import torch
from tqdm import tqdm


class EffectiveRank:
    def __init__(self, activation_datasets):
        self._activation_datasets = activation_datasets  # {layer name: ActivationDataset}, see data.py
        first_dataset = list(activation_datasets.values())[0]
        self.vector_types = first_dataset.vector_types  # {vector type: description}, advertised by the activation datasets (the same in every layer)
        self.results = None  # {vector type: {layer: effective rank}}, set by run()

    def run(self):
        # One effective rank value per (vector type, layer): results[vector type][layer]
        results = {}
        for vector_type in self.vector_types:  # iterate over vector types
            results[vector_type] = {}

            for layer_name, dataset in tqdm(self._activation_datasets.items(), desc=f"Effective rank ({vector_type})"):  # iterate over layers, first to last
                features = dataset.get_block("activations", vector_type)  # (n, C), one row per example (the whole layer, on the device)

                # center the embedding cloud (PCA-style) so the singular values describe the variation within it, not its offset from the origin
                features = features - features.mean(dim=0, keepdim=True)
                singular_values = torch.linalg.svdvals(features)
                # normalize to a probability distribution (the sum of the non-negative singular values is their L1 norm)
                probabilities = singular_values / singular_values.sum()
                entropy = torch.special.entr(probabilities).sum()
                results[vector_type][layer_name] = torch.exp(entropy).item()

        self.results = results

    # One line plot per representation: effective rank per layer. self.results is in forward-pass order, so the x axis
    # runs from the first to the last layer. Returns {file name: figure}; main() saves them.
    def plot(self):
        figures = {}
        for vector_type, title in self.vector_types.items():
            layer_names = list(self.results[vector_type])
            ranks = []
            for layer_name in layer_names:
                ranks.append(self.results[vector_type][layer_name])
            figure = Figure()
            axis = figure.subplots()
            axis.plot(layer_names, ranks, marker="o")
            axis.set_xlabel("layer (increasing depth)")
            axis.set_ylabel("effective rank")
            axis.set_title(f"Effective rank of the {title} per layer")
            figures[f"{vector_type}_effective_rank"] = figure
        return figures
