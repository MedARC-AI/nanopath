# interpret/methods/probe_base.py: the skeleton shared by the methods that train small models on the activations of every layer (LinearProbe, LayerInversion).
# It builds folds that never split a group, trains a grid of models together on the same batches, records the loss curves and averages the metrics over the folds.
# A subclass sets NAME, METRICS (the first one selects the best grid point), BEST (max or min) and VARIANTS_KEY (the key of its variants in the results), implements the hooks listed below (the wrappers after them assert their inputs and outputs), and brings its own briefing and plots.
# Inlining the hooks of LinearProbe into this class gives the code the linear probe had before it was split.

import inspect

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from interpret.data import DeviceLoader


class Probe:
    def __init__(self, activation_datasets, device, seed, method_cfg, variants):
        self._activation_datasets = activation_datasets  # {layer name: ActivationDataset}, see data.py
        self.device = device  # where the models live (the activation datasets already hand out tensors on it)
        self.seed = seed  # only for the fold splitter, the one exception to the seeding policy (main() in interpret.py)
        self._cfg = method_cfg
        self.variants = variants  # what the method varies: the targets of the linear probe, the decoders of the layer inversion. Every variant has its own folds, layer loop and results
        self.results = {}  # the outputs of the method (the contract is in main() of interpret.py); exists from the start, so that hooks can write into it during the run

        self.layer_names = list(activation_datasets)

        # the labels (integers) and their names are the same in the dataset of every layer, so the first one gives them all
        first_dataset = list(activation_datasets.values())[0]
        self.labels = first_dataset.get_block("labels", to_device=False)  # all labels, they stay on the CPU: the folds (sklearn) and the briefing work on the host
        self.label_names = first_dataset.label_names
        group_ids = self.labels[self._cfg["leakage_group"]].numpy()

        # pre-compute folds (once per variant)
        # DATA LEAKAGE PREVENTION
        # CLASS BALANCING
        self.folds = {}
        for variant in variants:
            splitter = StratifiedGroupKFold(n_splits=self._cfg["folds"], shuffle=True, random_state=seed)  # the one exception to the seeding policy (main() in interpret.py): all variants share their folds
            self.folds[variant] = []
            for train_indices, val_indices in splitter.split(list(range(len(first_dataset))), self._call_stratify_labels(variant).numpy(), groups=group_ids):
                self.folds[variant].append((train_indices.tolist(), val_indices.tolist()))

    # -------------------------------------------------------------------------------------------------
    # HOOK DECLARATIONS
    # -------------------------------------------------------------------------------------------------

    # THE HOOKS: the interface every subclass implements with exactly these arguments (a missing hook raises here).
    # `variant` is one of self.variants, `fold_statistics` is what fold_stats() returned for the training fold, `evaluation_state` is what start_evaluation_collection() returned.
    # The generic methods call the hooks only through the wrappers of the next section, which assert the inputs and outputs of each hook.

    # The labels the folds are stratified on. 
    def stratify_labels(self, variant):
        raise NotImplementedError

    # A dict of statistics of the training fold only (prepare_batch() standardizes every batch of both subsets with them): the keys "mean" and "std" (tensors of the same shape) and "class_weights" (the weight of every class in the loss, None if the method has no classes), and whatever else the method needs.
    def fold_stats(self, dataset, variant, train_indices):
        raise NotImplementedError

    # The module to train. 
    def make_model(self, variant, fold_statistics):
        raise NotImplementedError

    # One batch of an ActivationDataset, with the inputs standardized with the statistics of the training fold.
    def prepare_batch(self, batch, variant, fold_statistics):
        raise NotImplementedError

    # The loss of one batch (a scalar), train=False is the validation loss.
    def loss(self, outputs, targets, fold_statistics, train):
        raise NotImplementedError

    # The state that collects the extra data the final evaluation needs (the base itself tracks only the train and validation loss curves). It is made at the start of every validation pass.
    def start_evaluation_collection(self, num_models, fold_statistics):
        raise NotImplementedError

    # Called for every validation batch with the outputs of every model (a list with one tensor per model); adds what the final evaluation needs to the evaluation state.
    # `variant`, `layer_name`, `fold` and `setting_names` say what is being trained, `last_epoch` is True in the pass of the last epoch: what is only needed once (e.g. plot data) can be written into self.results then.
    def collect_evaluation_batch(self, evaluation_state, outputs_per_model, targets, variant, layer_name, fold, setting_names, last_epoch):
        raise NotImplementedError

    # The metrics of every model from the evaluation state of the last epoch.
    def final_evaluation(self, evaluation_state, variant, layer_name, fold, setting_names, fold_statistics):
        raise NotImplementedError

    # -------------------------------------------------------------------------------------------------
    # HOOK INTERFACE ASSERTIONS 
    # -------------------------------------------------------------------------------------------------
    
    # the generic methods call the hooks only through these wrappers: assert the inputs, call the hook, assert the outputs, return.
    # ASSERTED:
    #  - the number and the names of the hook arguments: every hook has exactly the arguments of its declaration above, in the same order (an argument a hook does not use keeps its declared name)
    #  - the type of the arguments and of the returned objects (dict, list, bool, int, tensor, module, float)
    #  - the keys of a returned dict (fold_stats: mean, std and class_weights; final_evaluation: one entry per setting name, each with exactly the keys of METRICS)
    # NOT ASSERTED:
    #  - values (ranges, finiteness, positivity, label values, whether the variant, layer and fold are known ones)
    #  - the structure of tensors (shapes, dtype, device, differentiability)
    #  - the contents of a dict beyond the keys above (the subclass's own keys, e.g. class_weights) and of the evaluation state

    # checked when a subclass is defined, so that a wrong hook fails before any data is read
    def __init_subclass__(cls):
        for hook_name in ("stratify_labels", "fold_stats", "make_model", "prepare_batch", "loss", "start_evaluation_collection", "collect_evaluation_batch", "final_evaluation"):
            declared_arguments = list(inspect.signature(getattr(Probe, hook_name)).parameters)
            implemented_arguments = list(inspect.signature(getattr(cls, hook_name)).parameters)
            assert implemented_arguments == declared_arguments, f"{cls.__name__}.{hook_name} takes {implemented_arguments}, the declaration in Probe takes {declared_arguments}"

    def _call_stratify_labels(self, variant):
        labels = self.stratify_labels(variant)
        assert isinstance(labels, torch.Tensor), f"stratify_labels must return a tensor, got {type(labels)}"
        return labels

    def _call_fold_stats(self, dataset, variant, train_indices):
        assert isinstance(train_indices, list), f"train_indices must be a list, got {type(train_indices)}"
        fold_statistics = self.fold_stats(dataset, variant, train_indices)
        assert isinstance(fold_statistics, dict), f"fold_stats must return a dict, got {type(fold_statistics)}"
        assert {"mean", "std", "class_weights"} <= fold_statistics.keys(), f"fold_stats must return the keys mean, std and class_weights, got {list(fold_statistics)}"
        return fold_statistics

    def _call_make_model(self, variant, fold_statistics):
        assert isinstance(fold_statistics, dict), f"fold_statistics must be a dict, got {type(fold_statistics)}"
        model = self.make_model(variant, fold_statistics)
        assert isinstance(model, torch.nn.Module), f"make_model must return a torch.nn.Module, got {type(model)}"
        return model

    def _call_prepare_batch(self, batch, variant, fold_statistics):
        assert isinstance(batch, dict), f"batch must be a dict, got {type(batch)}"
        assert isinstance(fold_statistics, dict), f"fold_statistics must be a dict, got {type(fold_statistics)}"
        inputs, targets = self.prepare_batch(batch, variant, fold_statistics)
        assert isinstance(inputs, torch.Tensor) and isinstance(targets, torch.Tensor), f"prepare_batch must return two tensors, got {type(inputs)} and {type(targets)}"
        return inputs, targets

    def _call_loss(self, outputs, targets, fold_statistics, train):
        assert isinstance(train, bool), f"train must be a bool, got {type(train)}"
        loss = self.loss(outputs, targets, fold_statistics, train)
        assert isinstance(loss, torch.Tensor), f"loss must return a tensor, got {type(loss)}"
        return loss

    def _call_start_evaluation_collection(self, num_models, fold_statistics):
        assert isinstance(num_models, int), f"num_models must be an int, got {type(num_models)}"
        assert isinstance(fold_statistics, dict), f"fold_statistics must be a dict, got {type(fold_statistics)}"
        return self.start_evaluation_collection(num_models, fold_statistics)  # the state is opaque to the base: nothing to assert about it

    def _call_collect_evaluation_batch(self, evaluation_state, outputs_per_model, targets, variant, layer_name, fold, setting_names, last_epoch):
        assert isinstance(outputs_per_model, list), f"outputs_per_model must be a list with one tensor per model, got {type(outputs_per_model)}"
        assert isinstance(last_epoch, bool), f"last_epoch must be a bool, got {type(last_epoch)}"
        returned = self.collect_evaluation_batch(evaluation_state, outputs_per_model, targets, variant, layer_name, fold, setting_names, last_epoch)
        assert returned is None, f"collect_evaluation_batch must change the state in place and return nothing, got {type(returned)}"

    def _call_final_evaluation(self, evaluation_state, variant, layer_name, fold, setting_names, fold_statistics):
        assert isinstance(setting_names, list), f"setting_names must be a list, got {type(setting_names)}"
        metrics = self.final_evaluation(evaluation_state, variant, layer_name, fold, setting_names, fold_statistics)
        assert isinstance(metrics, dict) and list(metrics) == setting_names, f"final_evaluation must return a dict with one entry per setting name, in order, got {list(metrics)}"
        for setting_name in setting_names:
            assert set(metrics[setting_name]) == set(self.METRICS), f"final_evaluation must return exactly the metrics {self.METRICS}, got {tuple(metrics[setting_name])}"
            for metric in self.METRICS:
                assert isinstance(metrics[setting_name][metric], (float, np.floating)), f"metric {metric} must be a float, got {type(metrics[setting_name][metric])}"
        return metrics

    # -------------------------------------------------------------------------------------------------
    # GENERIC METHODS (this is where the hooks that subclasses implement in a custom way get called, through the wrappers above)
    # -------------------------------------------------------------------------------------------------

    def _setting_name(self, lr, weight_decay):
        return f"lr={lr:g},weight_decay={weight_decay:g}"

    # Trains one model per grid position on the training subset, all on the same batches (every batch is fetched once and used by every model), and returns for every grid position
    # {setting name: (metrics on the validation subset after the final epoch, per-epoch train loss, per-epoch validation loss)}.
    # The validation loss is a diagnostic only. Both subsets are prepared and standardized in the same way (with the training statistics, see run()).
    def train_models_on_fold(self, variant, layer_name, fold, train_indices, val_indices, fold_statistics):
        dataset = self._activation_datasets[layer_name]
        train_loader = DeviceLoader(dataset, self.device, train_indices, self._cfg["batch_size"], shuffle=True, num_workers=self._cfg["num_workers"])  # the training examples in a new random order every epoch, batches already on the device
        val_loader = DeviceLoader(dataset, self.device, val_indices, self._cfg["batch_size"], num_workers=self._cfg["num_workers"])

        # create one model and optimizer per grid position
        setting_names = []
        models = []
        optimizers = []
        for lr in self._cfg["lrs"]:  # grid axis 1
            for weight_decay in self._cfg["weight_decays"]:  # grid axis 2
                model = self._call_make_model(variant, fold_statistics).to(self.device)
                setting_names.append(self._setting_name(lr, weight_decay))
                models.append(model)
                optimizers.append(torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay))

        train_losses = [[] for _ in models]  # per model, one value per epoch
        val_losses = [[] for _ in models]
        for epoch in range(self._cfg["epochs"]):
            last_epoch = epoch == self._cfg["epochs"] - 1
            # train epoch
            batch_losses = [[] for _ in models]  # per model, the loss of every batch
            for batch in train_loader:
                inputs, targets = self._call_prepare_batch(batch, variant, fold_statistics)

                # forward and backward passes of every model on this batch
                for model_index, model in enumerate(models):
                    outputs = model(inputs)
                    loss = self._call_loss(outputs, targets, fold_statistics, True)  # the training loss
                    assert loss.shape == ()
                    optimizers[model_index].zero_grad()
                    loss.backward()
                    optimizers[model_index].step()
                    batch_losses[model_index].append(loss.detach())

            # track loss (train) trajactory during training
            for model_index in range(len(models)):
                train_loss = torch.stack(batch_losses[model_index]).mean()  # mean of the batch losses: the smaller last batch is over-weighted here (accepted, the loss curve is a diagnostic; the final metrics do account for it)
                assert train_loss.shape == ()
                train_losses[model_index].append(train_loss.item())

            # validation at end of train epoch
            with torch.no_grad():
                batch_losses = [[] for _ in models]
                evaluation_state = self._call_start_evaluation_collection(len(models), fold_statistics)  # collected for the metrics of the last epoch
                for batch in val_loader:
                    inputs, targets = self._call_prepare_batch(batch, variant, fold_statistics)

                    # forward pass of every model
                    outputs_per_model = []
                    for model_index, model in enumerate(models):
                        outputs = model(inputs)
                        loss = self._call_loss(outputs, targets, fold_statistics, False)  # the validation loss
                        assert loss.shape == ()
                        batch_losses[model_index].append(loss)
                        outputs_per_model.append(outputs)
                    self._call_collect_evaluation_batch(evaluation_state, outputs_per_model, targets, variant, layer_name, fold, setting_names, last_epoch)
                
                # track loss (validation) trajactory during training
                for model_index in range(len(models)):
                    val_loss = torch.stack(batch_losses[model_index]).mean()  # mean of the batch losses: the smaller last batch is over-weighted here (accepted, the loss curve is a diagnostic; the final metrics do account for it)
                    assert val_loss.shape == ()
                    val_losses[model_index].append(val_loss.item())

        # final evaluation: the models have not changed since the validation of the last epoch, so what that pass collected is the final result
        metrics = self._call_final_evaluation(evaluation_state, variant, layer_name, fold, setting_names, fold_statistics)
        results_per_setting = {}
        for model_index, setting_name in enumerate(setting_names):
            results_per_setting[setting_name] = (metrics[setting_name], train_losses[model_index], val_losses[model_index])

        return results_per_setting

    def run(self):
        variants_results = {}

        for variant in self.variants:
            variants_results[variant] = {}

            for layer_name in tqdm(self.layer_names, desc=f"{self.NAME} ({variant})"):  # iterate over layers, first to last
                dataset = self._activation_datasets[layer_name]
                grid = {}

                # initialize grid data structure
                for lr in self._cfg["lrs"]:
                    for weight_decay in self._cfg["weight_decays"]:
                        grid[self._setting_name(lr, weight_decay)] = {"folds": [], "curves": {"train_loss": [], "val_loss": []}}

                # train all folds for all grid positions
                for fold, (train_indices, val_indices) in enumerate(self.folds[variant]):
                    # DATA LEAKAGE PREVENTION: the statistics of the training fold only, prepare_batch() standardizes every batch of both subsets with them
                    fold_statistics = self._call_fold_stats(dataset, variant, train_indices)

                    # train the models of all grid positions of the current fold together: the batches are fetched once and every model trains on them
                    fold_results = self.train_models_on_fold(variant, layer_name, fold, train_indices, val_indices, fold_statistics)
                    # store results
                    for setting_name, (metrics, train_losses, val_losses) in fold_results.items():
                        grid_point = grid[setting_name]
                        grid_point["folds"].append(metrics)
                        grid_point["curves"]["train_loss"].append(train_losses)
                        grid_point["curves"]["val_loss"].append(val_losses)

                # compute mean and std over the folds, and the best grid point by the first metric
                for grid_point in grid.values():
                    grid_point["mean"] = {}
                    grid_point["std"] = {}
                    for metric in self.METRICS:
                        values_per_fold = []
                        for fold_metrics in grid_point["folds"]:
                            values_per_fold.append(fold_metrics[metric])
                        grid_point["mean"][metric] = float(np.mean(values_per_fold))
                        grid_point["std"][metric] = float(np.std(values_per_fold))
                best_setting = self.BEST(grid, key=lambda setting: grid[setting]["mean"][self.METRICS[0]])
                variants_results[variant][layer_name] = {"grid": grid, "best_setting": best_setting}

        self.results[self.VARIANTS_KEY] = variants_results
