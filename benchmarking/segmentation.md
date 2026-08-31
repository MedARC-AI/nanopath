# Segmentation protocol

The segmentation family trains THUNDER-style MaskTransformer heads on frozen
dense patch tokens and scores official development validation data with
THUNDER's current per-image metric. OCELOT is excluded because it is entirely
TCGA. MoNuSAC is also excluded because its released cohort is TCGA-derived.

## Development data

| Task | Classes | Train | Validation | Boundary |
|---|---:|---:|---:|---|
| PanNuke | background + 5 nucleus classes | 2,656 images | 2,523 images | Complete Fold1 trains; complete Fold2 validates; Fold3 is absent |
| SegPath epithelial | background / epithelium | 32,768 crops from 8,192 source images | 4,518 crops from 2,259 source images | Four train crops and two validation crops per selected source image; source sets are disjoint |
| SegPath lymphocytes | background / lymphocyte | 20,906 crops from 10,453 source images | 2,164 crops from 1,082 source images | Two crops per selected source image; source sets are disjoint |

SegPath records are deterministic seed-1337 subsets of THUNDER's official train
and validation splits. Selection is stratified by foreground presence and
dominant foreground class where applicable. Crops from one source image never
cross train/validation boundaries. THUNDER's separate test records are absent.

PanNuke is the explicit provenance exception. Its released folds combine TCGA
and local-hospital images without a reliable per-image origin field. NanoPath
therefore cannot make Fold2 demonstrably TCGA-free. Fold1/Fold2 were retained
only after the two SegPath tasks alone did not preserve the desired
NanoPath-family ordering against official THUNDER segmentation. This possible
pretraining overlap must be considered when interpreting the segmentation
component; Fold3 remains fully sealed.

Task definitions and official development splits follow
[THUNDER](https://mics-lab.github.io/thunder/). Primary data references are the
[PanNuke release](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/) and
SegPath's [epithelial](https://zenodo.org/records/7412731) and
[lymphocyte](https://zenodo.org/records/7412529) releases. Original dataset
terms continue to govern the mirrored selected assets.

## Dense feature cache

The frozen encoder runs under fp16 autocast. `encode_image()` supplies patch
tokens, register tokens are removed, and all remaining model-defined feature
channels are retained. A recipe may concatenate layers or views. If that also
expands the spatial token grid, adaptive area pooling returns it to the native
`image_size / patch_size` grid so decoder attention does not grow quadratically.

Each patch vector is cached as signed int8 with its own fp16 absolute-maximum
scale and dequantized in decoder batches. This is a storage/runtime device, not
a learned projection. Labels are kept at source resolution and invalid label
values are masked.

## Decoder and training

Every task uses the same two-layer, eight-head pre-LayerNorm MaskTransformer
structure as THUNDER: patch projection, learned class tokens, transformer
blocks, normalized patch/class projections, and cosine mask logits. The
backbone remains frozen.

| Task | Epochs | Adam learning rate | Weight decay | Classes |
|---|---:|---:|---:|---:|
| PanNuke | 30 | 1e-3 | 1e-4 | 6 |
| SegPath epithelial | 9 | 1e-4 | 1e-3 | 2 |
| SegPath lymphocytes | 21 | 1e-3 | 1e-4 | 2 |

Batch size is 64 and the objective is THUNDER's multiclass soft Dice loss. The
fixed final epoch is scored once; validation never selects a checkpoint,
learning rate, or decay. Decoder initialization and training use seed 0. Only
decoder attention is forced through PyTorch's deterministic math SDPA kernel;
the frozen backbone retains its ordinary inference path.

THUNDER's published decoder uses `d_model=768` and `d_ff=3072`. NanoPath uses
`d_model=192` and `d_ff=768` with the same two-layer/eight-head structure. This
is the principal capacity deviation needed to keep the complete evaluation
under 25 minutes. Controlled wider/longer decoder trials did not improve the
ordering proxy and exceeded or eroded the runtime margin.

## Metric

For every validation image and every class, compute pixel counts `TP`, `FP`, and
`FN`. A class is present when `TP + FP + FN > 0`; absent classes do not enter
that image's class mean.

```text
F1_c      = 2 TP / (2 TP + FP + FN)
Jaccard_c = TP / (TP + FP + FN)
image metric = mean over present classes
```

Each image receives its valid-pixel count as a base weight. To reproduce
THUNDER's background-only balancing, every image containing foreground is
additionally multiplied by
`max(1, 16 * fraction_of_validation_images_that_are_background_only)`. The
task result is the weighted mean over images. Macro-F1 is scored;
macro-Jaccard is retained only as a diagnostic. `seg_mean_f1` is the unweighted
mean of the three task F1 values.

This differs materially from v1, which used MoNuSAC, CoNSeP, and PanNuke
Jaccard. That panel was neither dataset-matched nor metric-matched to current
THUNDER and had negative correlation with official segmentation. Protocol v2
uses two exact current THUNDER tasks, the same current metric, and PanNuke as a
mixed-source diversity task. See [validation.md](validation.md) for the observed
ordering fidelity and its remaining limitations.
