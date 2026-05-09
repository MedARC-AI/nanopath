# PathoROB

## Role In Nanopath

`pathorob` is a robustness probe. It contributes one scalar to `mean_probe_score`: the mean of the camelyon and tolkach_esca robustness indices.

## Source

- `bifold-pathomics/PathoROB-camelyon`
- `bifold-pathomics/PathoROB-tolkach_esca`

Both are downloaded from Hugging Face by `prepare.py`.

## Data Used

| subset | tissue / organ | patches |
|---|---|---:|
| camelyon | breast cancer lymph node metastasis | 22402 |
| tolkach_esca | esophageal cancer | 16300 |

PathoROB is not a supervised train/validation fit in Nanopath, so the train and val columns are not applicable.

## Implementation

`probe.py` embeds every patch in each subset using a no-crop square resize and the frozen backbone. For each patch it concatenates the normalized CLS token with the mean normalized patch-token vector, normalizes the resulting feature, drops same-slide neighbors, and computes the PathoROB-style site-vs-biology neighbor index with fixed `k` values:

- camelyon: `k = 11`
- tolkach_esca: `k = 46`

The dataset score is the mean of the two subset indices.

## Difference From Original Usage

The original PathoROB benchmark includes multiple robustness settings. Nanopath uses the camelyon and tolkach_esca public subsets, excludes the TCGA subset, and treats the resulting robustness index as one validation-style probe scalar.

## Runtime

| model | wall |
|---|---:|
| DINOv2-S | 69.9s |
| OpenMidnight | 101.4s |
| H-optimus-0 | 99.4s |
