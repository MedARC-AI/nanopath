# PathoROB robustness protocol

PathoROB is an auxiliary check that a representation is insensitive to center
while still separating biological classes. It contributes 10% of the final
score; the five task families contribute the other 90%.

## Data and fixed adapter

| Subset | Used patches | Slides | Biological classes | Centers | Fixed k |
|---|---:|---:|---:|---:|---:|
| Camelyon | 22,402 | 97 | normal 11,205; tumor 11,197 | CWZ, LPON, RST, RUMC, UMCU | 11 |
| Tolkach ESCA | 13,800 | 62 | 6 classes, 2,300 each | UKK, WNS, CHA | 46 |

Tolkach's 2,500 TCGA-center records are excluded from both the manifest-selected
data and the downloadable snapshot. Camelyon is used as published.

The upstream sources are
[`PathoROB-camelyon`](https://huggingface.co/datasets/bifold-pathomics/PathoROB-camelyon),
[`PathoROB-tolkach_esca`](https://huggingface.co/datasets/bifold-pathomics/PathoROB-tolkach_esca),
and the [PathoROB paper](https://arxiv.org/abs/2507.17845). The nanopath mirror
retains the upstream labels and selected images under their original terms.

PathoROB intentionally does not call a model's configurable
`probe_features()`. Its fixed adapter concatenates the final normalized CLS
token and the mean normalized patch token, then L2-normalizes the result. This
keeps the robustness test comparable even when a nanopath recipe changes
test-time layer or view aggregation for task probes.

## Different-slide neighbors

Cosine neighbors from the same slide are not eligible. For each query, the
search first expands by the maximum possible number of same-slide candidates,
then retains the first fixed-`k` neighbors from other slides. There is no tuned
`k` and no learned head.

Let:

- `SO` be neighbor pairs with the same biological class and a different center;
- `OS` be pairs with a different biological class and the same center.

The published-style robustness index is:

```text
robustness_index = SO / (SO + OS)
```

A representation can raise this quantity by erasing center signal without
becoming biologically useful, so the benchmark also predicts the query's
biological class by majority vote among the same fixed neighbors and reports
class-balanced accuracy.

```text
subset_quality = (robustness_index + biological_balanced_accuracy) / 2
robustness_quality_mean = mean(Camelyon quality, Tolkach ESCA quality)
```

Raw `SO`, `OS`, robustness index, and biological balanced accuracy remain in
the result for diagnosis. Only `robustness_quality_mean` enters the final
scalar, at 10%. This prevents a center-invariant but biologically collapsed
representation from dominating genuine task improvements.
