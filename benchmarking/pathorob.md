# CRoMa robustness protocol

Robustness measures whether biological similarity dominates acquisition-center
similarity in the frozen representation. It contributes 15% of `final_score`;
the five predictive families contribute the other 85%.

## Data and fixed adapter

| Cohort | Used patches | Slides | Biological classes | Centers | CRoMa m |
|---|---:|---:|---:|---:|---:|
| Camelyon | 22,402 | 97 | normal 11,205; tumor 11,197 | CWZ, LPON, RST, RUMC, UMCU | 5 |
| Tolkach ESCA | 13,800 | 62 | 6 classes, 2,300 each | UKK, WNS, CHA | 5 |

Tolkach's 2,500 TCGA-center records are excluded from both the scored data and
the downloadable snapshot. This revision retains nanopath's existing all-row
Camelyon and non-TCGA Tolkach cohorts; it changes the metric, not the images.

The upstream sources are
[`PathoROB-camelyon`](https://huggingface.co/datasets/bifold-pathomics/PathoROB-camelyon),
[`PathoROB-tolkach_esca`](https://huggingface.co/datasets/bifold-pathomics/PathoROB-tolkach_esca),
the [PathoROB paper](https://arxiv.org/abs/2507.17845), and the
[CRoMa implementation](https://github.com/clemsgrs/croma/tree/3f58d5e4bd9ecf74c34d0c76eb88d80cee9fb706)
at commit `3f58d5e4bd9ecf74c34d0c76eb88d80cee9fb706`. The nanopath mirror retains the
upstream labels and selected images under their original terms.

The robustness probe intentionally does not call a model's configurable
`probe_features()`. Its fixed adapter concatenates the final normalized CLS
token and the mean normalized patch token. CRoMa L2-normalizes this vector before
computing cosine distance. This keeps the test comparable even when a nanopath
recipe changes test-time layer or view aggregation for task probes.

## Cross-confounder margin

Cosine neighbors from the same slide are never eligible. For each query tile,
the search finds the nearest five neighbors of each informative type:

Let:

- `SO` be neighbor pairs with the same biological class and a different center;
- `OS` be pairs with a different biological class and the same center.

Let `d_SO` and `d_OS` be the mean cosine distances to those two sets. The signed
per-tile CRoMa margin is:

```text
croma_i = (d_OS - d_SO) / (d_OS + d_SO)
```

Positive values are biology-dominant, negative values are center-dominant, and
zero is contested. Following CRoMa's headline protocol, `m=5` and the cohort
score is the median over every tile. Every tile must have five eligible `SO` and
five eligible `OS` neighbors; missing support or a zero distance denominator
fails the probe rather than silently changing the evaluated population.

CRoMa itself is signed on `[-1, 1]`. Only for combination with the other 0–1
families, nanopath maps each cohort median to `(1 + croma) / 2` and averages the
two cohorts:

```text
robustness = mean((1 + Camelyon median CRoMa) / 2,
                  (1 + Tolkach median CRoMa) / 2)
```

This `robustness` value enters `final_score` at 15%. Signed cohort medians,
10th-percentile thresholds, lower-tail means, and fractions at or below zero
remain in the result for diagnosis. The implementation mirrors CRoMa's cosine
distance and summary arithmetic in float64 inside the existing chunked GPU
search, so no new runtime dependency is required.

## Why this replaces the Robustness Index

The fixed-k RI pools informative neighbor counts, so tiles with no `SO`/`OS`
neighbors contribute nothing and models can be compared on different effective
populations. CRoMa uses every requested tile and preserves relative distance
separation. Its median still hides vulnerable tails, which is why `croma_ltm10`
and `croma_f0` are retained. Neither metric proves downstream generalization.

On the frozen 20-model panel at the new weights, replacing quality-adjusted RI
with CRoMa changes THUNDER/HEST/CPTAC Kendall tau from 0.811/0.779/0.674 to
0.800/0.789/0.684. Pearson correlations change from 0.931/0.922/0.849 to
0.927/0.929/0.814. This is mixed retrospective evidence, not uniform empirical
superiority; adoption rests on the comparable population and geometry.
