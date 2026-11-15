# Interpretability module

`python -m interpret.interpret configs/main.yaml checkpoint_path=/path/to/checkpoint.pt [baseline_checkpoint_path=/path/to/init.pt] activation_cache_dir=/path/to/interpret results_dir=/path/to/results` (from the repo root) reads the `interpret:` block of the config: it
extracts the hooked activations of a seeded, stratified sample of a dataset into files (one `ActivationDataset` per layer, see Output), then runs the configured
`methods` on them and saves their results and plots. When `baseline_checkpoint_path` is provided, both the baseline and trained checkpoints are analyzed and compared side-by-side in `dashboard.html`. Everything runs on one device, the GPU: `main()` picks it, moves the model and gives it to the activation datasets, whose `get_block` and the `DeviceLoader` (both in `data.py`) are the only places that move data to it,
so the methods never move data themselves. A run needs a GPU, a
Nanopath checkpoint (it carries its training `config`, its `step` and the `model_ema`/`model` weights),
and `probe.dataset_roots` pointing at the data.

**SECTIONS**:
- [Running the rig](#running-the-rig)
- [Methods](#methods)
- [Config](#config-interpret-in-configsmainyaml)
- [Output](#output)
- [Integration into Labless](#nanopath-and-labless-flow)
- [Hook contract](#hook-contract)
- [Your to-do list to analyse a new model](#your-to-do-list-to-analyse-a-new-model)
- [Extending the rig](#extending-the-rig)
- [Notes](#notes)

## Running the rig

There is one interpretation rig and one input/output contract, not separate standalone and chained modes. Both callers provide the same settings YAML and the same required runtime paths: `checkpoint_path`, `activation_cache_dir`, and `results_dir`. The difference is only who chooses those paths.

| Caller workflow | Runtime paths | Result |
|---|---|---|
| Standalone analysis | The user or agent explicitly supplies a saved checkpoint (and optionally an untrained baseline checkpoint), extraction-cache base, and a dedicated results directory for this invocation. | The rig analyzes the checkpoint(s) and writes extraction, results, effective config, and `dashboard.html` to the chosen locations. |
| Chained after training | SLURM passes the enable/disable setting to `train.py`; after training and probes, `train.py` supplies the untrained baseline (`init.pt`), the final checkpoint (`latest.pt`), and run-local output locations. | The same rig analyzes both checkpoints, compares them side by side in `dashboard.html`, and writes results under that run's output directory. |

The intended command shape is the same in both workflows:

```bash
python -m interpret.interpret configs/main.yaml \
  checkpoint_path=/path/to/checkpoint.pt \
  baseline_checkpoint_path=/path/to/init.pt \
  activation_cache_dir=/path/to/interpret \
  results_dir=/path/to/interpret/results
```

`baseline_checkpoint_path` is optional. When supplied (as `train.py` does automatically using `output_dir/init.pt`), the interpreter runs both checkpoints and generates a side-by-side comparison dashboard comparing the untrained baseline against the trained model. When omitted, it runs single-checkpoint analysis.

`activation_cache_dir` is the base directory for temporary extracted activations, images, and labels, stored under `data/<hash>` and deleted after successful analysis. `results_dir` is the exact output directory for this invocation and contains `dashboard.html`; the interpreter replaces it at the start of each invocation, so use a dedicated results directory. In the chained workflow, `train.py` uses `${OUTPUT_DIR}/interpret` for the activation cache and `${OUTPUT_DIR}/interpret/results` for results. If interpretation is skipped, existing files are left untouched and no success flag is passed to the submitter. The settings YAML contains analysis choices, not a default checkpoint. Resolved runtime values are recorded in the effective config; the shared source YAML is not rewritten.

## Methods

Each method below has its own subsection: the questions its results can answer first, then the papers behind it and the author's
notes on the paper(s). What a method computes is in its code (`interpret/methods/<method>.py`).

### effective_rank

Questions this method answers:
1. How many of its dimensions does each layer effectively use, and how does that change with depth (separately for the mean-over-patch-tokens and the CLS vector)?
2. [From the RankMe paper, not yet validated in our context] Is ER a label-free proxy for downstream performance
   (i.e. we could try to correlate it with our obtained downstream scores in an effort to test the 2022 paper's premise in our context of ViT in DP)?

Papers:
- [Roy & Vetterli 2007](https://infoscience.epfl.ch/record/110188): introduced the effective rank (ER) definition.
- [RankMe, Garrido et al. 2022](https://arxiv.org/abs/2210.02885): applied it in the context of modern SSL.

Notes on the paper(s):
- Rank of embeddings collected for a certain layer across datapoints tells us "how many of its dimensions is this layer actually occupying".
  It will very rarely be the case that one embedding is an exact linear combination of other collected embeddings (analogously it will very rarely
  be the case that a singular value of the embedding matrix will be exactly zero), i.e. the rank of a real world embedding matrix will almost always
  be full even if some dimensions are very close to being unused. Effective rank aims to be more robust: it normalizes singular values to a
  probability distribution and then computes its exponentiated entropy (how uncertain are we about a draw from this distribution), i.e. if one
  singular value dominates (even if the others are non-zero) entropy is close to 0 (low uncertainty) (ER=1, rank would have been full unless the
  other sing vals are exactly 0); if sing vals are approx. uniformly distributed entropy is close to log(k) (ER=k).
- Centering: the embedding cloud is centered (inspired by PCA, not necessarily the linked papers). Without it a strong offset of the whole cloud
  from the origin gives one very large singular value, which can dominate the normalized distribution and can push the entropy (and ER) down even
  if the variation inside the cloud is spread evenly. After centering, the singular values only describe the variation within the cloud.

### cka

Question this method answers: 
- given layer A and layer B, how similar is the *structure* of their embedding spaces, i.e. do the two layers organize the examples in the same way?

Papers:
- [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414): centered kernel alignment (CKA) for comparing layers of neural networks (Section 3: the definition, Section 2: why these invariances, Section 6.1: sanity check).

Notes on the paper(s):

earlier layer-comparison indexes (CCA and relatives) are invariant to any invertible linear transformation, which the paper argues makes them unable to measure meaningful similarity once a layer is at least as wide as the number of examples (and ignores the scale of directions, which is consistent across networks); CKA is invariant only to rotations and uniform scaling. In the paper's sanity check on CNNs it finds the matching layers of networks trained from different initializations, unlike the other indexes; on Transformers (equal widths, like our ViT) all indexes pass it, but CCA misses feature-scale differences between attention and feed-forward sublayers.

Summary of the derivation of the CKA formula:

| Derivation steps for CKA (done in the paper) | Matrix notation |
|---|---|
| **Step 0: notation**<br>we compare layer A and layer B on the same n examples; each layer turns every example into one vector of C numbers. | `X` (n × C) holds layer A's activations, one row `xᵢ` per example; `Y` holds layer B's, same n examples. |
| **Step 1: what should a similarity index be invariant to?** (Section 2)<br>orthogonal transformation (which includes permutations of channels, a symmetry of neural networks) and isotropic scaling, but not invertible linear transformation (an index with that invariance cannot tell any two full-rank layers apart if both are at least as wide as the number of examples, and training is sensitive to the scale of directions). | `s(X, Y) = s(XU, YV)` for `UᵀU = I`, `VᵀV = I` and `s(X, Y) = s(αX, βY)`, but not `s(XA, YB)` for arbitrary invertible `A`, `B`. |
| **Step 2: how do we want to describe the space that a layer produces (so we can compare these across layers)** (Section 3, key insight; the symmetry argument is in Section 2.2)<br>*Naive:* compare the raw embeddings of the two layers directly. *Problem:* the coordinates of two layers are not comparable. A network may permute, rotate or mix its channels from layer to layer, so channel *i* of one layer need not mean anything for channel *i* of another. *Fix:* go up one level of abstraction and describe a layer by its *similarity table*: an n × n table with one entry per pair of examples, the dot product of their vectors, saying how similar the layer considers them (called a representational similarity matrix in the paper). The table does not depend on the axes, so two layers that arrange the examples the same way produce the same table even when their raw numbers look unrelated. The same n examples go through both layers, so the tables can be compared entry by entry. | `K = XXᵀ`, `L = YYᵀ`. Entry (i, j) of `K` is row i of `X` times column j of `Xᵀ`, i.e. the dot product `xᵢ·xⱼ`. |
| **Step 3: how to compare the two tables** (Eq. 1)<br>use the inner product of the tables flattened into 1D. The paper shows that this inner product also equals the sum of the squared dot products between every pair of features of `X` and `Y` (one feature of each, with the dot product taken over the examples), so the agreement between the two tables can also be computed from the features. | `⟨vec(XXᵀ), vec(YYᵀ)⟩ = tr(XXᵀYYᵀ) = ‖YᵀX‖²_F` (`vec` flattens a matrix, the trace `tr(M)` is the sum of the diagonal and for symmetric `A`, `B` `tr(AB) = Σᵢⱼ Aᵢⱼ Bᵢⱼ`, the entry-by-entry product summed up; `‖M‖_F = sqrt(Σᵢⱼ Mᵢⱼ²)` is the length of the flattened matrix). |
| **Step 4: centered data gives a covariance reading** (Eq. 2)<br>for centered `X` and `Y` (every feature has mean 0 across the examples; the paper states this as a condition and does not comment on it here) Eq. 1 is the squared size of the cross-covariance between the features of the two layers, up to the covariance normalization (the Bessel correction `n−1` of a covariance, squared here because the covariance is squared: `1/(n−1)²`). | Start from Eq. 1: `tr(XXᵀYYᵀ) = ‖YᵀX‖²_F`. Take `X`, `Y` centered (every column has mean 0). The cross-covariance of their features, with the Bessel correction `n−1`, is `cov(Xᵀ, Yᵀ) = XᵀY/(n−1)`. Divide Eq. 1 by `(n−1)²`: `tr(XXᵀYYᵀ)/(n−1)² = ‖YᵀX‖²_F/(n−1)² = ‖XᵀY/(n−1)‖²_F = ‖cov(Xᵀ, Yᵀ)‖²_F` (Eq. 2; transposing a matrix does not change its Frobenius norm). |
| **Step 5: HSIC** (Eq. 3)<br>Eq. 2 is the formula for the agreement between the two tables when the similarity is the dot product and the features are centered. Raw activations are not centered, so the formula has to do the centering itself: it is applied to `HX` and `HY` (for data that is already centered, `HX = X` and `H` changes nothing). The Hilbert-Schmidt Independence Criterion (Gretton et al. 2005) is the established general version for any kernel (= a similarity function), with the centering done on the similarity tables themselves (`HKH`) instead of on the features. The paper introduces the centering matrix `H` here. For the linear kernel its value equals the value of Eq. 2, so nothing changes for us - I think the reason the paper talks about HSIC at all is s.t. they introduce the form that accepts an arbitrary kernel (the paper does swap kernels in results section, we are not planning to use kernels other than the linear one). (our intuition why centering is beneficial: without centering, a constant offset shared by all examples, e.g. the color signature of one scanner, adds a potentially large component to every table entry; adding the same offset to two vectors pulls their angle towards 0 (and the normalized version of the step 3 measure, introduced in step 6, is a cosine, i.e. a measure of angle), e.g. (1, 0) and (0, 1) shifted by (10, 10) are only about 5° apart, so all layers would look similar under the normalized measure from step 6.) | `H = I − (1/n)11ᵀ` is the centering matrix (`1` is a column of n ones): `(1/n)1ᵀX` is the mean example `μᵀ`, so `HX = X − 1μᵀ` is every example minus the mean example. Start from Eq. 2 for the centered examples `HX`, `HY`: `tr((HX)(HX)ᵀ · (HY)(HY)ᵀ) / (n−1)²`. The table of the centered examples is `(HX)(HX)ᵀ = HXXᵀH = HKH` (H is symmetric), likewise `HLH`. So the expression is `tr(HKH·HLH) / (n−1)²`. Since `HH = H` (centering twice is the same as once) and the trace lets one H move around, `tr(HKH·HLH) = tr(KHLH)`. End point: `tr(KHLH) / (n−1)² = HSIC(K, L)`, the paper's Eq. 3. |
| **Step 6: normalize** (Eq. 4)<br>*Problem:* HSIC is not invariant to isotropic scaling. *Fix:* normalize it, which gives centered kernel alignment (CKA; Cortes et al. 2012, Cristianini et al. 2002). | Start from Eq. 3, `HSIC(K, L) = tr(KHLH)/(n−1)²`, and divide by the square root of the product of the HSICs of each table with itself, `HSIC(K, K) = tr(KHKH)/(n−1)²` and `HSIC(L, L) = tr(LHLH)/(n−1)²`: `CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K)·HSIC(L, L)) = [tr(KHLH)/(n−1)²] / [sqrt(tr(KHKH)·tr(LHLH))/(n−1)²] = tr(KHLH) / sqrt(tr(KHKH)·tr(LHLH))` (Eq. 4): the `(n−1)²` cancel. Since `‖M‖²_F = tr(MM)` for symmetric `M`, `tr(KHKH) = ‖HKH‖²_F` is the squared length of the flattened centered table, so CKA is the inner product of the two centered tables divided by their two lengths. |

*TLDR.* The key points of the above derivation: 
- Compare similarity structures, not features: the coordinates of two layers are not comparable (channels can be permuted, rotated or mixed), but how each layer arranges the examples is. This is the key reframe.
- Flatten the two similarity tables and take their inner product.
- Center first, so a constant offset shared by all examples cannot dominate the tables (our intuition; the paper only states the condition and defines `H`).
- Normalize, which adds the invariance to uniform scaling that the raw inner product lacks.
- HSIC is the paper's link to the general kernel version; we only use the linear kernel.
 
Ultimately, the normalized inner product of the centered similarity tables is **CKA**. It is unchanged by rotating the axes of a layer or scaling it uniformly, and is not invariant to arbitrary invertible linear maps (which the paper argues is desirable).

*sidenote.* The paper also gives a cheaper form for the linear kernel (regroup the sum by feature: `CKA = ‖YcᵀXc‖²_F / (‖XcᵀXc‖_F · ‖YcᵀYc‖_F)` with `Xc = HX`, `Yc = HY`, which needs only C × C matrices instead of n × n tables; Section 3 and Table 1 of the paper). We omit it: at our sample sizes the n × n version is still tractable, and we want to focus on methods, not on computational tricks - unless we have a good reason to read about and implement it (in which case we will do so in the future).

### linear_probe

Questions this method answers:
- At which layers can a linear probe recover high-level labels (e.g. tissue class) and low-level labels (e.g. site/stain), and how does this change with depth?

### layer_inversion

Questions this method answers:
- How much of the tile (colour, layout, texture, biological structures) can a decoder rebuild from each layer, and how does this change with depth?
- Is the tile information of a layer local to each patch-token or does it depend on the other patch-tokens? The linear decoder sees only the token of its own patch (the local baseline), the transformer decoder exchanges information between all tokens, so the gap between them is the share that needs context (it also contains the effect of the transformer's nonlinearity).

Papers:
- An extensive survey of the field with more relevant papers is in `layer_inversion_literature.md`. 
- [Dosovitskiy & Brox 2016](https://arxiv.org/abs/1506.02753): a decoder trained per layer to rebuild the input image from that layer's activations.
- [He et al. 2022 (MAE)](https://arxiv.org/abs/2111.06377): the per-patch linear head and the transformer decoder on ViT tokens.
- [Xu et al. 2020](https://arxiv.org/abs/2002.10689): why a reconstruction measures the information a given decoder family can extract, and not all the information of the layer.

### relax

Questions this method answers:
- Which regions of an image does its embedding depend on?

Papers:
- [Wickstrøm et al. 2023 (RELAX)](https://arxiv.org/abs/2112.10161): attributes an embedding, not a class score, to the regions of its input image: mask the image, embed it again and compare to the original embedding with cosine similarity.
- [Petsiuk et al. 2018 (RISE)](https://arxiv.org/abs/1806.07421): the random-mask attribution for classifiers that RELAX adapts. The masking scheme comes from here.

Notes on the paper(s):

The "black-box" argument of RISE (forward passes only, no gradients) separates it from gradient saliency methods. 

RELAX takes the scheme from RISE and gives almost no argument of its own; this is one row per design decision of the masks and what the papers give as the reason:

| Design decision of the masks | Justification in the papers |
|---|---|
| The similarity is weighted by the mask value of the pixel: R_ij = E_M[s(h, h') · M_ij] (RELAX Eq. 1) | Credit assignment: every mask achieves one similarity score (high = the embedding stayed close to the original), and that score, high or low, flows into the pixels in proportion to how little the mask tampered with them (M_ij = 1 untouched, 0 hidden). A pixel the mask hid gets nothing of it (because the original pixel value was barely/not at all present in the forward pass i.e. it could not affect the result of the forward pass i.e. it does not make sense to credit this pixel for the high/low similarity score the mask caused), a barely masked pixel gets almost all of it (because it's original value was largely intact i.e. that value could influence the forward pass and the resulting similarity score). |
| Random combinations of different regions hidden at once, not one region at a time (e.g. a sliding square masking out a certain region of the image) | • Our reading: the importance of a pixel is defined as an expectation over a distribution of random masks (RISE Sec. 3.1, RELAX Equation 1). A natural estimator (computing the expectation exactly is intractable) is to sample masks from that distribution and average (= Monte Carlo). In theory a sliding window mask is also a possible outcome of that distribution, but the windows form one fixed, atypical set (one hidden block, where a typical random mask hides about half the grid) and not independent random draws. Averaging over them therefore does not converge to the expectation, it estimates a different quantity: the effect of hiding one block with everything else visible. So once the importance is defined as this expectation and approximated with Monte Carlo, the masks have to be random and not structured like a sliding window. <br>• empirically: RISE beats a sliding-window occlusion baseline (64 px windows, stride 8) on deletion and insertion for VGG16 and ResNet50 on ImageNet (Table 1).<br>• Earlier work (secondary sources, originals not read) also seems to favour random masks: LIME (2016) - switches off random subsets of superpixels at once; Shapley view - one-region occlusion is a single coalition, so a region that duplicates the evidence of another looks unimportant. Averaging over many coalitions exposes it, and since all coalitions are infeasible for images they are sampled (Kernel SHAP).<br> |
| A low-resolution grid of 0s and 1s that is then smoothly upscaled to the image dimensions, not independently sampled mask values on the scale of the image | • Independent pixels give a huge mask space, which needs many more samples to approximate the expectation in Equation 1 of the RELAX paper  (RISE Sec. 3.2, RELAX Sec. 3.1).<br>• Sharp changes of single pixels may act as adversarial noise. Smooth (bilinear) upscaling does not introduce sharp edges, which avoids this and also gives a smooth importance map (RISE Sec. 3.2).<br>• Without it, smooth masks covering different portions of the image are hard to create (RELAX Sec. 3.1). |
| The upscaled mask is one grid square larger than the image, and an image-sized window is cut out of it at a random position (instead of always at the same corner), which shifts the mask randomly against the image | • RISE: the masks are shifted by a random number of pixels "to allow more flexible masking" (Sec. 3.2).<br>• RELAX: describes the same step as cropping the mask "randomly" and gives no reason why the position is random and not fixed (Sec. 3.1). |
| 7×7 grid, 224 px images, p = 0.5 | Chosen empirically. RISE uses equal probability for 0 and 1 (Sec. 4.2), RELAX follows an earlier work for the grid (Sec. 5) and never states p. |
| Number of masks | RISE chose it empirically (4000 for VGG16, 8000 for ResNet50). RELAX uses 3000, justified by an estimation-error bound (Theorem 1) that the authors call loose. |

### tsne

Questions this method answers:
- How is the space of each layer shaped (clusters, density, outliers), which labels organise it (class, centre, stain), and how does this change with depth? It visually displays, per layer, the geometry that CKA reduces to one number per layer pair.

Papers:
- [van der Maaten & Hinton 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)

Notes on the paper(s):

Why did we pick t-SNE and not UMAP: with the same informative initialisation both preserve structure about equally well (Kobak & Linderman 2021), and t-SNE has one key hyperparameter (perplexity) instead of two (so in case of t-SNE sweeping over the key hperparameters results in less plots that people need to inspect and pick from).

SNE (the predecessor, assumed known) gives cluttered 2-D maps that are hard to optimise. The paper's section on t-SNE (section 3) fixes this in four steps, one row per subsection:

| Subsection | What it improves over SNE, and how |
|---|---|
| Symmetric SNE | SNE sums one KL divergence per point over the conditional probabilities p_j\|i, which gives a gradient that is expensive to compute. Symmetric SNE uses one joint distribution over all pairs, p_ij = (p_j\|i + p_i\|j) / 2n: the gradient becomes simple and fast. They make sure to pick the definition of the joint distribution s.t. every point, even an outlier, keeps a share of at least 1/2n of the probability mass of the joint distribution, so it still makes a significant contribution to the cost function and is therefore not "stranded" (= not moved much) in the optimiztation (which could happen with alternative definitions of the joint distribution). |
| The crowding problem | The diagnosis of SNE's main flaw: the 2-D map has far less room than the original space (in 10 dimensions 11 points can all be equally far apart from each other, on a 2-D plane at most 3). So the neighbours of a point cannot all keep their distance on the 2-D map. SNE keeps the near neighbours accurate (it places them at the right short distances, s.t. their q_ij match their p_ij), because the KL cost weights the mismatch of every pair by its p_ij and the near neighbours carry most of the probability mass: getting them right dominates the cost, while low-mass pairs are cheap to get wrong. But then there is not enough room left at the right moderate distance, so most of the moderately distant points end up much further away than they should be. There the Gaussian q_ij is almost zero while their p_ij is small but not zero (p_ij > q_ij) - due to this mismatch each of them still pulls on the point with a very small attracting force; since there are so many of them that together they clump the points into the centre of the 2-D map, s.t. no gaps can form between the natural clusters. |
| Mismatched tails can compensate for mismatched dimensionalities | The fix: keep the Gaussian (light tails) for p_ij in the original space, but compute q_ij on the 2-D map with a Student-t distribution with one degree of freedom, whose tails fall off with the square of the distance instead of exponentially. The p_ij stay exactly as before; what changes is the 2-D map: the heavy tail gives the 2-D map more probability mass at large distances. A moderately distant pair that the lack of room pushes far out on the 2-D map therefore still gets a q_ij about as large as its p_ij there (with a Gaussian it would get q_ij ≈ 0), so the geometric mismatch stays and is accepted (the point still sits further out than in the original space, since the 2-D map has no room for it at the right distance), but there is no probability mismatch any more, and the KL cost only compares probabilities, so the pull back inwards disappears: the moderately distant points can stay far out, which gives the room the 2-D map lacks and opens gaps between the clusters. In short, for the same probability the Student-t allows a much larger distance on the 2-D map than the Gaussian. Side benefits: q_ij has no exponential, so it is cheaper to compute. TLDR: the lack of room is never solved, only made invisible to the cost: the moderately distant points still sit too far out on the 2-D map (here 384 dimensions are squeezed into 2), but with the heavy tail this no longer shows up as a probability mismatch between P and Q, so the KL divergence has nothing to pick up and creates no force that crushes the clumps together. |
| Optimization methods for t-SNE | Plain gradient descent from a small random start, with momentum (0.5 for the first 250 iterations, then 0.8) and an adaptive learning rate (a gain per parameter that grows while its gradient keeps its sign), for 1000 iterations (the paper's values). Plus early exaggeration: in the first 50 iterations all p_ij are multiplied by 4, s.t. the q_ij (which still sum to 1) cannot match them and the optimisation forms tight, widely separated clusters early; the clusters then have empty space around them to move into their final arrangement. (sklearn exaggerates by 12 and starts from PCA.) |

## Config (`interpret:` in `configs/main.yaml`)

- `model_weights` (the checkpoint key: `model_ema` for the teacher, `model` for the student): which weights to analyse. `checkpoint_path`, `activation_cache_dir`, and `results_dir` are required command-line paths, not YAML settings.
- `dataset` + `datasets.<id>.stratification_dimension`: one dataset per run (classes in the source-dataset section of `data.py`). Available:
  `pathorob_camelyon` (PathoROB Camelyon tiles, root `probe.dataset_roots.pathorob`), with the metadata fields
  `biological_class`, `medical_center` and `slide_id` to stratify by. The sampler is sklearn's `resample` with
  `replace=False` and `stratify=`: a seeded subsample that keeps the proportions of the stratification dimension.
  `num_samples` must be smaller than the dataset size.
- `seed`: the one seed of the interpret config. `set_seeds` in `main()` sets it once for every random number generator, plus deterministic GPU mode (an operation without a deterministic implementation raises an error), and nothing else in the rig seeds, with one exception: the fold splitter of `Probe`, so that all variants of a method share their folds (the seeding policy is at the `set_seeds` call in `interpret.py`).
- `extract.{num_samples, batch_size, num_workers, layer_stride}`: `layer_stride` keeps every n-th
  layer counted from the last (the last is always kept); `num_workers` only parallelizes image decoding.
- `methods`: the methods to run (`effective_rank`, `cka`, `linear_probe`, `layer_inversion`, `relax`, `tsne`); `cka.max_samples` caps the examples CKA uses (a seeded random subset, label-free so not exactly stratified), because its similarity tables are n × n (0.1 GB each at 5,000 examples, 0.4 GB at 10,000, 2.5 GB at 25,000); the other methods use all extracted examples.
- `linear_probe`: `targets` (what to predict: the metadata fields and the labels derived from the tile pixels, `brightness`, `tissue_brightness`, `tissue_fraction`, `hematoxylin` and `eosin`, each cut into 4 quantile bins), `lrs` and `weight_decays` (the grid), `epochs`, `batch_size`, `num_workers` (the DataLoader worker processes that fetch the batches), `leakage_group` (the metadata field whose values are never split across train and validation, `slide_id` for PathoROB) and `folds` (a `StratifiedGroupKFold` on that field).
- `layer_inversion`: `decoders` (`linear`: one linear head shared by all patch tokens, `transformer`: transformer blocks over the CLS and patch tokens, then the same head), `transformer` (the settings of that decoder: `depth` and, per `Block`, `head_dim`, `mlp_ratio` and `drop_path`), `lrs` and `weight_decays` (the grid), `epochs`, `batch_size`, `num_workers`, `num_collage_rows` (the rows of the collage), `stratify` (the metadata field the folds are stratified on), `leakage_group` and `folds` (as for the probe). The decoders train on the patch tokens and the images of every layer, so the datasets are opened with `patch_tokens=True, image=True`.
- `relax`: `num_masks` (random masks per image), `mask_batch_size` (masked images per forward pass), `num_images` (tiles evenly spaced through the dataset, a fixed stride with no randomness, so every run and every checkpoint is analysed on the same tiles), and the paper's settings `grid_side_length` (the masks start as a grid of that many squares per side), `visible_probability` (of a grid square) and `uncertainty_threshold_scale` (gamma of U-RELAX: the importance is kept where the uncertainty is below gamma times its median). RELAX runs the model itself on masked copies of the images, so unlike the other methods it does not read the extraction: it gets the model, the source dataset and the extraction's transform (the tile in [0, 1] and the normalised model input) from `main()`.
- `tsne`: `num_tiles` (tiles evenly spaced through the extraction, a fixed stride with no randomness, the same tiles in every layer and for every checkpoint) and `perplexity` (roughly the number of neighbours each point considers). Every vector type of every kept layer gets its own 2-D map, and every label of the extraction gets one figure.

Storage per sample is `kept_layers x 257 tokens x C x 4 bytes` (float32) plus the 0.6 MB input image; for ViT-S (C=384)
that is about 3.4 MB at stride 2 and 2.2 MB at stride 4, so 10,000 samples take about 34 GB / 22 GB on disk during analysis. The extraction holds everything in RAM before saving it
(as much as the files take on disk, plus the model and loader), so run it on a high-RAM runtime. After a successful interpretation, the activation cache is deleted; final results, plots, and the dashboard remain.

## Output

```
activation_cache_dir/            # activation cache base; cache is temporary and deleted after success
  data/
    <hash>/                      # hash of everything that determines the files (checkpoint + step, weights, dataset + its config, seed, num_samples, layer_stride); may remain if interrupted
      activations/
        <layer>_patch_tokens.pt  # (N, 256, C) float32, the patch tokens of that layer: the big files (about 3.9 GB each for 10,000 samples and C=384)
        <layer>_cls.pt           # (N, C) float32, the CLS token of that layer
      precomputed/
        <layer>_patch_mean.pt    # (N, C) float32, the mean over the patch tokens of that layer, computed once during the extraction
      images.pt                  # (N, 3, H, W) float32, the tile as the model saw it (resized, in [0, 1], before the normalization of the model input; about 6 GB for 10,000 samples)
      labels.pt                  # {field: (N,) int64}, every metadata field as integers plus the labels derived from the tile pixels (brightness, tissue_brightness, tissue_fraction, hematoxylin, eosin: quartile bins)
      label_names.json           # {field: names}, the name of integer k is names[k] (sorted unique values of the sampled tiles; "0-25%" .. "75-100%" for the derived labels)
      info.json                  # settings, git commit, kept layers, sampled dataset indices; written last by the extraction (marks it complete)
      config.json                # the full config that produced these files, written right after a fresh extraction
results_dir/                     # exact output directory supplied for this invocation, independent of the extraction cache
  results.pt                     # the results of all methods: their outputs (the final numbers, curves, images and maps, not the intermediates), read with torch.load(path, weights_only=False)
  dashboard.html                 # the whole run on one page, to open with a double click (see below)
  config.json                    # the effective config of this invocation, including runtime paths
  <method>/                      # the plots that method defines, e.g. cka/patch_mean_cka.png, effective_rank/patch_mean_effective_rank.png, layer_inversion/error.png, layer_inversion/collage.png and relax/image_<index>.png (one per analysed image) and tsne/<label>_tsne.png (one per label)
  baseline/                      # (optional, in comparison mode) results.pt and <method>/ plots for the baseline checkpoint
  trained/                       # (optional, in comparison mode) results.pt and <method>/ plots for the trained checkpoint
```

At the end of a run, `write_dashboard` (`utils.py`) writes `dashboard.html` into the results folder, then the interpreter attempts to delete the completed activation cache. The dashboard contains the settings of the run, then for every method its plots (embedded in the file, so it can be moved or sent around) and the parts of its subsection of this README (papers, question) as toggles. When running with both baseline and trained checkpoints, the dashboard displays a side-by-side two-column comparison (Untrained Baseline vs. Trained Model) for every method's plots and RELAX heatmaps. For RELAX, its card shows one row per image instead: the image, then each map over its own copy of the image with its own opacity slider, so they can be compared side by side (built from `results.pt`; its static figures stay in `relax/`). It splits a method's subsection at the lines starting with `Papers` and `Question`.

During extraction, row `i` of every cache file is the `i`-th sampled tile (`sampled_dataset_indices` in `info.json`), so the vectors, image, and labels of an example sit at the same index by construction.
Methods read one layer through `ActivationDataset(data_dir, layer, device)`, a `torch.utils.data.Dataset`: the constructor sends nothing to the device: the small tensors (`cls`, `patch_mean`, labels) are CPU tensors in memory, the two big files (`patch_tokens`, `image`) are opened only when enabled and memory-mapped (nothing is read until an item is fetched). Data leaves the dataset in two ways and both move it to the device (`__getitem__` itself returns one example as CPU tensors): `get_block(field, key)` returns a whole small block (`field` names a dict of the dataset, `"activations"` or `"labels"`, and `key` picks the block in it: `get_block("activations", "cls")`, `get_block("labels", "medical_center")`; what methods that need a whole layer or fold read; the big blocks are refused, they are memory-mapped so that they are never loaded whole), and `DeviceLoader(dataset, device, indices, batch_size, ...)` from `data.py` (a `DataLoader` over those examples that yields batches already on the device, the only way to read the big blocks, in batches)
returns one example as `{"image", "cls", "patch_tokens", "patch_mean", "labels"}`, always with these keys in this order: `cls`, `patch_mean` (both `(C,)`) and `labels` (`{metadata field: integer}`) are always filled, while
`image` and `patch_tokens` are only opened when enabled in the constructor (`ActivationDataset(data_dir, layer, device, patch_tokens=True, image=True)`) and are an empty tensor `torch.empty(0)` otherwise (empty means "not loaded").
All tensors are private to the dataset. Only two facts are public attributes, so that methods never hard-code them: `.label_names` (`{metadata field: names}`, the name of integer k is names[k]) and `.vector_types` (`{vector type: description}` of the vectors every item holds); the number of examples is `len(dataset)` and the labels are read with `get_block("labels", field)`, or all at once with `get_block("labels", to_device=False)` (`to_device=False` skips the device move, for the few host-side callers like sklearn and printouts). `dataset[i]` is one example as CPU tensors. The loader stacks examples on the CPU, optionally in `num_workers` worker processes, and the main process moves each whole batch to the device once (workers cannot use the GPU), which is far cheaper than moving every example.

## Integration into Labless

This ordered protocol describes each handoff and its failure behavior.

| Step | Action and data flow | Error handling |
|---|---|---|
| 1. Launch option | The SLURM launcher accepts `interpretability=false` and forwards the setting as a CLI argument; `train.py` defaults to interpretation enabled and also accepts the CLI override for direct runs. | Reject invalid values with a clear message. If explicitly disabled, print a notice, skip interpretation, and leave the existing training/evaluation and submission behavior unchanged. |
| 2. Supply runtime paths | After training, `train.py` passes the completed checkpoint path (`latest.pt`), the untrained baseline checkpoint (`init.pt`), and run-local activation-cache/results paths as required CLI arguments alongside the checked-in `interpret:` settings. These interpreter arguments are internal to Nanopath and are not sent to Labless. | Missing required arguments fail before analysis starts. A bad checkpoint or output path is an interpretation-stage failure and does not invalidate completed training/evaluation. |
| 3. Run interpretation | After training/evaluation, `train.py` invokes the standalone interpreter as a subprocess using the inputs defined in step 2. The interpreter runs both checkpoints, writes run-local analysis results, and generates a side-by-side comparison in `dashboard.html`. `train.py` then unlinks `init.pt`. | On a missing hook or another analysis error, print an actionable warning, keep details in the training process logs, and continue without an interpretation artifact. For a missing hook, name `register_interpretability_hooks` and point to the Hook contract. |
| 4. Keep local-only results | If Labless auto-submit is not armed, interpretation still runs with the required CLI paths and leaves outputs under the run directory; the interpretation dashboard is not uploaded to W&B or Labless. Training's normal W&B logging is unchanged. | Interpretation remains supplementary; its failure must not invalidate training/evaluation results. No submission is attempted unless the user already opted into the existing Labless flow. |
| 5. Handoff the dashboard to submission | After successful interpretation, `train.py` passes `interpretability_dashboard=true` to the existing submitter; otherwise the dashboard is omitted. | The submitter ignores existing dashboards unless the current interpreter process succeeds. |
| 6. Publish and display the optional dashboard | The submitter uploads dashboard HTML to a W&B interpretation run grouped with the training run, then sends its URL as an `interpretability` artifact. The Labless API stores it and the run page displays its kind and clickable URL alongside the training result. | W&B publication failure omits the optional artifact but still allows a training-only submission. Labless API/auth/validation failures remain submission failures, not interpretation warnings. |

### Known Integration Failure Modes

These are the known limitations of the current design. We judge them infrequent enough to accept rather than add automatic retries, a second job, or more submission machinery; interpretation and submission failures may require manual recovery.

| Failure mode | Current design consequence | Training/evaluation results intact locally? | Training/evaluation results in Labless? |
|---|---|---|---|
| Invalid interpret config, missing hook, or Python error | Training/evaluation remain complete. `train.py` warns and continues to a training-only Labless submission if auto-submit is armed; partial interpretation files may remain. | ✅ | ✅ |
| GPU/CPU OOM or disk error confined to the interpreter | Same as an interpretation error above. If the event kills the whole SLURM job, see the job-level failure below. | ✅ | ✅ |
| Extraction is interrupted before `info.json` is written | The partial cache remains. A retry will fail when it reads the missing completion marker; this fail-fast behavior is intentional. Manually delete the incomplete cache before retrying. | ✅ | ✅ |
| Activation-cache cleanup fails after successful analysis | Results and dashboard remain complete, but the large cache remains and needs manual deletion. | ✅ | ✅ |
| W&B dashboard upload raises a normal error | Local interpretation results remain complete. The submitter warns and continues to a training-only Labless submission; W&B may contain a partial run. | ✅ | ✅ |
| Submitter process is killed during dashboard upload (for example, by OOM) | Local interpretation results remain complete, but the shell may stop before the Labless POST; the W&B run may be partial. Submit the training output manually. | ✅ | ❌ |
| Labless rejects the submission after W&B upload | The W&B artifact may exist without a Labless run record. Retrying may create another W&B run. | ✅ | ❌ |
| SLURM timeout, preemption, node loss, or job-wide OOM during interpretation | Training outputs are already complete, but `train.py` may never reach the single Labless submission. Interpretation files may be partial; submission requires manual recovery. | ✅ | ❌ |
| Concurrent jobs share an activation cache directory or training output directory | Outputs can collide or be deleted. Each concurrent run needs distinct directories. | ⚠️ | ⚠️ |
| Interpretation fails after replacing its results directory | Previous results are removed at invocation start; partial output from the failed attempt may remain. | ✅ | ✅ |

`✅` means the training/evaluation results are intact/present, `❌` means they are absent, and `⚠️` means partial or uncertain. Labless outcomes assume auto-submit is armed and its POST succeeds unless the row describes a submission failure.


## Hook contract

Any model analysed by the rig implements one method, `register_interpretability_hooks(self, storage_dict) -> list[str]`
(see `ViT` in `model.py` for the reference implementation). It is the ONE place where model-specific details and quirks live:
the rig never reads model code and never slices tokens by position. It assumes a transformer that passes token sequences
between layers. The contract below is enforced by `assert_hook_interface` in `interpret.py`; when you add a check there,
document the new property here.

The method registers forward hooks on the milestones (layers) the rig should analyse and returns their string keys in true
forward-pass order, e.g. `["block_0", ..., "block_11", "norm"]`. The rig empties `storage_dict` in place
(`storage_dict.clear()`) before every forward pass, so hooks must keep writing into the same dict object they were given.

Each hook writes `storage_dict[layer_name] = {group_name: tensor}` with two REQUIRED groups (`REQUIRED_GROUPS` in
`interpret.py`), the objects downstream methods consume:

- `"patch"` `(B, N, C)`: the patch tokens. `N` must be a perfect square and the tokens in row-major order on the square grid.
- `"cls"` `(B, 1, C)`: the global summary token.
- Each required tensor is detached and finite with shape `(B, T, C)`: `B` is the batch size, `T` the tokens, `C` the hidden
  dim. `C` is identical across the two groups of a layer, and neither may be empty.
- Groups are disjoint (a token is in exactly one group). Token counts are identical across all layers and batches.
- Any other group you write (e.g. `"register"`) is ignored: it is neither validated nor stored. Write it or not.
- Tensors are written as-is: no pooling or flattening, no dtype casts, no device moves (the rig handles those).
- Hooks only observe: they must not alter the forward pass or its outputs.

The rig attaches no meaning to milestones: pre- or post-norm, tuple unpacking, etc. are the implementer's choice.
Cross-model depth comparison uses the relative position in the returned list. Recommended (not enforced): make the last
milestone the representation your downstream heads consume. Disjoint groups, row-major patch order and forward-pass order
of the returned list cannot be checked by the rig: they are promises by the implementer.

## Your to-do list to analyse a new model

- [ ] Implement `register_interpretability_hooks(self, storage_dict) -> list[str]` on your model, following the contract above.
- [ ] If your model is not a standard Nanopath ViT checkpoint, update `load_model` in `interpret.py` so it loads and builds your model
  and returns `(model, train_cfg, step)`.
- [ ] Ensure your model's forward pass returns `{"cls": (B, C), "patches": (B, N, C)}` so attribution methods like RELAX can evaluate embeddings.

Nothing else in the rig needs to change.

## Extending the rig

- New dataset: a class in the source-dataset section of `data.py` that takes its root folder as an argument (`load_dataset` passes it from
  `probe.dataset_roots`; `probe.py`'s loaders are the precedent), returns `(PIL image, metadata dict)` and exposes
  `records`, a list of `(image, metadata)` per sample (the sampler stratifies over `metadata[stratification_dimension]`);
  add it to `load_dataset` and a block under `datasets:` in the config.
- New method: a class in its own file in `interpret/methods/`, `Method(activation_datasets)` (`Method(activation_datasets, device)` if it builds its own modules, like the probe's model; a method that trains models on folds subclasses `Probe` in `probe_base.py` and implements its hooks, like `linear_probe.py` and `layer_inversion.py` (the hooks, their arguments, and what is asserted about them and what is not are documented in `probe_base.py`; not to be confused with the model hooks above); `activation_datasets` is `{layer name: ActivationDataset}`; enable `patch_tokens` or `image` on a dataset if the method needs token-level data or images) with `run()`, which stores the outputs of the method in `self.results`
  (plain Python values, dicts, lists and CPU tensors, the results contract in `main()`), and `plot()`, which returns `{file name: matplotlib Figure}` built from `self.results`
  (`main()` saves the figures); add it to the dispatch in `main()` and an entry in `methods` in the config (a method with tunables gets its own config sub-block, passed to its constructor in the dispatch), and a subsection under
  Methods here (the questions it answers, the papers behind it, notes on the paper(s)).

## Notes

The model input uses probe.py's eval transform plus the normalization that probe.py applies at its model-call sites
(`evaluation_preprocess` in `interpret.py`). probe.py is locked, so that normalization is duplicated: keep the two in sync.