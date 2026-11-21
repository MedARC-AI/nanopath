<!-- Literature survey for the planned layer-inversion method of interpret/: a decoder per layer reconstructs the tile image from that layer's activations.
     Deliverable of the survey only (no code); the implementation is a later, separate step. Sources are cited by section, figure or table. -->

# Layer inversion: literature survey

Layer inversion trains one decoder per kept layer (`block_0`, `block_4`, `block_8`, `norm`) to turn that layer's activations of a tile back into the tile's image. How well it succeeds, and in what (colour, layout, texture, nuclei), shows what visual information is still recoverable at that layer. It is the linear probe's idea with the whole image as the target.

Scope, as agreed: decoders trained once per layer and then applied to all tiles. Per-image optimization (optimizing an image until its activations match, Mahendran & Vedaldi 2015) is not covered. Generative decoders (GANs, diffusion) get one brief section. The survey was stopped early for budget: 19 papers were read in full and about 100 screened, across six topics. Claims carry a section, figure or table pointer. "(ours)" marks our own reasoning, and "unverified" marks what was seen only in an abstract or summary.

## 1. Summary

**What inversion measures.**
- Reconstruction quality is the visual information that a given decoder family can extract from a layer: "usable information" (Xu et al. 2020, Def. 3). It is not the total information the layer holds.
- Layers are therefore only comparable with the same decoder and the same training budget.
- A decline with depth means the information became harder to extract. It does not prove the information is gone: i-RevNet classifies ImageNet while staying invertible (Jacobsen et al. 2018, Table 1).
- Recoverable does not mean used by the model, because the residual connections carry the input forward (Raghu et al. 2021, Sec. 6; Walmer et al. 2023, App. D.4).

**State of the literature.**
- **CNNs:** per-layer decoders are well studied, starting with Dosovitskiy & Brox 2016. So is the privacy work on split learning, which trains "inverse networks" at every split depth.
- **ViTs:** only a handful of studies. They cover supervised ViT-B (Rathjens et al. 2024/25; Leggio et al. 2026), CLIP (Lei et al. 2025), supervised ViTs with linear decoders (Krishna Kumar 2025), and a SigLIP 2 PSNR curve (Tong et al. 2026).
- **Gaps:** no paper decodes a DINO/iBOT ViT one layer at a time with a fixed, non-generative decoder. No pathology paper trains a non-generative pixel decoder on a frozen layer. Our rig would fill both gaps.

**What to expect (ours, from the cited evidence).**
- **Gradual decline:** ViTs lose pixel fidelity gradually with depth, not in the steps CNNs show at pooling layers (Leggio Fig. 2: SSIM 0.92 to 0.49 over blocks 1 to 11; Lei Table 2).
- **Colour and layout survive deep:** colour and coarse layout survive the longest, and fine texture and high frequencies fade first (Park et al. 2023, Fig. 9 and 12c; DecQ Table 6).
- **Stain:** stain and centre information is likely still recoverable at `block_8`, and possibly at `norm`. Pathology models keep the scanner signal until their last few blocks (Filiot et al. 2026, Fig. 4), and the centre is 88 to 98% linearly decodable from final embeddings (PathoROB).
- **A rough reference:** a final-layer DINOv2 ViT-S, our size, decodes to about 18 dB PSNR on ImageNet at 256 px (SVG Table 3), with a large decoder and far more data than we have.

**Recommended first version, in five lines.**
1. One decoder per kept layer, trained on the saved, frozen patch tokens, with the same architecture, budget and seeds for every layer.
2. Two decoder sizes: a per-token linear head (384 to 14 × 14 × 3 pixels; the headline number, the analogue of our linear probe) and a small MAE-style decoder (2 to 4 transformer blocks, then a per-patch linear head).
3. Pixel MSE, with no perceptual or adversarial terms. Tokens are standardized per channel and per layer with statistics from the training split.
4. Metrics on held-out slides:
   - the gain over the mean image (R² and PSNR), SSIM and LPIPS;
   - colour error separated from structure error;
   - our derived pixel labels recomputed on the reconstructions;
   - image grids and per-token error maps.
5. Controls: the mean image, a pixel ceiling (PCA-384 per patch), a randomly initialized ViT-S/14 (one extra extraction), shuffled tokens, and 3 seeds.

## 2. Glossary

| Term | Meaning |
|---|---|
| Patch token, CLS, registers | The 384-number vector per 14 × 14-pixel patch (256 per tile), the one global token, and the 4 extra learnable tokens with no image location. Our extraction saves patch tokens and CLS, not registers (`model.py`, `_store_tokens`). |
| Residual stream | Every ViT block adds its output to its input, so information from the input stays unless later blocks overwrite it. |
| Per-token (per-patch) linear head | One linear map from a token to the 588 pixel values of its own patch; the patches are then tiled back into the image (MAE Sec. 3). |
| MSE, PSNR | Mean squared pixel error; PSNR = 10·log10(1 / MSE) in dB for pixels in [0, 1]. PSNR carries no new information beyond MSE (Wang & Bovik 2009). Higher PSNR is better, and +3 dB is about half the error. |
| R² over the mean image | 1 − MSE(decoder) / MSE(mean training image): the share of pixel variance explained. With an MSE loss this is the usable information (Xu Prop. 1.5; derivation in the theory notes). |
| SSIM, MS-SSIM | Structural similarity from local brightness, contrast and correlation, where 1 means identical (Wang et al. 2004). MS-SSIM is the multi-scale version. |
| LPIPS | Distance between deep features (AlexNet or VGG) with weights fitted to human judgements; lower is more similar. Calibrated on 64-pixel natural-image patches (Zhang et al. 2018, Sec. 2.2). |
| rFID | Distance between the feature distributions of all reconstructions and all originals: realism of the set, not per-image fidelity. |
| Expected pre-image | The average of all images that produce the same features. An MSE-trained decoder outputs this, so blur means "the features do not pin this down" (Dosovitskiy & Brox 2016, Eq. 2). |
| Usable (V-) information | Information that a chosen family of decoders can extract; unlike Shannon information it can rise or fall across layers (Xu et al. 2020, Sec. 3.2). |
| Perceptual, adversarial loss | A loss on the features of a pretrained network, and a loss from a discriminator that judges realism. Both make outputs sharper but add another network's priors. |
| Normalized-pixel loss | The MAE target in which each patch is standardized by its own mean and standard deviation, which removes the patch's colour and contrast. |
| Frequency bands | Coarse (low-frequency) versus fine (high-frequency) image content; errors can be reported per band. |
| Mean-image baseline | The best prediction that ignores the input: the average training tile. This is the floor, not zeros. |
| Random-init control | The same architecture with untrained weights, decoded with the same decoder. |
| H&E, stain deconvolution | Hematoxylin colours nuclei blue-purple and eosin colours the rest pink. Stain deconvolution turns RGB into per-pixel H and E amounts; we have this as `rgb_to_hed` in `dataloader.py`. |

## 3. The families of inversion

| Family | How it works | Answers | Cost | Key papers |
|---|---|---|---|---|
| **Trained per-layer decoder** (our focus) | Train a decoder from layer activations to the image on many (activation, image) pairs, then apply it to held-out tiles in one forward pass. | What is **recoverable** by that decoder family | Training once per layer; inference in milliseconds (Dosovitskiy & Brox 2016, Sec. 1.1: 5 ms vs 6 s per image for optimization) | Dosovitskiy & Brox 2016; He, Zhang & Lee 2019; Rathjens et al. 2024/25; Lei et al. 2025; Leggio et al. 2026 |
| Chained (modular) inverse | One small inverse per stage (stage j to stage j−1), chained at test time. | A different question: what survives along the forward path. Reconstructions are biased toward the path (Rathjens Sec. 3.2, 4.2). | Higher | Rathjens et al. 2024/25 |
| Invertible networks | Architectures with an exact inverse. | Theory: whether a network must discard information at all. It need not (Jacobsen et al. 2018, Table 1). | Not applicable to our trained model | Jacobsen et al. 2018; Behrmann et al. 2019 |
| Generative decoders | GAN or diffusion decoders conditioned on the activations. | What looks **plausible**; detail is invented (Sec. 4). | High | RCDM; RAE; HistoXGAN |
| Per-image optimization | Optimize the image until its activations match. | Excluded by decision. | High per image | Mahendran & Vedaldi 2015 (not covered) |
| Companions without decoding | Token Fourier spectra, sparse autoencoders, linear probes, retrieval. | Cheap side views of the same question | Low | Park et al. 2023; Le et al. 2024; our linear probe |

## 4. Generative decoders: brief overview

Generative decoders are a major topic: frozen DINOv2 encoders are used as image tokenizers, and pathology diffusion models are conditioned on embeddings. We leave them out because they invent detail by design. The rest is what their papers found about what representations keep.

- **Realism is not fidelity.** RAE decodes frozen DINOv2-B final tokens with rFID 0.49 (Table 1a), yet re-measured PSNR is only about 18 to 19.6 dB (Prism Table 1; DecQ Table 1; DINO-Tok Table 6). A set can look realistic while each image is unfaithful.
- **Shallow layers hold the detail.** Adding DINOv2-B layers 0 to 3 to the final layer lifts PSNR from 19.61 to 23.47 dB; adding layers 8 to 11 reaches only 20.72 dB (DecQ Table 6, verified).
- **Decoder size barely matters for the final layer.** Going from 125M to 268M parameters moves PSNR from 19.44 to 19.56 dB (DINO-Tok Fig. 8, verified).
- **The main network keeps more than the projection head.** In RCDM the backbone keeps grayscale status and the background colour palette despite colour jitter, while the projection head discards them (Bordes et al. 2022, Fig. 3, 25). All our saved layers are in the main network.
- **Pathology: one vector fixes content, not layout.** HistoXGAN reconstructions from one feature vector keep grade and stain batch effects, but not spatial positions (Howard et al. 2024, Results, Tables 2 to 4). Graikos et al. 2024 get layout only from a grid of per-patch embeddings (Sec. 1, Fig. 9).

## 5. The design space

### 5.1 Decoder

| Option | Evidence | Pros and cons | Fit for us |
|---|---|---|---|
| Per-token linear head (384 to 588) | The MAE read-out (Sec. 3). Used as a pixel probe in Darcet et al. 2024 (Fig. 5b) and Baxevanakis et al. 2026 (Sec. 3.3). Heavier heads barely beat it in masked modelling (SimMIM loss 0.3743 vs 0.3722, Sec. 4.1.3). | The most selective option (Hewitt & Liang 2019, Sec. 3.6). Cheap (about 0.23M parameters, ours). Error maps per token come for free. Cannot use neighbouring tokens. | **Headline decoder.** The analogue of our linear probe. |
| Per-token MLP or non-overlapping transposed conv | Rathjens' inverse backbone (code: kernel 2, stride 2, no mixing across patches). | Non-linear but still local. | Optional middle rung. |
| Small token-mixing decoder: 2 to 4 transformer blocks plus a per-patch head (MAE-style) | Lei et al. 2025 (4 blocks of width 512, about 18M parameters, Sec. 4.4); MAE (8 × 512); Allakhverdov et al. 2025 (4 blocks plus upsampling). | Tokens can share information before the read-out. The gap to the per-token head shows how much of a patch's content has moved into other tokens (ours). | **Second rung.** Keep it small, because we only have 10k tiles. |
| Convolutional upsampler on the 16 × 16 grid | Leggio et al. 2026 (Table 1); DINO-WM (conv with residual blocks, DINOv2 ViT-S); He, Zhang & Lee 2019. | Standard in the privacy literature. Transposed convolutions cause checkerboard artefacts; bilinear upsampling causes fewer (Wojna et al. 2017, Sec. 5). | Alternative to the transformer rung. |
| Growing the decoder with depth | He 2019 code; Yang 2022, Sec. 5.1. | Confounds depth with capacity. | **Avoid.** |

Capacity rule from the literature: a weak decoder understates what is recoverable (ResSFL Sec. 3.1). Wider decoders change absolute scores but rarely the ranking (Spiridonov et al. 2026, App. A.2). So report two sizes and trust conclusions that hold at both.

### 5.2 Input

| Option | Evidence | Fit for us |
|---|---|---|
| Patch tokens of one layer | Patch tokens stay tied to their own patch even at the last layer (Raghu Fig. 9; Rathjens Sec. 4.5). Layout needs patch tokens (HistoXGAN; Graikos Fig. 9). | **Main input.** |
| Standardization per layer | Block outputs and the final norm have very different scales. In our own probe briefing the input-vector norm is 3.55 at `block_0` versus 36.28 at `norm`. Methods seen: per-channel standardization (Drexlin et al. 2026, App. A.1), LayerNorm without affine parameters (RAE App. C.1), per-token L2 normalization (Allakhverdov Sec. 3.2, which discards the norm). | Per-channel mean and standard deviation from the training split, per layer (ours). |
| CLS or patch mean only | Pooled vectors keep colour, pose and scale but not layout (RCDM Fig. 2 and 3; HistoXGAN). | Optional "global content" decoder. Judge it on colour and coarse targets. |
| Plus register tokens | Registers carry global appearance. Adding them raises PSNR from 19.49 to 19.73 dB, 43% of the gain lies in mean colour, and registers beat CLS as an extra input (Drexlin Tables 2 and 7, Sec. 4.2; verified). | Not saved today. Saving them costs about 61 MB per layer and needs a re-extraction. An open decision. |
| Several layers fused | Raises fidelity a lot (DecQ; DINO-Tok). | Only as an upper-bound control; it mixes layers. |

### 5.3 Loss

| Option | Evidence | Fit for us |
|---|---|---|
| Pixel MSE (or L1) | Gives the expected pre-image (Dosovitskiy & Brox Eq. 2) and matches usable information with a Gaussian likelihood (Xu Prop. 1). Used by almost all probe-style inversions (Rathjens Eq. 1; DINO-WM; Leggio; Lei). L1 and L2 behave alike (SimMIM Table 5). | **Main loss.** Blur where the layer leaves uncertainty is the signal, not a bug. |
| Normalized-pixel target | Helps MAE's encoder (Table 1d) but removes each patch's colour and contrast. MAE recommends turning it off for visualization (`PRETRAIN.md`). Pathology MAEs use it (SD-MAE; MMAE). | Not as the main loss, because it would hide stain colour. Optional "structure without colour" variant (ours). |
| Perceptual (LPIPS or VGG) | Sharper output at lower PSNR and SSIM (Johnson et al. 2016, Fig. 8; Rojas-Gomez Table 1). | Not in v1. If used, never evaluate with the network that is inside the loss (DeePSiM Fig. 8). |
| Adversarial or diffusion | Invents content (DeePSiM Fig. 7). Sharp output costs up to about 3 dB of PSNR (Blau & Michaeli, Thm 3). | Excluded. |

### 5.4 Metrics

| Metric | Why | Caveat |
|---|---|---|
| R² and PSNR gain over the mean image, on held-out slides | Usable information under MSE, with the right floor (Xu Def. 2; derivation in the theory notes). | Dominated by high-variance content such as colour and low frequencies (Lampinen et al. 2024, Sec. 3). |
| MSE split into a mean-colour term and a centred (structure) term | Separates stain colour from morphology (Drexlin Eq. 5). | (ours) Maps directly onto our question. |
| SSIM or MS-SSIM | Structure. Survives longer with depth than perceptual metrics (Lei Table 2). | Validated on brightness only; misbehaves in dark regions (Wang et al. 2004, Sec. IV-B; Nilsson & Akenine-Möller 2020, Sec. 4). |
| LPIPS (AlexNet, v0.1) | Perceptual similarity. | Calibrated on natural images, not H&E; sensitive to shifts (Zhang et al. 2018; ST-LPIPS, unverified). |
| Error per frequency band | Late layers are expected to lose high frequencies (Park Fig. 9, 12c). | Needs a pixel ceiling for reference. |
| Colour and stain error | Colour shift is the reported failure of DINO decoders (DecQ Sec. 1; SVG Fig. 5). Use H and E channels (MMAE Eq. 1) and LAB statistics (Le et al. 2024, Sec. 2.3). | (ours) Our derived labels (`brightness`, `tissue_brightness`, `tissue_fraction`, `hematoxylin`, `eosin`) recomputed on reconstructions do this with existing code. |
| Re-encoding or classifying the reconstruction | Whether the reconstruction keeps what the model uses (Allakhverdov Sec. 5.1; HistoXGAN Tables 2 to 4; Lei Sec. 4.3). | Use our own model and our linear probes (tumour, centre). |
| Image grids and per-token error maps | Numbers can mislead: an MSE above the mean image can still show structure (Rathjens Sec. 4.1). | The visual output you want. |

### 5.5 Controls and baselines

| Control | Why | Evidence |
|---|---|---|
| Mean image | The floor: what is predictable with no input. | Xu Def. 2; Rathjens Fig. 4 |
| Pixel ceiling (raw patches, or the best 384-dimensional linear code per patch, PCA-384) | The patch embedding maps 588 pixel values to 384 dimensions (verified in `model.py`: `Conv2d(3, 384, kernel 14, stride 14)`), so even `block_0` cannot be lossless. The ceiling also exposes decoder-caused blur from spectral bias (Rahaman et al. 2019, Sec. 3). | theory notes; ours |
| Randomly initialized ViT-S/14, same decoder | Random networks are often **easier** to invert than trained ones. A decoder on random-weight VGG features reconstructs better (He et al. 2017, Fig. 3); pretrained minus random recoverability is negative in 10 of 13 CNNs (Spiridonov et al. 2026, Sec. 4.3, verified). The informative number is trained minus random. Our LayerScale starts at 1 (`model.py`), so the random network is not just the identity. | He et al. 2016, 2017; Spiridonov 2026; Adebayo et al. 2018 |
| Tokens shuffled across positions | How much of the score comes from the known layout. A conv decoder on shuffled ViT tokens gives SSIM 0.18 to 0.27 at every depth (Leggio Fig. 2, verified). | Leggio 2026 |
| Same decoder, budget and seeds for every layer (3 seeds) | Budget alone moves PSNR by about 2 dB (Rojas-Gomez Table 1 vs Table 4). | Rathjens Sec. 3.3 |
| Optional: random target per slide (memorization) | Whether a decoder can learn slide identity, the shortcut to the "right" colour. | Hewitt & Liang 2019, Sec. 2 (adapted, ours) |
| Optional: conditional decoding [colour thumbnail; tokens] vs [thumbnail; 0] | What a layer adds beyond colour. | Hewitt et al. 2021, Eq. 2 |
| Optional: ImageNet DINOv2 ViT-S/14 | Whether pathology training changes what is kept. | Le et al. 2024, Sec. 3.3 (analogous control) |

### 5.6 Data and splits

- **Split by `slide_id`**, as the probe does. Slides repeat stain and scanner signatures, so a mixed split rewards memorization (Hewitt & Liang Sec. 2.3; ours). Report the train versus held-out gap.
- **Data size:** 10,000 tiles is small next to ImageNet-scale decoders. One privacy study saw gains flatten beyond about 10K training images (Dong et al. 2022, App. D). A per-token decoder sees 2.56M patch examples, a whole-image decoder only 10k images, so the overfitting risk differs by decoder (Xu Thm 1; ours).
- **Optional centre hold-out:** train on RUMC and UMCU, test on CWZ, RST and LPON. This tests whether stain reconstruction generalizes to unseen stain shades. RUMC and UMCU hold about 91% of PathoROB Camelyon (Kömen et al. 2025, Table 3).

## 6. What is known across depth

| Paper | Model and input | Decoder | Result across depth |
|---|---|---|---|
| Dosovitskiy & Brox 2016 | AlexNet conv1 to fc8 | Up-convolutional, MSE | Fine detail fades with depth, with a clear drop from CONV5 to FC6. FC8 still keeps colour and rough object position (Sec. 4.1, Fig. 5, 7). |
| Zhang, Lee & Lee 2016 | AlexNet, VGG | Decoders given the max-pooling locations | Almost perfect up to fc8 when those locations are given (Sec. 4.2). |
| He et al. 2017 | VGG16 and AlexNet, trained vs random | Deconvolutional, L2 | Random weights are more invertible, and the gap grows with depth for VGG (Sec. 3.2, Fig. 3). |
| He, Zhang & Lee 2019 | 6-conv CNN, CIFAR-10 | Transposed conv, MSE | 49.88, 19.81 and 15.42 dB at conv11, ReLU22 and ReLU32 (Table 4, verified). |
| Raghu et al. 2021 | Supervised ViT | (similarity analysis) | Final-block tokens are most similar to their own input patch; skip connections dominate (Sec. 6 to 7, Fig. 7, 9). |
| Dong et al. 2022; Yang et al. 2022 | RepVGG; ResNet-50 | Learned block-wise inverse; transposed conv | CNN curves fall in steps at stride-2 and pooling layers and collapse at fc (Dong Sec. 4.1; Yang Table 2: SSIM 0.785 to 0.452). Self-supervised MoCoV2 leaks more than supervised (Dong Sec. 4.1). |
| Walmer et al. 2023 | DINO, MAE and four other ViT-B models | (analysis) | DINO's residual stream stays similar to each block's input at all depths (App. D.4, Fig. 23). |
| Park et al. 2023 | MoCo v3 vs SimMIM ViT-B | (token Fourier analysis) | Contrastive models cut high frequencies in the last layers. A contrastive plus masked-modelling hybrid amplifies them early and cuts them late (Fig. 9, 12c). |
| Darcet et al. 2024 | DINOv2-g, final tokens | Linear pixel probe per token | High-norm outlier tokens hold less pixel information about their own patch (L2 error 25.23 vs 18.38). Registers leave normal patches unchanged (Fig. 5b; App. D.2). |
| Rathjens et al. 2024/25 | Supervised ViT-B/16, embedding vs final | Per-patch inverse plus a 12-block inverse | Reconstructions stay faithful, with colours preserved under recolouring at every stage. MSE is about 2e-4 at the embedding versus 0.8e-2 at the final layer and 7e-2 for the mean image, read from a bar chart (Sec. 4.1, 4.3, Fig. 4). |
| Lei et al. 2025 (inverse network only) | CLIP ViT-B/16, layers 0 to 12 | MAE-style, MSE | MS-SSIM 0.99, 0.98, 0.91, 0.82, 0.74; LPIPS 0.011 to 0.362 (Table 2, verified). |
| Krishna Kumar 2025 | Supervised ViT-S/16, ViT-B/16 | Linear per patch, to the patch embeddings (not pixels) | ViT-S falls from 0.658 to 0.035, ViT-B from 0.785 (verified) to 0.054; mixing across patches adds at most about 0.03 (Tables 8 to 10). |
| Leggio et al. 2026 | Supervised ViT-B/16, blocks 1 to 11 | Conv upsampler, MSE | SSIM 0.92, 0.89, 0.84, 0.79, 0.74, 0.69, 0.65, 0.61, 0.56, 0.53, 0.49: smooth, with no cliff. MAE-B stays more invertible at depth (Fig. 2, Sec. 5.1; verified). |
| Tong et al. 2026 | SigLIP 2, layers 0 to 26 | RAE-style (decoder not fully clear) | PSNR falls from 29.6 to 20.9 dB while ImageNet accuracy rises from 12.8 to 86.3% (App. D.1, verified). |
| Spiridonov et al. 2026 | 13 CNNs, each layer's own input | Linear and MLP | Recoverability R² falls to 0.1 to 0.4 at the last layer in 7 of 9 ImageNet CNNs. Pretrained minus random is negative in 10 of 13 models (Sec. 4.1, 4.3, verified). |
| SVG 2025 | Final DINOv2 ViT-S | Conv decoder (VA-VAE style) | 18.10 dB, SSIM 0.40, LPIPS 0.30; MAE-B reaches 25.04 dB (Table 3, verified). |
| Le et al. 2024 (pathology) | PLUTO ViT-S, CLS of layers 1 to 12 | Sparse autoencoders | Layers 1 to 6 track colour, layers 7 to 12 track cell types. The scanner effect falls to near zero at the last layer (Sec. 4, 5.2, Fig. 5, 7). |
| Filiot et al. 2026 (pathology) | H-Optimus-0, per block | Cross-scanner retrieval | Robust matching across scanners "only develops in the last few blocks" (Sec. 5, Fig. 4). |
| Vadori et al. 2025, "Mind the Gap" (pathology) | UNI2 | Nucleus segmentation | Shallow blocks beat deep ones: PQ 62.18 vs 53.46 (Table I). |

## 7. Digests of the core papers (chronological)

**Dosovitskiy & Brox 2016, CVPR.** *Inverting Visual Representations with Convolutional Networks.*

| | |
|---|---|
| What | HOG, SIFT and LBP features, and AlexNet conv1 to fc8 (Sec. 2.1). |
| Decoder and loss | Up-convolutional decoders with leaky ReLU, one template for conv and one for fc layers (Table 1); pixel squared error (Eq. 1). |
| Data and metrics | ImageNet, Adam, 15 to 60 epochs. Normalized error: L2 divided by the mean distance between test images (Sec. 2.2). |
| Findings | MSE gives the expected pre-image, and blur reflects the features' invariance (Sec. 1, Eq. 2). Colour and rough position survive up to fc8 (Sec. 4.1, 4.3). In fc6 the information sits in which units are active (Sec. 4.4). The decoder's prior supplies colour from grayscale features (Sec. 3) and realistic images from random vectors (Fig. 11). A jointly trained autoencoder is the upper bound (Sec. 4.2). |
| Caveats | The kernel size differs between the text (5×5) and the supplement (3×3). Decoder size differs between conv and fc layers. Numbers only in a plot. |
| For us | The normalized error (1.0 is about as good as a random test image) is a scale-free reference. Perturbation tests (binarize, drop) need no retraining (ours). |

**Jacobsen, Smeulders & Oyallon 2018, ICLR.** *i-RevNet.*

| | |
|---|---|
| What | Whether a network must discard input information to classify. |
| Model | Invertible CNNs with an analytic inverse (Sec. 3.1). |
| Findings | Top-1 error 24.7% injective and 26.7% bijective, versus 24.7% for ResNet (Table 1). The Jacobian is ill-conditioned, so a learned inverse would be hard (Sec. 4.1). Linear separability rises with depth without any loss of information (Sec. 5.1). |
| For us | A falling curve means "less usable for this decoder family", not "information gone". Ill-conditioning predicts that fine detail blurs first (ours). |

**Zhang et al. 2018, CVPR.** *LPIPS.*

| | |
|---|---|
| What | A learned perceptual distance. |
| Method | Channel-normalized AlexNet, VGG or SqueezeNet features with per-channel weights (Eq. 1), calibrated on 484k human judgements of 64-px patches (Sec. 2). |
| Findings | Agreement with humans: LPIPS 69 to 70%, against 63% for L2 and SSIM and 73.9% between humans (App. Table 5). |
| For us | LPIPS-alex as a secondary metric, with the natural-image caveat. |

**Hewitt & Liang 2019, EMNLP.** *Designing and Interpreting Probes with Control Tasks.*

| | |
|---|---|
| What | Whether probe accuracy reflects the representation or the probe. |
| Method | Control tasks with random outputs per word type; selectivity is task accuracy minus control accuracy (Sec. 2). |
| Findings | MLP probes gain 0.1 accuracy but are almost pure memorizers: selectivity 4.2 versus 26.0 for linear probes (Table 1). Dropout is not a capacity control (Sec. 3.5). Layer comparisons are confounded by ease of access (Sec. 4, Table 2). |
| For us | Prefer the simplest decoder; compare layers with one family; expect `block_0` to win simply because it is closest to the pixels. |

**He, Zhang & Lee 2019, ACSAC.** *Model Inversion Attacks against Collaborative Inference.*

| | |
|---|---|
| What | A CNN split at three depths, CIFAR-10. |
| Decoder and loss | Transposed-conv inverse network, deeper for deeper splits (code); MSE (Eq. 4). |
| Findings | PSNR 49.88, 19.81 and 15.42 dB from shallow to deep (Table 4). A decoder trained on noise images still inverts the first layer (48.59 dB) but fails on deep layers (12.37 dB), so deep layers need the data prior (Sec. 5.2). |
| For us | Keep the decoder fixed across layers. "Decoder trained without tiles" is a test of how linear a layer is (ours). |

**Xu et al. 2020, ICLR.** *A Theory of Usable Information under Computational Constraints.*

| | |
|---|---|
| What | Information measured relative to a family of predictors V. |
| Definitions | V-entropy is the best achievable log loss within V. V-information is the gain over ignoring the input (Def. 2 to 3). |
| Findings | With linear-Gaussian families it equals R² times the variance (Prop. 1.5). It can increase through computation, unlike Shannon information (Sec. 3.2). Estimation bounds grow with the family's complexity (Thm 1). |
| For us | PSNR gain over the mean image on held-out slides is a V-information estimate, about 0.115 nats per dB (our derivation). Curves across layers are only comparable for one family. |

**Xiao et al. 2021, ICLR.** *What Should Not Be Contrastive in Contrastive Learning.*

| | |
|---|---|
| What | MoCo v2 ResNet-50 with and without each augmentation. |
| Findings | An augmentation measurably removes its factor from the backbone. For rotation, probe accuracy falls from 61.1 to 43.3% (Table 1). A colour-sensitive head helps fine-grained classes (Table 2). |
| For us | Colour and stain, which nanopath jitters, are the factors to expect reduced with depth. Report colour error separately (ours). |

**Rojas-Gomez et al. 2022, ACCV.** *Inverting Adversarially Robust Networks for Image Synthesis.*

| | |
|---|---|
| What | Standard versus adversarially robust encoders, one layer each. |
| Decoder | A 4.7M-parameter mirror decoder, the same for every encoder (Sec. 5.1). |
| Findings | Robust conv5: 19.90 versus 17.56 dB with pixel loss (Table 1). The losses rank the conditions differently. Data and budget alone move PSNR by about 2 dB (Table 1 vs Table 4). |
| For us | Fix decoder, loss and budget for every layer, and report PSNR, SSIM and LPIPS together. |

**He et al. 2022, CVPR.** *Masked Autoencoders* (the decoder part).

| | |
|---|---|
| Decoder | 8 transformer blocks of width 512 and a per-patch linear head to the pixels (Sec. 3; code). |
| Loss | MSE on masked patches; the normalized-pixel target helps representation learning (Table 1d) but is not for visualization (`PRETRAIN.md`). |
| Findings | Linear and non-linear read-outs rank models differently (Sec. 4.3). |
| For us | The per-patch head is the canonical ViT read-out (384 to 588 for our patches). A few blocks before it make the middle rung. |

**Walmer et al. 2023, CVPR.** *Teaching Matters.*

| | |
|---|---|
| What | ViT-B trained six ways, analysed at every block. |
| Findings | DINO's last layers are all-global, while MAE keeps local heads late (Fig. 4). The best layer depends on objective and task (Table 1). DINO's residual stream stays similar to each block's input (App. D.4, Fig. 23). |
| For us | Report the whole curve. A random-init control is needed because the residual stream carries the input forward (ours). |

**Park et al. 2023, ICLR.** *What Do Self-Supervised Vision Transformers Learn?*

| | |
|---|---|
| What | Contrastive (MoCo v3) versus masked modelling (SimMIM), per block. |
| Method | A Fourier analysis of token maps; the code is public. |
| Findings | Contrastive models reduce high frequencies in their last layers (Fig. 9). The hybrid amplifies early and reduces late (Fig. 12c). Shape versus texture bias (Fig. 8). |
| For us | Expect late-layer errors to concentrate in high frequencies. Caveat: iBOT predicts features, not pixels, and this is untested. The token Fourier metric runs on our saved tokens without any decoder. |

**Darcet et al. 2024, ICLR.** *Vision Transformers Need Registers.*

| | |
|---|---|
| What | High-norm outlier tokens in large ViTs. |
| Method | Per-token linear pixel and position probes. |
| Findings | Outliers carry global information and less about their own patch (Fig. 5b, Table 1). They appear mid-depth, only in large DINOv2 models (Fig. 4). Registers absorb them (App. D.1). |
| For us | Our ViT-S with registers likely has no outliers; DINOv2-S lacks them (Baxevanakis et al. 2026). Check norm histograms per layer. |

**Le et al. 2024.** *Sparse autoencoders on PLUTO* (a pathology ViT-S).

| | |
|---|---|
| What | The CLS token of all 12 layers. |
| Findings | Layers 1 to 6 track colour, intensity and saturation; layers 7 to 12 track cell types, with plasma cells appearing at layer 10 (Sec. 4, Fig. 5). The scanner effect is moderate early and near zero at the last layer (Sec. 5.2). |
| Caveats | CLS only. PLUTO also has MAE and Fourier pixel losses. "Colour less dominant" is not "colour unrecoverable". |
| For us | Inversion can settle that last distinction. Reuse their colour statistics as metrics. |

**Rathjens et al. 2024/25** (arXiv 2412.06534 v4). *Understanding Transformer-based Vision Models through Inversion.*

| | |
|---|---|
| What | DETR and a supervised ViT-B/16, per stage. |
| Decoder | A per-patch inverse backbone (no mixing across patches) plus a 12-block inverse encoder (code), with plain MSE. |
| Findings | ViT reconstructions stay faithful across stages and keep recolourings (Sec. 4.1, 4.3, Fig. 7). Direct ("classic") inverses pull colours toward the dataset mean (Sec. 4.2). Noise in 20% of tokens stays in its patches (Sec. 4.5). |
| Caveats | Supervised only, two ViT stages, mostly qualitative. The TMLR version was not checked. |
| For us | Show the mean-image baseline. The recolouring test transfers to an HED stain-shift test (ours). |

**Kömen et al. 2025** (PathoROB; Nature Communications 2026 per a repository record, unverified).

| | |
|---|---|
| What | 20 pathology models, final layer only. |
| Findings | Our Camelyon tiles are 256 px at 20x (0.5 µm/px; App. A.1). The centre is 88 to 98% linearly decodable. Phikon, an iBOT ViT, has Camelyon robustness 0.011 (Table 9). Reinhard normalization raises robustness by 16.2% (Sec. 2.4). |
| For us | Expect stain colour to remain reconstructable deep. Centre hold-out; optional Reinhard-normalized re-extraction as a colour control. |

**Allakhverdov et al. 2025** (WACV 2026 version not read). *Image Reconstruction as a Tool for Feature Analysis.*

| | |
|---|---|
| What | Final tokens of SigLIP, SigLIP2, CLIP, DINOv2 and others. |
| Decoder | 4 transformer blocks plus upsampling, L2 loss, per-token L2 normalization (Sec. 3). |
| Findings | Richer objectives retain more (Table 1). A red-blue swap is an orthogonal map in feature space (Sec. 5.2). |
| For us | Re-encoding similarity as a metric. Per-layer analysis is their stated future work (Sec. 7). |

**Lei et al. 2025** (DRAG; ICML 2025 per citation, unverified). *Data Reconstruction Attack using Guided Diffusion*: only the inverse-network part is in scope.

| | |
|---|---|
| What | CLIP ViT-B/16, layers 0 to 12. |
| Decoder | MAE-style, 4 blocks of width 512, pixel MSE, 25% token masking (Sec. 4.4). |
| Findings | Structure survives while perceptual and semantic similarity fall (Table 2). Token positions stay partly decodable (Sec. 5.5). |
| Caveats | Only 30 test images. |
| For us | A ready recipe for 14-pixel patches. Report a structural, a perceptual and a re-encoding metric. |

**Filiot et al. 2026.** *Robustifying pathology foundation models via fine-tuning.*

| | |
|---|---|
| What | Per-block cross-scanner retrieval in H-Optimus-0. |
| Findings | Scanner invariance develops only in the last few blocks; fine-tuning moves it about eight blocks earlier (Sec. 5, Fig. 4). The scanner offset looks roughly linear (Fig. 3, qualitative). |
| Caveats | Recipe not disclosed; one model. |
| For us | Expect centre colour to be recoverable through `block_8`. |

**Leggio et al. 2026** (arXiv preprint). *Position Matters: Feature Inversion Attacks in ViT Split Inference.*

| | |
|---|---|
| What | Supervised ViT-B/16 and MAE-B/16, split after blocks 1 to 11. |
| Decoder | Conv upsampler from the 14 × 14 token grid, MSE (Table 1). |
| Findings | A smooth SSIM decline from 0.92 to 0.49. Shuffled tokens give about 0.25 at every depth (Fig. 2, verified). MAE-B stays invertible deeper (Sec. 5.1). |
| For us | Expect a gradual curve. Shuffling is a cheap control. Our DINO + iBOT model may fall between the supervised and MAE models (ours). |

## 8. Pitfalls and guards

| Pitfall | Guard | Source |
|---|---|---|
| "Worse reconstruction at depth means the information is gone." | Say "less usable by this decoder family". Use two decoder sizes. | Jacobsen Table 1; Xu Sec. 3.2 |
| "Good reconstruction means the layer learned something." | Random-init control; report trained minus random. | He 2016, 2017; Spiridonov Sec. 4.3; Adebayo Sec. 3 |
| "Decodable means used by the model." | Phrase results as availability. The residual stream carries the input forward. | Elazar et al. 2021, Table 1; Raghu Sec. 6; Walmer App. D.4 |
| The decoder's prior fills in detail (colour, texture). | Mean-image floor; held-out slides; MSE-only decoders; image grids. | Dosovitskiy & Brox Sec. 3, Fig. 11 |
| Blur caused by the decoder (spectral bias), not the layer. | Pixel or PCA-384 ceiling with the same decoder; check convergence. | Rahaman Sec. 3 |
| Depth confounded with decoder capacity or budget. | One decoder, budget and set of seeds for every layer. | He 2019 code; Rojas-Gomez Tables 1, 4 |
| `block_0` wins only because it is closest to the pixels. | Read the curve relative to the ceiling. Optional conditional decoding. | Hewitt & Liang Sec. 4; Hewitt 2021 |
| Metrics disagree, and some fail on H&E. | Report structural, perceptual, colour and re-encoding metrics together. | Lei Table 2; Wang & Bovik; Nilsson Sec. 4 |
| Colour dominates the MSE. | Split colour and structure error; report error per band. | Lampinen Sec. 3; Drexlin Eq. 5 |
| The normalized-pixel target hides stain colour. | Raw-pixel loss. | MAE Table 1d; SD-MAE |
| Memorization on 10k tiles; stain confounded with slide and centre. | Slide split; train/held-out gap; optional centre hold-out. | Xu Thm 1; PathoROB Table 3 |
| Shannon-information language ("the layer contains X bits"). | Speak only of usable information under the stated decoder. | Goldfeld et al. 2019 (unverified) |

## 9. Pathology-specific notes

- **Our approach would be new in pathology.** 32 searches found no pathology paper with a non-generative per-layer pixel decoder on a frozen model. The only image decoders are generative (HistoXGAN, Graikos et al. 2024, ZoomLDM, PixCell).
- **Scale.** PathoROB tiles are 256 px at 0.5 µm/px (App. A.1), resized to 224 px (`model.py`, `probe_transforms`), giving about 0.57 µm/px. A 14-px patch is about 8 µm, roughly one lymphocyte nucleus (size from educational sources only, unverified). So nucleus outlines and chromatin texture are detail inside a single patch that the token must store.
- **Colour and stain across depth.** Colour dominates the early CLS directions in a pathology ViT-S (Le et al. 2024). Scanner invariance arrives only in the last blocks (Filiot et al. 2026). The centre is linearly decodable from final embeddings (PathoROB). Whether nanopath's HED and colour jitter remove stain colour by `norm` has not been tested anywhere; inversion at `norm` measures it directly (ours).
- **Nuclei across depth.** Shallow blocks are better for nucleus segmentation (Mind the Gap, Table I). Expect nuclei to blur first.
- **Stain metrics.** H and E channels via stain deconvolution (MMAE Eq. 1; our `rgb_to_hed`). LAB and intensity statistics (Le et al. Sec. 2.3). Our derived labels.
- **Masked-modelling practice in histology.** Shallow decoders (SD-MAE uses 4 × 192, Sec. 3.2). 8-pixel patches make reconstruction "a trivial interpolation" (MMAE Discussion), so histology is locally redundant: a decoder with spatial context can fill texture from neighbours (ours).
- **Pathology controls.** A centre hold-out. Reinhard stain normalization (PathoROB Sec. 2.4). Same tissue on several scanners (SCORPION, PLISM; Filiot Sec. 5), later and out of scope for the 10k tiles.

## 10. Recommended first version and the open decisions

**Recommendation.** Train decoders per kept layer on the saved patch tokens: a per-token linear head (headline) and a small MAE-style decoder, both with pixel MSE and the same budget and 3 seeds for every layer. Use a slide-grouped split. Report:
- the gain over the mean image (R², PSNR), SSIM, LPIPS and the colour/structure split;
- our derived labels on the reconstructions;
- image grids and per-token error maps;
- the mean-image, PCA-384, random-init and shuffled-token controls.

| Decision | Options | Recommendation | Why |
|---|---|---|---|
| Headline decoder | per-token linear / small transformer | **Per-token linear**, transformer as the second rung | The most selective option, closest to our linear probe, cheap. The gap between the two is itself a result. |
| Random-init control | yes / no | **Yes** (one extra extraction with an untrained ViT-S/14) | The most important control in the literature. |
| Register tokens | save them (about 61 MB per layer, re-extraction) / skip | **Skip for v1, add later** | Mostly global colour; patch tokens answer the spatial question. |
| CLS or patch-mean decoder | add / skip | **Skip for v1** | Pooled vectors give content without layout; a later extension. |
| Split | one slide-grouped split / k folds | **One slide-grouped split** (e.g. 75/25) | Decoders cost more than probes; folds can come later. |
| Centre hold-out | add / skip | **Later** | Only about 9% of tiles come from the three small centres. |
| Normalized-pixel variant | add / skip | **Skip for v1** | It removes colour, which is half of our question. |

## 11. Bibliography

Read in full (core):
- Dosovitskiy & Brox 2016, CVPR. https://arxiv.org/abs/1506.02753
- Jacobsen, Smeulders & Oyallon 2018, ICLR, i-RevNet. https://arxiv.org/abs/1802.07088
- Zhang et al. 2018, CVPR, LPIPS. https://arxiv.org/abs/1801.03924
- Hewitt & Liang 2019, EMNLP, control tasks. https://arxiv.org/abs/1909.03368
- He, Zhang & Lee 2019, ACSAC, model inversion in collaborative inference. https://github.com/zechenghe/Inverse_Collaborative_Inference
- Xu et al. 2020, ICLR, usable information. https://arxiv.org/abs/2002.10689
- Xiao et al. 2021, ICLR, what should not be contrastive. https://arxiv.org/abs/2008.05659
- Rojas-Gomez et al. 2022, ACCV, inverting robust networks. https://arxiv.org/abs/2106.06927
- He et al. 2022, CVPR, MAE. https://arxiv.org/abs/2111.06377
- Walmer et al. 2023, CVPR, Teaching Matters. https://arxiv.org/abs/2212.03862
- Park et al. 2023, ICLR, what SSL ViTs learn. https://arxiv.org/abs/2305.00729
- Darcet et al. 2024, ICLR, registers. https://arxiv.org/abs/2309.16588
- Le et al. 2024, SAEs on PLUTO. https://arxiv.org/abs/2407.10785
- Rathjens et al. 2024/25, inverting transformer-based vision models. https://arxiv.org/abs/2412.06534
- Kömen et al. 2025, PathoROB. https://arxiv.org/abs/2507.17845
- Allakhverdov et al. 2025, image reconstruction as a tool for feature analysis. https://arxiv.org/abs/2506.07803
- Lei, Chen & Chen 2025, DRAG. https://arxiv.org/abs/2509.11724
- Filiot et al. 2026, robustifying pathology foundation models. https://arxiv.org/abs/2607.22861
- Leggio, Rossolini & Biondi 2026, Position Matters. https://arxiv.org/abs/2609.01232

Screened and cited:
- Wang et al. 2004, SSIM. https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
- Wang & Bovik 2009, MSE: love it or leave it? https://ece.uwaterloo.ca/~z70wang/publications/SPM09.pdf
- Zeiler & Fergus 2014. https://arxiv.org/abs/1311.2901
- Johnson et al. 2016, perceptual losses. https://arxiv.org/abs/1603.08155
- Dosovitskiy & Brox 2016, NeurIPS, DeePSiM. https://arxiv.org/abs/1602.02644
- Zhang, Lee & Lee 2016, ICML. https://proceedings.mlr.press/v48/zhangc16.html
- He, Wang & Hopcroft 2016, NeurIPS, random weights. https://arxiv.org/abs/1606.04801
- He et al. 2017, randomness in deconvolutional networks. https://arxiv.org/abs/1704.00330
- Wojna et al. 2017, the devil is in the decoder. https://arxiv.org/abs/1707.05847
- Blau & Michaeli 2018, perception-distortion tradeoff. https://arxiv.org/abs/1711.06077
- Adebayo et al. 2018, sanity checks for saliency maps. https://arxiv.org/abs/1810.03292
- Goldfeld et al. 2019, information flow (unverified). https://arxiv.org/abs/1810.05728
- Behrmann et al. 2019, invertible residual networks. https://arxiv.org/abs/1811.00995
- Rahaman et al. 2019, spectral bias. https://arxiv.org/abs/1806.08734
- Nash et al. 2019, autoregressive inversion. https://arxiv.org/abs/1806.00400
- Pimentel et al. 2020, information-theoretic probing. https://arxiv.org/abs/2004.03061
- Voita & Titov 2020, MDL probing. https://arxiv.org/abs/2003.12298
- Nilsson & Akenine-Möller 2020, understanding SSIM. https://arxiv.org/abs/2006.13846
- Rombach, Esser & Ommer 2020, ECCV. https://arxiv.org/abs/2008.01777
- Ding et al. 2021, IJCV, image quality models as losses. https://arxiv.org/abs/2005.01338
- Raghu et al. 2021, NeurIPS. https://arxiv.org/abs/2108.08810
- Hewitt et al. 2021, conditional probing. https://arxiv.org/abs/2109.09234
- Elazar et al. 2021, amnesic probing. https://arxiv.org/abs/2006.00995
- Elhage et al. 2021, transformer circuits. https://transformer-circuits.pub/2021/framework/index.html
- Teterwak et al. 2021, ICML. https://arxiv.org/abs/2103.07470
- Lee et al. 2021, AugSelf. https://arxiv.org/abs/2111.09613
- Bordes, Balestriero & Vincent 2022, RCDM. https://arxiv.org/abs/2112.09164
- Xie et al. 2022, SimMIM. https://arxiv.org/abs/2111.09886
- Zhou et al. 2022, iBOT. https://arxiv.org/abs/2111.07832
- Li et al. 2022, ResSFL. https://arxiv.org/abs/2205.04007
- Dong et al. 2022, BMVC, split-computing inversion. https://arxiv.org/abs/2107.06304
- Yang et al. 2022, NeurIPS, reconstruction defences. https://papers.neurips.cc/paper_files/paper/2022/file/53f1c3ec5df814b5aabe9ae88a29bb49-Paper-Conference.pdf
- Ikezogwo et al. 2022, MMAE. https://arxiv.org/abs/2209.01534
- Luo et al., SD-MAE. https://arxiv.org/abs/2203.16983
- Oquab et al. 2024, DINOv2. https://arxiv.org/abs/2304.07193
- Lampinen et al. 2024, TMLR, representation biases. https://arxiv.org/abs/2405.05847
- Juyal et al. 2024, PLUTO. https://arxiv.org/abs/2405.07905
- Graikos et al. 2024, CVPR. https://arxiv.org/abs/2312.07330
- Howard et al. 2024, HistoXGAN, Science Advances. https://pmc.ncbi.nlm.nih.gov/articles/PMC11567005/
- Zhou et al. 2024, DINO-WM. https://arxiv.org/abs/2411.04983
- Yellapragada et al. 2025, ZoomLDM, CVPR. https://arxiv.org/abs/2411.16969
- Yellapragada et al. 2025, PixCell. https://arxiv.org/abs/2506.05127
- Vadori et al. 2025, Mind the Gap. https://arxiv.org/abs/2502.02471
- Krishna Kumar 2025, non-monotonic scaling in ViTs. https://arxiv.org/abs/2511.21635
- Zheng et al. 2025, RAE. https://arxiv.org/abs/2510.11690
- Shi et al. 2025, SVG. https://arxiv.org/abs/2510.15301
- Jia et al. 2025, DINO-Tok. https://arxiv.org/abs/2511.20565
- Fan et al. 2025, Prism hypothesis. https://arxiv.org/abs/2512.19693
- Ren et al. 2025, FIA-Flow. https://arxiv.org/abs/2511.15316
- Maeng et al. 2023, NeurIPS, Fisher-information bounds. https://arxiv.org/abs/2305.04146
- Baxevanakis et al. 2026, do all ViTs need registers? https://arxiv.org/abs/2603.25803
- Tong et al. 2026, Beyond Language Modeling. https://arxiv.org/abs/2603.03276
- Wang et al. 2026, DecQ. https://arxiv.org/abs/2605.22777
- Spiridonov et al. 2026, layer-wise input recoverability. https://arxiv.org/abs/2610.14290
- Drexlin, Hense & Dippel 2026, decoding register tokens. https://arxiv.org/abs/2610.13299

## 12. Detailed summary

A compact recap of the whole survey, with the technical terms explained, for readers who want the short version.

### 12.1 The idea

For each saved layer (`block_0`, `block_4`, `block_8`, `norm`), a small network, the **decoder**, is trained to rebuild the tile image from that layer's activations. Where the model is frozen, the decoder is trained on the saved activations and the original tiles. The reconstructions show what visual information the layer still holds.

Terms in play:
- **Patch tokens:** the model splits a 224-pixel tile into 14 × 14-pixel patches (a 16 × 16 grid, 256 patches). Each layer keeps one 384-number vector per patch.
- **CLS token:** one extra vector that summarizes the whole tile.
- **Register tokens:** 4 extra learnable vectors with no image position. They hold global information. Our extraction does not save them.

### 12.2 How to read a result

| Term | Plain meaning | What it means for our results |
|---|---|---|
| **Usable information** (Xu et al. 2020) | Information measured relative to a chosen family of decoders. It is not the total information in the layer. | A reconstruction curve reflects what our decoder type can extract. Layers are comparable only with the same decoder and the same training budget. |
| **Expected pre-image** (Dosovitskiy & Brox 2016) | A decoder trained with squared pixel error outputs the average of all images that could produce those activations. | Blur means "the layer doesn't pin this detail down". It is a signal, not a bug. |
| **Invertible network** (Jacobsen et al. 2018, i-RevNet) | A network that loses no information, with an exact inverse. | The one with an exact inverse still classifies ImageNet well. So a falling curve doesn't prove information is lost, only that it became harder to extract. |
| **Residual stream** | Every transformer block adds its output to its input, so the input persists unless later blocks overwrite it. | Information can be recoverable without the model using it. |

### 12.3 Main findings

- **No close precedent.** No paper decodes a DINO/iBOT ViT one layer at a time with a fixed, non-generative decoder. None does this in pathology; 32 pathology searches found only generative image decoders. Our rig would fill both gaps.
- **Expected curve.** A gradual decline, as in other ViTs, not the steps CNNs show at pooling layers. Leggio 2026: SSIM falls 0.92 to 0.49 over blocks 1 to 11. Lei 2025: MS-SSIM 0.99 to 0.74.
- **What survives longest.** Colour and coarse layout. Fine texture and high frequencies fade first (Park 2023; DecQ Table 6). Contrastive training removes high frequencies in late layers (Park Fig. 9). Nobody has tested iBOT.
- **Reference number.** A final-layer DINOv2 ViT-S decodes to about 18 dB PSNR on ImageNet (SVG Table 3), with a much larger decoder and dataset than ours.
- **Stain.** Pathology models keep scanner and stain signals until their last few blocks (Filiot 2026). The centre is 88 to 98% linearly decodable from final embeddings (PathoROB). So stain colour is likely still recoverable at `block_8`. Whether our HED and colour jitter remove it by `norm` is untested anywhere, and inversion measures it directly.
- **Nuclei.** One 14-pixel patch is about 8 µm, roughly one lymphocyte nucleus (educational sources only, unverified). Shallow blocks are better at nucleus segmentation (Mind the Gap, Table I). Expect nuclei to blur first.

### 12.4 The design choices, with reasons

**Decoder**
- **Per-token linear head:** one linear map from one token to the 588 pixel values of its own patch. About 0.23M parameters. It is the image version of our linear probe, the most selective option, and it cannot use neighbouring tokens. It is the headline.
- **MAE-style decoder:** 2 to 4 transformer blocks that let tokens share information, then a per-patch head. It is the second size. The gap between the two shows how much of a patch's content moved into other tokens (ours).
- **Capacity:** weak decoders understate what is recoverable. Wider ones change absolute scores but rarely the ranking. So report two sizes.
- **Depth:** keep the same decoder for every layer. Growing it with depth, as some papers do, mixes depth with capacity.

**Input**
- **Patch tokens** are the main input. They stay tied to their own patch even at the last layer (Raghu 2021).
- **Per-layer standardization:** block outputs and `norm` differ in scale. Our probe briefing shows a norm of 3.55 at `block_0` versus 36.28 at `norm`. So standardize each channel with training-split statistics, per layer.
- **CLS or patch mean alone:** content but not layout (HistoXGAN; RCDM). That is a later extension.
- **Registers:** they carry global colour. Adding them raised PSNR from 19.49 to 19.73 dB in one study (Drexlin 2026). Saving them costs about 61 MB per layer and needs a re-extraction.

**Loss**
- **Pixel MSE** is the main loss. It matches the usable-information definition for a Gaussian likelihood.
- **Perceptual loss** compares features of a pretrained network. **Adversarial loss** uses a discriminator that judges realism. Both give sharper images with invented detail.
- **Normalized-pixel target** (from MAE): each patch is standardized by its own mean and spread. That removes the patch's colour, so it would hide stain.
- A sharp-output decoder costs at most about 3 dB of PSNR in theory (Blau & Michaeli 2018, Thm 3).

**Metrics**
- **PSNR** is mean squared error on a log scale (Wang & Bovik 2009). Higher is better, and +3 dB is about half the error.
- **R² over the mean image:** the share of pixel variance explained, compared with always predicting the average tile. With MSE this is a usable-information estimate. About 0.115 nats of information per dB gained (our derivation).
- **SSIM:** local brightness, contrast and structure, from 0 to 1. It was validated on brightness only and misbehaves in dark regions.
- **LPIPS:** distance between deep features, with weights fitted to human judgements. It was calibrated on natural-image patches, not H&E.
- **rFID:** the realism of a whole set of reconstructions, not per-image fidelity. An rFID of about 0.5 coexists with about 19 dB PSNR (RAE).
- **Colour versus structure error:** split the error into a mean-colour part and a structure part (Drexlin Eq. 5). That answers our stain question.
- **Our derived labels** (`brightness`, `tissue_brightness`, `tissue_fraction`, `hematoxylin`, `eosin`) recomputed on the reconstructions. We already have the code.
- **Re-encoding:** feed the reconstruction back through our model and probes, to see whether it keeps what the model uses.
- **Visuals:** image grids and per-token error maps. Numbers can mislead: an MSE above the mean-image level can still show structure (Rathjens Sec. 4.1).
- **Metric disagreement:** the metrics diverge with depth. Structure survives while perceptual similarity falls (Lei Table 2).

**Controls**
- **Mean image:** the floor, what you predict with no input.
- **Pixel ceiling:** the best linear 384-number code per patch (PCA-384). In `model.py` the patch embedding is `Conv2d(3, 384, kernel 14, stride 14)`, mapping 588 pixel values to 384 numbers. So even `block_0` cannot be lossless. The ceiling also shows blur caused by the decoder: networks fit coarse patterns before fine ones, called spectral bias (Rahaman 2019).
- **Random-init control:** the same architecture with untrained weights, decoded with the same decoder. Random networks are often *easier* to invert than trained ones (He 2017; Spiridonov 2026: negative gap in 10 of 13 CNNs). The informative number is trained minus random. Our LayerScale starts at 1, so this control is not just the identity.
- **Shuffled tokens:** the same decoder on tokens in random positions gave SSIM of only 0.18 to 0.27 at every depth (Leggio Fig. 2). It shows how much of the score comes from the known layout.
- **Seeds and budget:** 3 seeds, the same budget for every layer. Budget alone moved PSNR by about 2 dB in one study (Rojas-Gomez Table 1 versus Table 4).

**Data and splits**
- **Split by `slide_id`,** as the probe does. Slides repeat a stain and scanner signature, so a mixed split rewards memorization.
- **Size:** 10k tiles is small. One study saw gains flatten beyond about 10K training images (Dong 2022). A per-token decoder sees 2.56M patch examples, a whole-image decoder only 10k images, so the overfitting risk differs.
- **Centre hold-out:** train on RUMC and UMCU, test on CWZ, RST and LPON. It tests whether stain reconstruction generalizes to unseen shades. It is optional and later, because those three centres hold only about 9% of the tiles.

### 12.5 Generative decoders (brief)

They are common: frozen DINOv2 encoders serve as image tokenizers, and pathology diffusion models condition on embeddings. We leave them out because they invent detail. What their papers say:
- **Detail sits early:** adding the first 4 layers lifts DINOv2-B PSNR from 19.61 to 23.47 dB, and the last 4 layers only reach 20.72 dB (DecQ Table 6).
- **Decoder size:** going from 125M to 268M parameters moves PSNR from 19.44 to 19.56 dB (DINO-Tok Fig. 8).
- **Colour:** the backbone keeps colour palette and grayscale status despite colour jitter, while the projection head discards them (RCDM). All our saved layers are in the backbone.
- **Pathology:** one vector fixes content, not layout (HistoXGAN; Graikos 2024).

### 12.6 Pitfalls and guards

| Pitfall | Guard |
|---|---|
| "Falling curve means information is gone" | Say "less usable by this decoder"; use two decoder sizes |
| "Good reconstruction means the layer learned something" | Report trained minus random-init |
| "Recoverable means used" | Phrase results as availability |
| Decoder fills in colour or texture from its learned prior | Mean-image floor, held-out slides, image grids |
| Blur caused by the decoder | Pixel ceiling with the same decoder |
| `block_0` wins only for being closest to the pixels | Read the curve relative to the ceiling |
| Colour dominates the error | Split colour and structure error |
| Memorization, and stain confounded with slide and centre | Slide split, train versus held-out gap |

### 12.7 Recommended first version

- **Decoders:** per-token linear (headline) plus a small MAE-style decoder, on the saved patch tokens of each kept layer.
- **Training:** pixel MSE, per-layer standardization, same budget and 3 seeds for every layer.
- **Evaluation:** slide-grouped split, with the metrics and controls above.

| Decision | Recommendation |
|---|---|
| Per-token linear as the headline | Yes, with the transformer as the second size |
| Random-init control | Yes, one extra extraction |
| Register tokens | Skip for now, add later |
| CLS or patch-mean decoder | Skip for the first version |
| Split | One slide-grouped split, for example 75/25 |
| Centre hold-out | Later |
| Normalized-pixel variant | Skip, it removes colour |

### 12.8 Reliability

- **Checked:** 10 key numbers against the downloaded paper texts and 3 facts against `model.py` (patch embedding, registers not saved, LayerScale initialization).
- **Not checked:** the rest comes from six parallel research tracks that were stopped early for budget. Claims seen only in abstracts are marked "unverified" in the document.
