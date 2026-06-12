# JEPA × FINO campaign log

**Goal:** maximize mean_probe_score. Beat the live Labless leader **jepa-mask25 = 0.6471** (I-JEPA contig-patch,
NimaAsh). Hypothesis: FINO metadata-guidance acts on the CLS token and is **orthogonal** to the JEPA patch
objective (DINO CLS + smooth-L1 latent regression + KDE), so it should *stack* — unlike on the DINO+iBOT base
where FINO's best (subtype+expr512) only reached 0.6351 (+0.0074 over 0.6277, below the +0.01 bar).

**Setup:** worktree `nanopath-jepafino` (branch `jepa-fino`), FINO cherry-picked onto `origin/leader`. JEPA
objective untouched; FINO grafted as a 4th loss term `meta` on the CLS token (GradScale DANN gate, EMA prototype
banks for discrete factors, MLP regressors for continuous, lambda_meta=0.03/branch). Smoke (job 81946) validated:
dino+jepa+kde+meta all flow, val + checkpoint(JEPA-predictor + FINO protos/predictors) OK. NOT submitting to labless.

Reference points (live board, same probe suite): leader **0.6471** (jepa family) · curation `lr-and-curation`
0.6357 · `dinov2-s-kde` 0.6277. FINO-on-DINO best: `abl_sub_expr512` 0.6351, `abl_subtype` 0.6323.

## Wave 1 (launched 2026-06-11 ~06:18, 4-wide, pinned n-1/3/4/8)
| id | job | recipe Δ vs JEPA leader (main.yaml) | M+ / M- | final | Δ vs 0.6471 | decision |
|----|-----|------------------------------------|---------|------:|------------:|----------|
| W1-base | 81947 | none — JEPA leader control (harness reproducibility) | — | running | — | — |
| W1-se   | 81948 | + FINO subtype + expr512 | subtype, expr512 M+ | running | — | — |
| W1-s    | 81949 | + FINO subtype | subtype M+ | running | — | — |
| W1-se-rr| 81950 | + FINO subtype + expr512, ramp:run | subtype, expr512 M+ | running | — | — |

Hypotheses: W1-base confirms we reproduce 0.6471 (control for all FINO deltas). W1-se = best DINO-era FINO lever
on JEPA. W1-s isolates the histotype anchor (most robust single factor in the DINO sweep). W1-se-rr tests the
run-keyed DANN ramp (nanopath is sample-capped at ~19% of FLOP budget; flop-keyed gamma stalls at ~0.74).
Results + Wave 2 design pending (~1 hr).

## Literature (scan 2026-06-11, ingested AdvDINO -> geist)
- **Novelty:** FINO M+/M- on a JEPA objective in pathology is novel. Precedents: AdvDINO (2508.04955, DANN on
  slide-ID atop DINOv2 — gate stacks, slide-ARI 0.663->0.037, survival +0.010); JEPA-T (2510.00974, metadata-
  conditioned JEPA predictor, T2I not pathology); GenBio-PathFM (JEPA as stage-2 on frozen DINO). None combine all.
- **JEPA framing:** "Pretext Matters" (2603.22649) — JEAs > JEPAs for spatially-localized signal (tile pathology);
  JEPA earns its keep only as an AUXILIARY on a DINO spine. Our leader recipe IS that shape -> keep DINO-CLS spine.
- **Weighting:** don't import lambda blindly (AdvDINO 50 vs FINO 0.03; magnitudes differ). GRADNORM-MATCH the meta
  branch to the SSL gradient. We log grad_norm -> use it. Wave-2 lever: sweep gamma_max / lambda_meta for JEPA.
- **Entanglement gotcha:** U(cancer|TSS)=1.0 -> solo TSS/scanner-M- reverses cancer signal; suppression must be
  PAIRED with an M+ anchor (subtype/cancer). Pure-suppression alone ~+0.01 (borderline). M+ anchor carries the EV.
- **Mechanism check:** for any M- run, measure slide/TSS-clustering ARI drop ("adversary working" signal) before
  expecting probe gains. **Novel idea (Wave 2/3):** condition JEPAPredictor on cancer-type embedding (JEPA-T style).

## Wave 2 candidate directions (finalize FROM Wave-1 results)
- if jf_sub_expr512 > jepa_base by >+0.01: push M+ stacking (subtype+expr512+morphology / +fga) + gamma_max sweep.
- if FINO ~flat on JEPA: gradnorm-match meta (raise gamma_max), OR try the JEPA-T metadata-conditioned predictor.
- paired cancer-M+/TSS-M- (entanglement-safe suppression) as a separate track once an M+ anchor is confirmed.

## Wave 1 RESULTS (2026-06-11 07:39) — FINO stacks on JEPA
| id | job | final | Δ vs base | decision |
|----|-----|------:|----------:|----------|
| W1-base | jepa_base | 0.6436 | — | control (≈ board I-JEPA-contig 0.6444 ref; mask10 recipe, not the 0.6471 mask25) |
| W1-se | jf_sub_expr512 | **0.6473** | **+0.0037** | KEEP — best; ≈ live leader 0.6471 |
| W1-s | jf_sub | 0.6454 | +0.0018 | subtype alone weaker than +expr512 |
| W1-se-rr | jf_sub_expr512_rr | 0.6453 | +0.0017 | ramp:run no better than flop -> DROP ramp:run |

Components (jf_sub_expr512 vs base): knn +0.022, survival +0.025, fewshot +0.008 (FINO M+ wins) | seg -0.023,
slide -0.006 (CLS steering hurts dense/local). Net +0.0037 (below +0.01 bar). **Seg loss is the drag.** FINO helps
LESS on JEPA (+0.0037) than on DINO+iBOT (+0.0074) — JEPA's patch objective already captures some of what FINO adds.

## Wave 2 (launched 07:39, 4-wide) — map gamma_max + anchor stacking on the best recipe
| id | job | recipe Δ vs jf_sub_expr512 | hypothesis |
|----|-----|---------------------------|-----------|
| W2-g05 | jf_se_g05 | gamma_max 1.0->0.5 | gentler guidance preserves seg/slide, keeps some knn/surv -> better net |
| W2-g15 | jf_se_g15 | gamma_max 1.0->1.5 | stronger (gradnorm-match) -> more M+ gain if seg loss sublinear |
| W2-g20 | jf_se_g20 | gamma_max 1.0->2.0 | upper bound of the gamma curve |
| W2-morph | jf_se_morph | + morphology M+ (gamma 1.0) | 2nd-best DINO anchor; does stacking add on JEPA |

## Wave 2 RESULTS (08:50) — gentler gamma wins (monotonic)
| id | gamma | score | knn | seg | surv | note |
|----|------:|------:|----:|----:|-----:|------|
| W2-g05 | 0.5 | **0.6482** | 0.714 | 0.304 | 0.591 | NEW BEST; +0.0046 vs base, +0.0011 vs leader 0.6471 |
| W2-morph | 1.0 | 0.6477 | 0.723 | 0.298 | 0.579 | +morph: knn up, surv down -> wash |
| (W1-se) | 1.0 | 0.6473 | 0.718 | 0.285 | 0.597 | prior best |
| W2-g15 | 1.5 | 0.6469 | 0.720 | 0.293 | 0.590 | |
| W2-g20 | 2.0 | 0.6462 | 0.718 | 0.288 | 0.583 | too strong; seg loss dominates |
Trend: gamma 0.5>1.0>1.5>2.0 monotonic. Seg loss scales with gamma; gentle gamma keeps M+ knn/surv gains w/o the
seg cost. Optimum at low gamma. NOTE deltas now ~noise floor (DINO-era 3-seed std 0.0005) -> reseed best.

## Wave 3 (launched 08:50, 4-wide) — find gamma floor + confirm best + stack at good gamma
| id | job | recipe | hypothesis |
|----|-----|--------|-----------|
| W3-g03 | jf_se_g03 | subtype+expr512, gamma 0.3 | even gentler — is optimum <0.5 or does it decay to control? |
| W3-g05s2 | jf_se_g05_s2 | gamma 0.5, seed 1337 | RESEED of best — is 0.6482 real vs noise? |
| W3-morph-g05 | jf_morph_g05 | +morphology, gamma 0.5 | best stack at gentle gamma (less seg damage) |
| W3-fga-g05 | jf_se_fga_g05 | +fga (continuous), gamma 0.5 | DINO-era #3 stack at gentle gamma |

## Wave 3 RESULTS (10:01) — best recipe beats leader (2-seed); gamma peak = 0.5
| id | recipe | score | knn | seg | surv | note |
|----|--------|------:|----:|----:|-----:|------|
| W3-g05s2 | subtype+expr512 g0.5 seed1337 | **0.6510** | 0.719 | 0.303 | 0.601 | reseed > orig 0.6482; +0.0074 vs control |
| W3-fga-g05 | +fga g0.5 | 0.6477 | 0.709 | 0.299 | 0.589 | neutral |
| W3-g03 | g0.3 | 0.6465 | 0.717 | 0.302 | 0.581 | too gentle; peak is 0.5 |
| W3-morph-g05 | +morph g0.5 | 0.6429 | 0.709 | 0.291 | 0.560 | morph HURTS (surv crash) |
Best = subtype+expr512 @ gamma 0.5: seeds {0.6482, 0.6510} mean ~0.6496 (both > leader 0.6471, > control 0.6436).
gamma peak confirmed at 0.5; at g0.5 seg loss ~vanishes -> seg no longer the limit, M+ magnitude is. Stacking
(morph/fga) does not help -> subtype+expr512 is THE recipe. Noise ~0.0028 -> need more seeds for a firm lead claim.

## Wave 4 (launched 10:01, 4-wide) — exploit config space + 3rd seed
| id | job | recipe | hypothesis |
|----|-----|--------|-----------|
| W4-g065 | jf_se_g065 | subtype+expr512 gamma 0.65 | true peak between 0.5 and 1.0? |
| W4-cancer | jf_cancer_expr512_g05 | cancer+expr512 g0.5 | organ-level anchor cleaner than 40-class subtype? |
| W4-til | jf_subexpr_til_g05 | subtype+expr512+til g0.5 | TIL was DINO-era best generalist add (E19) |
| W4-g05s3 | jf_se_g05_s3 | subtype+expr512 g0.5 seed2024 | 3rd seed -> credible mean for the lead claim |

## Wave 4 RESULTS (11:12) — config space converged
| id | recipe | score | note |
|----|--------|------:|------|
| W4-til | subtype+expr512+til g0.5 | 0.6491 | marginal best of wave (1 seed) |
| W4-g065 | g0.65 | 0.6470 | confirms gamma peak = 0.5 |
| W4-g05s3 | g0.5 seed2024 | 0.6450 | 3rd seed of best |
| W4-cancer | cancer+expr512 g0.5 | 0.6444 | subtype > cancer anchor |
**Best recipe (subtype+expr512 @ gamma0.5) 3-seed = 0.6481 ± 0.0025 {.6482,.6510,.6450}** — tied w/ leader 0.6471,
+0.0045 vs control 0.6436. Sub-threshold (< +0.01), same ceiling as DINO era. Config tuning tapped out.

## Wave 5 (JEPA-T structural lever) — metadata-condition the JEPA predictor
Idea (novel in pathology): inject a learned subtype embedding into the JEPAPredictor so the masked-patch
latent-regression target is metadata-aware — a different mechanism than CLS-steering (which caps at the seg/M+
tradeoff). model.py JEPAPredictor gets n_cond + cond_emb; train.py passes per-image subtype label (crop-major).
| id | job | recipe | hypothesis |
|----|-----|--------|-----------|
| W5-jt | jepa_t_sub | pure JEPA-T (subtype cond, no CLS M+/M-) | does conditioning the predictor alone help dense/knn? |
| W5-jtf | jepa_t_fino | JEPA-T cond + CLS-FINO subtype+expr512 g0.5 | both mechanisms stacked |
| W5-s4 | jf_se_g05_s4 | best recipe, seed 7 | 4th seed -> firm mean |
| W5-base2 | jepa_base_s2 | control, seed 1337 | control reseed -> honest delta |

## Wave 5 RESULTS (12:30) — NULL: FINO is a wash on JEPA (control reseed exposed it)
| id | recipe | score | note |
|----|--------|------:|------|
| W5-base2 | control seed1337 | **0.6512** | control reseed JUMPS from 0.6436 -> control is high-variance |
| W5-s4 | FINO subtype+expr512 g0.5 seed7 | 0.6456 | 4th FINO seed |
| W5-jt | pure JEPA-T (subtype cond) | 0.6435 | = control; conditioning predictor alone does NOTHING |
| W5-jtf | JEPA-T + CLS-FINO | 0.6414 | WORSE; surv crash 0.547. JEPA-T net-negative |

### Honest conclusion (the headline)
- **FINO subtype+expr512 @ g0.5:** 4 seeds {0.6482,0.6510,0.6450,0.6456} = **0.6475 ± 0.0026**
- **JEPA control (no FINO):** 2 seeds {0.6436,0.6512} = **0.6474 ± 0.0038** (4v4 confirmation 82135/82136 running)
- **=> FINO provides ZERO net lift on the JEPA base.** The earlier "+0.0046" was a single-seed artifact: the
  first control draw (0.6436) was a low outlier; reseeding it -> 0.6512 collapses the effect. FINO only RESHUFFLES
  the probe profile (knn +0.02, survival +0.025 / seg -0.005..-0.02, slide -0.006), mean unchanged.
- **JEPA-T conditioned predictor:** neutral (pure, =control) to negative (stacked). Dead end.
- Mirrors & strengthens the DINO-era read: on a strong, well-tuned SSL base, metadata guidance is redundant. The
  single-seed noise floor here (~0.003) is LARGER than the effect; the DINO-era +0.0074 was likely also seed luck.

### What worked / didn't (quantitative)
- gamma_max sweep is real & monotonic (0.5>1.0>1.5>2.0) — but it's tuning the *shape* of a zero-mean perturbation.
- subtype > cancer anchor; stacking (morph/fga/til) neutral-to-negative; ramp:run no help.
- Integration itself is clean & validated (FINO grafts onto JEPA with no interaction bug; smoke + 14 full runs, 0 errors).

### Ceiling read
At ViT-S / 1M tiles, the JEPA leader recipe is at a plateau where metadata guidance can't move the *mean* — only
trade probe categories. A real gain needs a different axis (scale, data curation, or the objective itself), not
metadata. FINO's value here is as a *profile knob* (buy survival/knn at the cost of seg/slide), not a mean lift.
NOTHING submitted to labless (per standing instruction + it's a null anyway).

## FINAL 4v4 paired (13:41) — locked
CONTROL (no FINO), seeds {7777,1337,2024,7}: 0.6436, 0.6512, 0.6445, 0.6408 -> mean 0.6450 sd 0.0044
FINO subtype+expr512 g0.5, same seeds:        0.6482, 0.6510, 0.6450, 0.6456 -> mean 0.6475 sd 0.0027
**FINO - CONTROL = +0.0024, SE~0.0026, t~0.9, p~0.4 -> NOT SIGNIFICANT; ~4x under the +0.01 bar.**
Point estimate is a hair positive (and FINO has lower variance), but indistinguishable from zero at n=4 given the
control's large seed-variance (range 0.0104). Verdict: no reliable improvement. FINO mean 0.6475 ~ leader 0.6471.
CAMPAIGN CLOSED (past 7h window). Nothing submitted. FINO = a probe-profile knob, not a mean lift, on the JEPA base.

## SWEEP (Wave 6+) — broad M+/M- factor exploration (20-config matrix from 4-lens design workflow)
Extended fino_meta.json: 34 discrete + 22 continuous (curated 33 + 12 dense raw TCGA cols: project_id/race/
ethnicity/country/year_of_diagnosis + slide_percent_{tumor_nuclei,stromal,necrosis,lymphocyte,normal} +
cbio_{mutation_count,fraction_genome_altered,msi_score,subtype} + ajcc_stage/t/n/m + tumor_grade + lymph_nodes_pos).
Reference: jepa control 0.6450 (4-seed), best-so-far jf_se_g05 (=matrix #3) 0.6481 (4-seed). 20 configs ranked by EV;
running 4-wide, M-suppression + M+morphology (the untested theory-backed levers) first.

## Wave 6 (launched) — one bet per untested regime
| rank | job | recipe | regime / hypothesis |
|-----:|-----|--------|---------------------|
| #1 | jf_supp_scanner_sub_expr_g08 | subtype+ scanner- expr512+, g0.8 | M- scanner suppression (AdvDINO domain-gen) on the proven anchor — THE untested lever |
| #2 | jf_sub_morph_comp | subtype+ slide_%_tumor_nuclei+, g0.5 | M+ tile-readable morphology (vs latent expr) |
| #4 | jf_stack_sub_expr512_fga | subtype+ expr512+ fga+, g0.5 | DINO-era's one robust lever: orthogonal histotype+transcriptome+CNV stack |
| #5 | jf_sub_comp_immune | subtype+ tumor_nuclei+ lymphocyte+, g0.5 | orthogonal morphology stack (composition + immune) |
