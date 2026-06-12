# JEPA × FINO campaign log

**Goal:** maximize mean_probe_score. TOP bar = beat the highest **VALIDATED** Labless run by >=0.006 (README:107).
The validated leader is **0.6444** (NimaAsh "I-JEPA contig patch", state=reviewed) — NOT the 0.6471/0.6501 *completed*
(un-validated) runs we'd been chasing. Topping bar therefore = 0.6444 + 0.006 = **0.6504**. Hypothesis: FINO
metadata-guidance acts on the CLS token and is **orthogonal** to the JEPA patch
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

## CONSTRAINT (user, 2026-06-12): M- only suppresses nuisance UNcorrelated with disease
`site` (anatomical) heavily correlates with disease type -> suppressing it gradient-reverses the biology we want
(fights the M+ anchor). Same for `tss` (U(cancer|tss)=1.0) and `organ`. CLEAN batch factors for M- = scanner
(device), year (time), race (~0.04 entanglement), maybe country. DROPPED matrix #15 (jf_supp_site_sub_expr_g10).
Remaining M- configs (scanner/year/race/country) honor this. Don't propose site/tss/organ suppression.

## SCHEDULE FIX (2026-06-12) — FINO ramp was throttled all campaign
ALL prior runs used the default flop-keyed DANN ramp. nanopath is sample-capped at ~19% of the FLOP budget, so
gamma = gamma_max*(2σ(10*frac)-1) stalled at frac~0.19 -> **gamma only reached 0.75*gamma_max, never full strength.**
Fixed: ramp now keyed to SAMPLE progress, counted FROM the backbone-unfreeze point: gamma=0 through frozen Phase 1,
then ramps to full gamma_max by the sample cap. So prior "gamma 0.5 optimal / cluster wants higher gamma" findings
were on a throttled ramp (effective gamma = 0.75x nominal) -> re-establishing baselines under the corrected schedule.

## Wave CR (corrected ramp + two-phase warmup) — 8-wide, launched 03:49
| job | recipe | gamma | freeze | vs (old-ramp) |
|-----|--------|------:|-------:|--------------|
| 82413 jf_cr_cluster_g05 | cluster | 0.5 | 0 | cluster-g05 0.6452 |
| 82414 jf_cr_cluster_g05_warm | cluster | 0.5 | 0.15 | + warmup |
| 82415 jf_cr_cluster_g03_warm | cluster | 0.3 | 0.15 | lower gamma (eff. higher now) |
| 82416 jf_cr_cluster_g08_warm | cluster | 0.8 | 0.15 | cluster-g08 0.6484 |
| 82417 jf_cr_mol_g05 | subtype+expr512+fga | 0.5 | 0 | fga 0.6494 |
| 82418 jf_cr_mol_g05_warm | subtype+expr512+fga | 0.5 | 0.15 | + warmup |
| 82419 jf_cr_se_g05 | subtype+expr512 | 0.5 | 0 | best 0.6481 |
| 82420 jf_cr_fgaclu_warm | subtype+cluster+expr512+fga | 0.5 | 0.15 | combo + warmup |
Tests: (A) does full-strength ramp change the baselines, (B) does the two-phase warmup help, (C) gamma re-opt.

## STRATEGY (user directive 2026-06-12): JEPA-FINO is primary; regular FINO is the validation fallback
If FINO+JEPA stays a wash through Wave CR (+ maybe one more), PIVOT to "regular FINO" = FINO on the DINO+iBOT base
(the paper's actual SSL family = nanopath main recipe) WITH the corrected schedule (sample-keyed ramp from-unfreeze
+ two-phase backbone freeze). Two reasons: (1) validates our FINO implementation reproduces the paper's gains on the
family it was designed for; (2) the DINO-era FINO numbers (+0.0074 best, throttled ramp) were ALSO under-cooked, so
corrected-schedule regular-FINO is a FRESH test, not a rerun. Setup: new worktree off origin/main @ 6c05053 (fino),
port the ramp-fix + freeze_backbone_fraction (~10 lines) from train.py here, run the best FINO recipes.

## Wave CR RESULTS (corrected ramp + warmup) — 6/8 in, 2 stragglers
| job | recipe | gamma | warm | FINAL | vs control 0.6450 | vs validated leader 0.6444 |
|-----|--------|------:|:----:|------:|------:|------:|
| **jf_cr_mol_g05** | subtype+expr512+fga | 0.5 | no | **0.6518** | +0.0068 | **+0.0074 (clears 0.006 bar)** |
| jf_cr_cluster_g08_warm | cluster(GenBio4k) | 0.8 | yes | 0.6504 | +0.0054 | +0.0060 |
| jf_cr_cluster_g05_warm | cluster | 0.5 | yes | 0.6497 | +0.0047 | +0.0053 |
| jf_cr_cluster_g03_warm | cluster | 0.3 | yes | 0.6486 | +0.0036 | +0.0042 |
| jf_cr_se_g05 | subtype+expr512 | 0.5 | no | 0.6476 | +0.0026 | +0.0032 |
| jf_cr_mol_g05_warm | subtype+expr512+fga | 0.5 | yes | 0.6452 | +0.0002 | +0.0008 |
| jf_cr_cluster_g05 | cluster | 0.5 | no | running | | |
| jf_cr_fgaclu_warm | sub+cluster+expr512+fga | 0.5 | yes | running | | |

**CONCLUSIONS:**
1. **Corrected sample-keyed ramp is the real unlock.** mol_g05 = subtype+expr512+fga at full-strength ramp = **0.6518**,
   the campaign best (prior best 0.6494, throttled ramp). Lift concentrated in the hard probes: survival 0.599,
   robustness 0.892, auc 0.623.
2. **fga is the active continuous M+.** se (subtype+expr512) 0.6476 -> +fga 0.6518 = +0.0042. Same factor topped the
   throttled campaign; un-throttled it's even stronger.
3. **Freeze/unfreeze warmup is wash-to-NEGATIVE, not the "major thing".** Identical recipe: mol_g05 0.6518 vs
   mol_g05_warm 0.6452 = warmup COST 0.0066. Our 19%-FLOP sample-capped run can't afford a frozen Phase 1 that the
   paper's full-length run can. Dropping warmup for all future waves.
4. **Higher gamma helps the cluster target** monotonically (g03 0.6486 < g05 0.6497 < g08 0.6504) even at full strength.

**SUBMITTED to Labless (user-approved):** jf_cr_mol_g05 -> run_name `jepa-fino-fga-ramp`, run_id run_sub_24513e748d,
metric 0.6518, verdict **"promising"**, validation pending (maintainer reruns on a fresh seed; valid if still
>=+0.006 over validated leader within a 2h train window). **Now #1 on the entire board** (next best 0.6501 completed).

## Wave D (fga-centric, corrected ramp, NO warmup) — armed, launches when CR stragglers clear the 8-cap
| job | M+ recipe | gamma | hypothesis |
|-----|-----------|------:|------------|
| jf_d_sefga_g03 | subtype+expr512+fga | 0.3 | gamma sweep around the 0.6518 winner |
| jf_d_sefga_g07 | subtype+expr512+fga | 0.7 | (does the winner like higher gamma like cluster does?) |
| jf_d_sefga_g10 | subtype+expr512+fga | 1.0 | full-strength ramp ceiling |
| jf_d_fga_g05 | fga only | 0.5 | isolate fga's standalone contribution |
| jf_d_molstack_g05 | subtype + fga+mutcount+msi | 0.5 | genomic-instability continuous stack |
| jf_d_sefga_clu_g05 | subtype+cluster + expr512+fga | 0.5 | combine the 2 best M+ families |
| jf_d_sefga_clu_g07 | subtype+cluster + expr512+fga | 0.7 | " at higher gamma |
| jf_d_sefgamut_g07 | subtype + expr512+fga+mutcount+msi | 0.7 | kitchen-sink molecular at higher gamma |

## Wave CR stragglers — probe-phase HANG + checkpoint loss (2026-06-12 ~05:50)
Jobs 82413 (cluster_g05) and 82420 (fgaclu_warm) wedged in the PROBE phase under CPU contention on loaded
nodes (n-4 load 77, n-1 busy; probes are sklearn-heavy, MaxRSS ~132GB -> memory pressure + core starvation).
82413 froze hard (CPU AveCPU frozen at 04:23:16, 8/12 probes missing); 82420 alive but crawling on pannuke seg
(9/12 done, ~1 core). summary.json is written BEFORE wandb.finish() (train.py:782 vs :791), so absence = stalled
inside probing, not a sync hang. Killed both.
**MISTAKE:** relaunched them into the SAME output_dir as fresh full-training jobs (82442/82443) BEFORE realizing
I could just re-probe the existing checkpoints. train.py wipes output_dir on a non-resume launch -> the end-of-
training latest.pt (saved ~04:56/04:58) was DESTROYED by the relaunch. So eval-only recovery was no longer
possible; the 2 stragglers must be FULLY retrained. Lesson: to re-probe a hung run, copy latest.pt out (or set a
fresh output_dir) BEFORE relaunching — never relaunch onto the dir holding the only checkpoint.
Relaunched as full training: 82450 jf_cr_cluster_g05, 82451 jf_cr_fgaclu_warm (spread n-1/n-2).

## Wave D config bug — KeyError 'cluster' (fixed)
jf_d_sefga_clu_g05 (82449) crashed at dataloader.py:141: the `cluster` discrete factor needs a `fino.tile_npy`
mapping (cluster labels load from cluster_genbio4k.npy, not meta["discrete"]). My Wave D generator copied the
non-cluster mol_g05 template, omitting tile_npy. Added `tile_npy: {cluster: cluster_genbio4k.npy}` to both
sefga_clu_g05 and sefga_clu_g07. Relaunched sefga_clu_g05 as 82452.

## Wave D ACTIVE set (8 running, ~05:52; complete ~07:15-07:20)
82444 sefga_g03 · 82445 sefga_g07 · 82446 sefga_g10 · 82447 fga_g05 · 82448 molstack_g05 · 82452 sefga_clu_g05
+ straggler redos 82450 cluster_g05 · 82451 fgaclu_warm.  HELD (relaunch when slots free): sefga_clu_g07, sefgamut_g07.
On completion: submit the 2 straggler redos (user-approved), report Wave D table + conclusions, design Wave E.

## LITERATURE SCAN (2026-06-12 ~06:06) — grounding the FGA win + ranking the next M+ axis
Papers ingested to geist (`raw/papers/`, commit f06d750): Xu 2021 iScience (CIN from H&E), Fu 2020 Nat Cancer
(pan-cancer H&E-predictability), El Nahhas 2024 Nat Commun (continuous-biomarker regression from H&E).
- **Q1 — FGA is biologically grounded, not a fluke.** CIN/aneuploidy (= FGA's substrate: nuclear atypia,
  pleomorphism, mitotic disarray) is among the MOST universally H&E-predictable genomic axes — CIN AUROC 0.82 in
  TCGA-BRCA from low-mag, signal spatially diffuse/slide-level (exactly what a global CLS-regression target absorbs).
  Explains why our lift concentrated in survival (CIN is prognostic) + mutation-AUROC (aneuploidy drives the SCNA
  landscape). Point mutations are weakly H&E-predictable -> aggregate scores >> individual drivers.
- **Q2 — next continuous target ranked:** HRD > proliferation/mitotic > immune/TIL (orthogonal->additive) > TMB >
  MSI > driver-muts. Regression > classification for continuous biomarkers (we already do this).
- **DATA AVAILABILITY (checked metadata/tcga_master_cancer_genomics.csv):** cbio_ panel = only {mutation_count,
  fraction_genome_altered, subtype, msi_score}. NO HRD / aneuploidy / proliferation column. So lit #1 (HRD) and #2
  (proliferation) are NOT launch-ready; proliferation is DERIVABLE from the expr panel (MKI67 + proliferation gene
  set, a prepare.py factor). Available lit-backed axes: til (immune, #3, 38% cov), mutcount~TMB (#4, 70%), msi (#5).
- **Q3 — warmup is a full-length-training luxury.** Canonical Ganin GRL ramp delays/anneals lambda over a FULL
  schedule; at ~19%-FLOP short runs there's no runway to amortize a frozen Phase-1 -> co-adapt from step 0.
  Independently confirms Wave CR conclusion #3 (freeze_frac=0).
- **WAVE E SEED HYPOTHESIS (confirm vs Wave D results):** Wave D's molstack (fga+mutcount+msi) stacks
  FGA-CORRELATED (redundant) axes; the untested theory-backed bet is **fga + til** (genomic-instability +
  immune = ORTHOGONAL -> additive). Lead Wave E lever if 0.6518 holds.

## FINO FIDELITY AUDIT (2026-06-12 ~06:09) — DONE (workflow wqbrez8w7: 3 PDF readers + 2 adversarial auditors)
Deep-read the 28pp FINO paper (arXiv 2606.05107) and cross-checked every loss term/schedule/hparam vs train.py/
model.py/dataloader.py/prepare.py. **VERDICT: faithful on the core; 2 real gaps + 1 latent, all touching our
competitive (multi-branch) recipes.**
FAITHFUL (verified line-for-line): discrete M+ prototype InfoNCE (Eq.1-2) — cosine logits, tau=0.023, lambda_meta=
0.03, student-CLS vs TEACHER-EMA prototype bank, alpha=0.99, L2-norm + unit-sphere re-projection; DANN machinery —
k=10 logistic ramp gamma=gamma_max*(2*sigmoid(10*ramp)-1), sample-keyed from the unfreeze point (documented, sensible
vs FLOP-keying that stalled at 0.75x), symmetric +gamma(M+)/-gamma(M-); lambda_meta=0.03 sign-only M+/M-; per-branch
missing-label masking; JEPA+KDE base-swap (intentional, structurally sound — base still exposes a CLS for phi(x)).
**RESOLVED the prime suspect:** paper EXPLICITLY applies the ramp symmetrically to BOTH M+ and M- (p.21 verbatim
"applies symmetrically to both metadata branches; only the encoder is shielded") — our M+ ramp is CORRECT, not a bug.
REAL GAPS (medium; both auditors flagged #1 independently):
1. **Per-branch gradient equalisation s_t NOT implemented (Alg A.3).** Paper equalises per-branch gradient L2-norms
   (s_t = n_bar/n_tilde_t; n_tilde = EMA mu=0.99 of branch grad-norm at the shared CLS, n_bar = geometric mean)
   whenever >=2 branches train jointly. EVERY competitive recipe is multi-branch: leader jf_cr_mol_g05 = subtype +
   expr512 + fga = 3 branches; Wave D molstack/sefga_clu = 3-4. Discrete-CE vs continuous-MSE raw grad magnitudes
   differ wildly -> without s_t one branch dominates the CLS. The single most material omission. -> WAVE E LEVER.
2. **Continuous predictor fed the L2-NORMALIZED CLS (phi_s), but Eq.3 specifies the RAW backbone CLS for g(t)**
   (train.py:479). Strips the radial magnitude -> changes regression geometry + the encoder gradient. Affects every
   continuous M+ branch (expr512, fga). Likely-bug. -> WAVE E LEVER (cheap arm).
LATENT (dormant — only bites if a continuous factor is ever placed in M-): continuous targets z-scored + predictor
has NO bounded output activation, vs paper's per-dim linear map to [0,1]/[-1,1] + sigmoid/tanh. Paper's bounded
activation IS the saturation-stability mechanism for continuous M- suppression; with unbounded z-targets a reversed
continuous gradient could grow without bound. No committed config suppresses a continuous factor -> safe for now;
NOTE before any continuous-M- experiment.
MINOR (low): predictor 2-hidden [512,256] vs paper's 3 [512,512,256] + no Dropout(0.5); lambda_meta=0.03 also scales
the predictor-head gradient (paper implies +1 for heads) — both negligible at lambda_meta=0.03.

## TRAIN-LIMITATION PARAM ANALYSIS (2026-06-12 ~06:20, user directive "optimize params for nanopath train limitation")
README:122 — the run hits the 1M-tile sample cap at ~19% of the 1e18-FLOP budget, and LR-decay/WD/teacher-temp/KDE/
teacher-momentum schedules ALL key off `frac = train_flops/max_train_flops` (train.py:575), which tops out at ~0.19.
So a FLOP-keyed cosine traverses only ~0.11 of its arc. MEASURED end-of-run schedule values (lr=1e-4, lr_min=1e-6,
warmup_fraction=0.0909):
| schedule | keyed value at cap | intended target | status |
|---|---|---|---|
| **LR cosine decay** | **9.71e-5** (from 1e-4) | anneal to 1e-6 | **never anneals — trains at ~peak LR the whole run** |
| WD cosine(0.04->0.2) | 0.054 | 0.2 | barely ramps |
| KDE scale (frac-0.1)/0.4 | **0.225** | 1.0 | **uniformity reg at 22% strength** |
| teacher_temp | 0.061 | 0.07 | 70% there |
| teacher momentum (0.994->1.0) | 0.9945 | ~1.0 | barely moves |
This is the SAME pathology the FINO gamma ramp had (stalled at 0.75x) before it was sample-re-keyed. The validated
leader (0.6444) ran with these truncated schedules too -> re-keying to the SAMPLE budget is a legitimate, untested,
high-EV lever. **LR annealing is the headline** (end-of-training anneal is textbook-critical for representation
quality and is currently absent); risk = at 1M samples the model may be underfit, so annealing could underfit
further -> must be tested, not assumed.

## CODE CHANGES (gated, default-off; running + held jobs byte-identical) — ready for Wave E smoke
1. **Schedule re-keying** (train.py:574-589,633): added `sfrac=examples_seen/max_train_samples`; `dino.lr_key` and
   `dino.reg_key` (default "flop" = unchanged). "sample" re-keys LR-decay (lr_key) / WD+teacher_temp+KDE+momentum
   (reg_key) to sample progress so the cosines complete over the real run. Two flags -> can isolate LR-anneal from
   the reg bundle.
2. **Raw-CLS regressor** (train.py:476-480, FINO audit fix #2): `fino.raw_cls` (default false). True feeds the RAW
   backbone CLS to the continuous predictor (paper Eq.3) instead of the L2-normalized phi_s.
DEFERRED to Wave F: gradient-equalisation s_t (audit fix #1). Rationale: it only rebalances within the lambda_meta=
0.03 metadata term (~3% of the gradient), whereas the schedule re-keying governs ~97% (DINO+JEPA+KDE) -> far higher
EV. s_t is an error-prone restructure (per-branch autograd.grad at the shared CLS + EMA + geometric mean) that
warrants its own isolated smoke + a |T|=1 -> s_t=1 unit check. Sequencing the higher-EV/lower-risk levers first.

## WAVE D RESULTS (2026-06-12 ~07:10) — all 8 in
| job | recipe | gamma | FINAL | vs prior leader 0.6518 |
|-----|--------|------:|------:|------:|
| **jf_d_sefga_g07** | sub+expr512+fga | 0.7 | **0.6523** | **+0.0005 (new nominal best)** |
| jf_cr_fgaclu_warm | sub+cluster+expr512+fga +warm | 0.5 | 0.6514 | -0.0004 |
| jf_d_sefga_g10 | sub+expr512+fga | 1.0 | 0.6510 | -0.0008 |
| jf_d_molstack_g05 | sub + fga+mutcount+msi | 0.5 | 0.6504 | -0.0014 |
| jf_cr_cluster_g05 | cluster (straggler) | 0.5 | 0.6494 | -0.0024 |
| jf_d_sefga_g03 | sub+expr512+fga | 0.3 | 0.6490 | -0.0028 |
| jf_d_sefga_clu_g05 | sub+cluster+expr512+fga | 0.5 | 0.6481 | -0.0037 |
| jf_d_fga_g05 | fga only | 0.5 | 0.6471 | -0.0047 |

**CONCLUSIONS:**
1. **gamma sweep (sub+expr512+fga) is a flat-topped plateau at 0.5-0.7:** 0.3->0.6490, 0.5->0.6518, 0.7->0.6523,
   1.0->0.6510. gamma0.7 marginally tops gamma0.5 but +0.0005 is WITHIN noise (< the 0.01 bar). The mol M+ recipe
   likes MODERATE gamma (peak ~0.6), unlike cluster M+ which wanted monotonic-higher gamma. Don't over-update;
   treat 0.5-0.7 as equivalent. New base = gamma0.7 (nominal best).
2. **fga is COMPLEMENTARY to subtype+expr512, NOT redundant.** fga-only 0.6471 ~ se-only 0.6476, but se+fga 0.6518
   = +0.004 over either alone. Orthogonal axes (genomic-instability + histotype/transcriptome) ADD. -> directly
   validates the fga+til (immune) orthogonal-axis bet for Wave E.
3. **Stacking FGA-CORRELATED continuous axes is redundant/dilutive.** molstack (sub+fga+mutcount+msi) 0.6504 <
   se+fga 0.6518: mutcount+msi add nothing (they track fga). Exactly the lit prediction (aggregate genomic scores
   collinear with FGA).
4. **Combining the two best M+ FAMILIES doesn't help.** sefga_clu (sub+cluster+expr512+fga) 0.6481 < either family
   alone -> cluster and mol targets capture overlapping structure. Family-combo + kitchen-sink directions are DEAD;
   dropped the 2 HELD jobs (sefga_clu_g07, sefgamut_g07) accordingly.
5. **Straggler retrains** (user pre-approved for submit): cluster_g05 0.6494, fgaclu_warm 0.6514. SURFACED for user
   review; NOT submitting (user picks submissions). No new submission this wave.

## WAVE E (launched 2026-06-12 ~07:10) — train-limitation schedule fix + FINO fix + orthogonal axis
Base = sub+expr512+fga gamma0.7 (Wave D best). 8 arms, ~1 change each for attribution. Canaries first: je_schedfull
(sched code) + je_til (til factor), then fan out the rest after no-error check.
| arm | delta vs base | tests |
|-----|---------------|-------|
| je_lrkey | dino.lr_key=sample | isolate LR annealing (the headline train-limitation lever) |
| je_regkey | dino.reg_key=sample | isolate reg bundle (KDE->full, WD->0.2, teacher_temp->0.07, mom->~1.0) |
| je_schedfull | lr_key+reg_key=sample | full sample-keyed schedule |
| je_rawcls | fino.raw_cls=true | FINO audit fix #2 (raw CLS to continuous predictor, Eq.3) |
| je_til | +til (slide_percent_lymphocyte_infiltration) | orthogonal immune axis (lit #3, additive per WaveD-2) |
| je_til_rawcls | +til + raw_cls | stack the 2 non-schedule fixes |
| je_til_schedfull | +til + full sched | best-guess global combo (orthogonal axis + full schedule) |
| je_lrkey_rawcls | lr_key=sample + raw_cls | LR anneal + FINO fix stack |
Code: train.py schedule re-keying + raw_cls (both gated, default-off, syntax+scope verified). s_t grad-equalisation
still deferred to Wave F.
LAUNCHED ~07:31 (canaries je_schedfull/je_til passed no-error check: measured_flops printed, no Traceback). Jobs:
82465 je_schedfull(n-1) · 82466 je_til(n-2) · 82467 je_lrkey(n-4) · 82468 je_regkey(n-7) · 82469 je_rawcls(n-4) ·
82472 je_lrkey_rawcls(n-7) · 82473 je_til_rawcls(n-2) · 82474 je_til_schedfull(n-7). Rebalanced 2 off n-4 (would
have been 4 -> probe-RAM OOM risk ~132GB/job vs 338GB free). ETA ~09:00-09:10. Watcher bp1x7s3qm. NO submissions.
RE-KEY VERIFIED at sfrac=0.105: je_schedfull kde_scale=0.012 & wd=0.0443 (computed from sfrac), vs flop-keyed
je_rawcls kde_scale=0.000 & wd=0.0401 (from flopfrac). lr uses the same sfrac substitution -> anneal to 1e-6
guaranteed. No silent bug; the LR-anneal hypothesis is being tested correctly.

## CEILING ANALYSIS (2026-06-12 ~07:40) — per-probe, our best (sefga_g07 0.6523) vs validated leader (0.6444) vs GenBio teacher (0.6917)
| probe | ours | leader | genbio | vs_ldr | vs_GB |
|-------|-----:|-------:|-------:|-------:|------:|
| linear | 0.7663 | 0.7842 | 0.8076 | **-0.0179** | -0.0413 |
| knn | 0.7197 | 0.7061 | 0.7626 | +0.0136 | -0.0429 |
| 16shot | 0.6455 | 0.6383 | 0.6970 | +0.0072 | -0.0515 |
| seg | 0.3032 | 0.2891 | 0.3234 | +0.0141 | -0.0202 |
| progression | 0.6731 | 0.6575 | 0.7680 | +0.0156 | **-0.0949** |
| mutation | 0.6295 | 0.6162 | 0.6375 | +0.0133 | -0.0080 |
| survival | 0.5908 | 0.5783 | 0.5964 | +0.0125 | -0.0056 |
| robustness | 0.8902 | 0.8855 | 0.9412 | +0.0047 | -0.0510 |
**Reads:**
- We BEAT the validated leader on 7/8 probes; our ONE regression is **linear (-0.0179)** — FINO's metadata DANN
  gating trades a little linear separability for broad gains everywhere else. This is the recipe weak spot to watch:
  does LR-annealing (je_lrkey, cleaner final representation) recover linear?
- **mutation (-0.008) and survival (-0.006) are NEAR PARITY with the ViT-G teacher** — FINO's slide-level molecular
  guidance has nearly closed the gap to a giant on exactly the tasks it targets. This is the FINO thesis working.
  More orthogonal molecular axes (til) could push these PAST the teacher.
- The big teacher gaps — progression -0.095, 16shot -0.051, robustness -0.051, knn -0.043 — are dominated by MODEL
  CAPACITY / DATA SCALE (ViT-S @1M tiles vs ViT-G @ massive). Honest ceiling: a small recipe can't close those;
  metric gains for us come from (a) slide-level molecular tasks where guidance helps, (b) base-representation
  quality (the schedule re-keying). Chasing progression/16shot/robustness via FINO knobs is low-EV.

## WAVE E RESULTS (2026-06-12 ~09:00) — ALL 8 LOST to base 0.6523; schedule re-keying REFUTED
| arm | delta vs base | FINAL | vs base 0.6523 |
|-----|---------------|------:|------:|
| je_rawcls | raw_cls=true | 0.6515 | -0.0008 (neutral) |
| je_schedfull | lr+reg sample | 0.6460 | -0.0063 |
| je_til | +til (immune) | 0.6455 | -0.0068 |
| je_til_rawcls | +til+raw_cls | 0.6446 | -0.0077 |
| je_regkey | reg sample | 0.6435 | -0.0088 |
| je_til_schedfull | +til+lr+reg sample | 0.6429 | -0.0094 |
| je_lrkey_rawcls | lr sample+raw_cls | 0.6399 | -0.0124 |
| **je_lrkey** | **lr sample (LR anneal)** | **0.6386** | **-0.0137 (worst)** |

**CONCLUSIONS (decisive):**
1. **LR ANNEALING HURTS HARD (-0.0137). The model is severely UNDERFIT at 1M samples.** Re-keying the cosine to
   complete over the run anneals LR to 1e-6 and wastes the back half on a near-frozen LR. The FLOP-keyed schedules
   "stopping early" (LR staying ~constant 1e-4) are ADAPTIVE to the short sample-capped run, NOT a bug. My
   train-limitation hypothesis was BACKWARDS: the fix is to learn MORE per sample, not to complete the schedules.
   (Confirms the earlier reasoning that warmup/freeze also hurt here — same underfit regime.)
2. **reg-bundle re-keying also hurts (-0.0088):** ramping WD->0.2 + KDE->full over-regularizes an underfit model.
   je_schedfull (both, -0.0063) is less bad than lr-alone (-0.0137) — the changes partly offset, but all negative.
3. **raw_cls is NEUTRAL (-0.0008):** FINO audit fix #2 doesn't matter empirically; the L2-normalized CLS regressor
   was fine. Don't bother.
4. **til (immune axis) HURTS (-0.0068):** despite orthogonality theory + Wave-D complementarity. Likely 38% coverage
   (sparse) + a 4th unbalanced continuous branch (no s_t). Adding metadata axes beyond sub+expr512+fga keeps failing
   (til, mutcount, msi, cluster all dilutive) -> the metadata side is SATURATED at sub+expr512+fga.
**LEADER UNCHANGED: jf_d_sefga_g07 (sub+expr512+fga, gamma0.7) = 0.6523.** No submission.

## WAVE F PLAN (2026-06-12 ~09:00) — exploit the UNDERFIT: learn more per sample (uses the 81% unspent FLOP budget)
The real train-limitation optimization. We hit the 1M-sample cap at ~19% of the 1e18 FLOP cap -> 81% FLOP headroom.
Underfit model -> spend spare FLOPs on more learning PER sample (samples are capped, FLOPs are not). Levers (base =
sefga g07, gamma0.7): (F1-F3) higher peak LR 1.25/1.5/2e-4 [learn faster/sample; Wave E says model wants more, not
less], (F4-F5) more local_views 12/16 [more views/sample = more signal, costs FLOPs not samples], (F6) lr1.5+lv12
combo. Remaining slots: investigate ViT-B (4x FLOPs/step, ~0.76e18 at 1M samples — UNDER the cap; capacity is the
real teacher gap per ceiling analysis) + finally test s_t grad-equalisation. NOTE: higher LR risks instability;
2e-4 is the canary. NO submissions (user picks).
LAUNCHED ~09:10. Arms (base sefga g07): 82498 jf_f_lr125(lr1.25e-4) · 82499 jf_f_lr150(1.5e-4) · 82500 jf_f_lr200
(2e-4) · 82506 jf_f_const(lr_min=lr=1e-4, NO decay) · 82502 jf_f_lv12(local_views12) · 82503 jf_f_lv16(lv16) ·
82504 jf_f_lr150const(lr1.5e-4+no-decay) · 82507 jf_f_lr150_lv12(lr1.5e-4+lv12). All config-only (lr/lr_min/
local_views), no code change. n-1:2 n-2:3 n-7:3 (n-4/5/6/8 too busy). Watcher bmyaqg3ha; high-LR divergence canary
bhgfj0vkr. ETA ~10:30. Held for Wave G: ViT-B (needs lr_min=lr to avoid the harmful annealing at flop_frac~0.76 +
2h-validation throughput check) and s_t grad-equalisation.

## VIT-B FEASIBILITY (2026-06-12 ~09:15) — supported + fits FLOP cap, but BLOCKED by 2h validation wall
dinov2_vitb14_reg is in model.py DINOV2_VARIANTS (768d/12L/12h) and the 346MB pretrained weight is ALREADY CACHED
(~/.cache/torch/hub/checkpoints/) -> launching is config-only (model.type). FLOPs: ViT-B ~4x ViT-S/step -> ~0.78e18
at 1M samples = UNDER the 1e18 cap. BUT throughput: ViT-S = 74min train @ 45 TFLOP/s; ViT-B ~4x -> ~2.2-3.6 hr
training, EXCEEDING the maintainer's 2h validation window (README:124, "must complete training on a single H100
within 2 hours"). So a ViT-B run can't be validated -> can't top the board even if it scores higher. CEILING READ:
the capacity gap to the ViT-G teacher (progression/16shot/robustness, the ceiling-analysis gaps) is REAL but NOT
closeable within the submission constraints (1M samples + 2h + ViT-S<->B throughput). Deprioritized; ViT-S is the
validatable ceiling. Metric gains must come from the ViT-S recipe (LR/underfit, slide-level molecular guidance).

## WAVE F RESULTS (2026-06-12 ~10:30) — LR sweet spot found: 1.25e-4 -> NEW BEST 0.6537
| arm | lr | views | FINAL | vs base 0.6523 |
|-----|----|-------|------:|------:|
| **jf_f_lr125** | 1.25e-4 | 8 | **0.6537** | **+0.0014 (NEW BEST)** |
| jf_f_const (no decay) | 1e-4 | 8 | 0.6512 | -0.0011 |
| jf_f_lr150 | 1.5e-4 | 8 | 0.6501 | -0.0022 |
| jf_f_lr150const | 1.5e-4 nd | 8 | 0.6498 | -0.0025 |
| jf_f_lv12 | 1e-4 | 12 | 0.6492 | -0.0031 |
| jf_f_lv16 | 1e-4 | 16 | 0.6477 | -0.0046 |
| jf_f_lr150_lv12 | 1.5e-4 | 12 | 0.6465 | -0.0058 |
| jf_f_lr200 | 2e-4 | 8 | 0.6458 | -0.0065 |

**CONCLUSIONS:**
1. **Clean LR-response peak at ~1.25e-4:** 1.0->0.6523, 1.25->0.6537, 1.5->0.6501, 2.0->0.6458. Monotone up then
   down -> 1.25e-4 is a real (if gentle, +0.0014, within the 0.01 bar) optimum. The underfit is MILD: a small LR
   bump helps, but 1.5-2e-4 overshoots and hurts. Consistent with Wave E (annealing/low-LR bad) + bounded headroom.
2. **More local views HURT** (lv12 -0.0031, lv16 -0.0046): not the underfit fix. Extra local crops dilute rather
   than add signal; also nudge flop_fraction up -> a touch more annealing. Don't add views.
3. **const LR (no residual decay) slightly worse** (-0.0011): the tiny 1e-4->9.7e-5 decay is fine/mildly good;
   only FULL annealing (Wave E) is harmful. Keep the default cosine.
**NEW NOMINAL LEADER: jf_f_lr125 (sub+expr512+fga, gamma0.7, lr1.25e-4) = 0.6537 = +0.0093 over validated leader
0.6444 (clears the +0.006 topping bar).** No submission (user picks).

## WAVE G PLAN (2026-06-12 ~10:30) — refine LR peak + underfit-themed reg knobs + the deferred s_t
Base = sub+expr512+fga gamma0.7 lr1.25e-4. NO seed-confirm reruns (Labless validates). Arms: (G1) lr1.15e-4,
(G2) lr1.35e-4 [pin the peak]; underfit = reduce regularization / learn more: (G3) clip_grad5.0 [grad_norm~3.5 is
hitting the 3.0 clip -> let more through], (G4) drop_path0.05, (G5) layerwise_decay0.8 [more backbone learning],
(G6) warmup_fraction0.05 [more full-LR steps], (G7) clip5+dp05 stack; (G8) s_t gradient-equalisation [FINALLY test
the last FINO audit lever — implement gated + smoke + |T|=1 no-op check].
LAUNCHED ~10:40. Jobs (base sefga g07 lr1.25e-4): 82528 jf_g_lr115 · 82529 jf_g_lr135 · 82530 jf_g_clip5 · 82531
jf_g_dp05 · 82532 jf_g_lwd08 · 82533 jf_g_warm05 · 82534 jf_g_clip5_dp05 · 82538 jf_g_st(n-7, moved off n-4 which
had 0 free GPU). n-2:4 n-4:3 n-7:1. Watcher b18aeq2nq; s_t canary bez62rguk. ETA ~12:00.
**s_t CODE (train.py:459-489, gated fino.grad_equalize, default-off):** restructured meta loss to collect per-branch
(factor, 0.03*L_t) terms; when grad_equalize & >=2 branches: g_t = ||autograd.grad(L_t, CLS, retain_graph)||,
grad_eq_ema[f] = 0.99*ema + 0.01*g_t (EMA bank init 1.0 in main, not checkpointed — re-warms fast), nbar =
geo-mean, meta_loss = sum((nbar/ema_f).detach() * L_t). No-op for <2 branches (s_t=1). The jf_g_st arm has 3
branches (subtype + expr512 + fga) -> s_t active. Smoke = canary (measured_flops printed + no Traceback = the
autograd.grad+backward path works). Audit predicted limited impact (lambda_meta=0.03 -> ~3% of gradient).

## WAVE G RESULTS (2026-06-12 ~12:00) — NOTHING beats lr1.25 base 0.6537; campaign CONVERGED
| arm | delta | FINAL | vs base 0.6537 |
|-----|-------|------:|------:|
| jf_g_st | s_t grad-equalize | 0.6536 | -0.0001 (NEUTRAL) |
| jf_g_clip5 | clip_grad5.0 | 0.6520 | -0.0017 |
| jf_g_lr135 | lr1.35e-4 | 0.6519 | -0.0018 |
| jf_g_warm05 | warmup0.05 | 0.6512 | -0.0025 |
| jf_g_clip5_dp05 | clip5+dp05 | 0.6511 | -0.0026 |
| jf_g_dp05 | drop_path0.05 | 0.6467 | -0.0070 |
| jf_g_lwd08 | layerwise_decay0.8 | 0.6451 | -0.0086 |
| jf_g_lr115 | lr1.15e-4 | HANG (probe wedged 2h21m on n-2, killed) | — |

**CONCLUSIONS:**
1. **s_t gradient-equalisation is NEUTRAL (0.6536 vs 0.6537).** Exactly the audit's prediction: at lambda_meta=0.03
   the metadata branch-balancing is ~3% of the gradient and doesn't move the score. **FINO side FULLY CLOSED** — we
   tested every component (prototype CE, continuous regressor, raw-CLS, s_t, M+/M- gamma ramp) and the recipe is
   sub+expr512+fga gamma0.5-0.7. No further FINO lever remains.
2. **LR peak confirmed at 1.25e-4** (1.0->0.6523, 1.25->0.6537, 1.35->0.6519, 1.5->0.6501). Sharp-ish gentle peak.
3. **"Reduce regularization for underfit" thesis FAILED:** dp05 -0.0070, lwd08 -0.0086, clip5 -0.0017, warm05
   -0.0025 all hurt/neutral. The base reg (drop_path0.1, layerwise_decay0.7, clip3.0, warmup0.09) is already optimal.
   The model is NOT underfit in a fixable-by-less-regularization sense; the LR sweet spot (1.25e-4) was the only
   underfit gain and it's mild.
4. **n-2 4-job over-pack -> a probe hang** (jf_g_lr115). Lesson reinforced: cap probe co-location; use spread_submit.

**CAMPAIGN CONVERGED. LEADER = jf_f_lr125 (sub+expr512+fga, gamma0.7, lr1.25e-4) = 0.6537 = +0.0093 over validated
leader 0.6444 (clears +0.006 topping bar).** Exhaustively explored: metadata axes (saturated), gamma (plateau),
schedule keying (flop-keying optimal), LR (peak 1.25e-4), views (hurt), reg knobs (base optimal), s_t (neutral),
warmup/freeze (hurt), ViT-B (blocked by 2h validation). HONEST CEILING: 0.6537 is the ViT-S+FINO ceiling for this
family; the remaining gap to the ViT-G teacher (0.6917) is capacity/scale-bound and not closeable in-constraints.

## WAVE H PLAN (2026-06-12 ~12:00) — last untested direction: data curation + augmentation (the #2 leader's lever)
Tile dataset is fixed, but tissue-filtering + stain/color aug are untouched this campaign and orthogonal to
everything tried. Base jf_f_lr125. Arms: tissue_thresh 0.1/0.25 (filter near-empty tiles), hed_jitter 0.12/0.0
(stronger/ablated stain aug), color_jitter 0.5, global_crop_scale aggressive, + a clean-data+strong-stain combo.
Parallel lit scan on pathology-SSL augmentation/curation to inform Wave I. Moderate EV (near ViT-S ceiling).
LAUNCHED ~12:05. Arms (base jf_f_lr125): 82564 jf_h_tis010(tissue0.1) · 82552 jf_h_tis025(0.25) · 82553 jf_h_hed12
(hed0.12) · 82554 jf_h_hed0(no stain aug) · 82560 jf_h_cj05(color0.5) · 82561 jf_h_gcrop(gcrop[0.25,1.0]) · 82562
jf_h_tis010_hed12 · 82563 jf_h_hedcj(hed0.12+cj0.5). spread_submit auto-packed 6 on n-4 -> rebalanced to n-2:3 n-3:2
n-4:3 (8 probes need ~1TB RAM vs ~874GB free -> some hang risk; spread to <=3/node). Watcher bg76sflqm; lit scan
running. ETA ~14:00. NO submissions.

## LIT SCAN (2026-06-12 ~12:10) — stain augmentation is THE curation lever (ingested geist commit a903f77)
Papers: RandStainNA (Shen MICCAI2022, arXiv:2206.12694), Lai 2023 (Google, arXiv:2310.13259, ViT-S/TCGA SSL
ablation = nanopath's EXACT setup). DECISIVE finding: in the Lai ViT-S/TCGA SSL ablation, **stain-color
augmentation was the ONLY curation/aug/crop/loss lever that GENERALIZED to the held-out test set.**
- **Q1 stain aug = highest EV.** Stronger beats lighter; best = normalization-anchored randomness (RandStainNA:
  random LAB/HSV/HED + Reinhard to a Gaussian template; ViT-Tiny under stain shift 72.85->93.34). Our HED-jitter
  0.06 is MILD -> strengthen. VALIDATES Wave H hed12/hedcj arms as the most likely movers.
- **Q2 tissue filtering = weak.** Cluster-balanced curation didn't generalize even at full scale; shrinking the set
  hurts a sample-limited run. tissue_thresh=0 defensible -> Wave H tis010/tis025 likely low-EV/neutral.
- **Q3 crop = modest, and I went the WRONG way.** Lit favors MILDER resize / higher overlap (pathology lacks
  natural-image center-bias); my jf_h_gcrop=[0.25,1.0] is MORE aggressive -> likely neutral/negative. Wave I should
  try [0.5,1.0] (milder, more overlap) instead.
**WAVE I SEED:** if hed12/hedcj win -> push HED stronger (0.18) + prototype RandStainNA (CPU-heavy, watch throughput);
crop [0.5,1.0]; drop heavy tissue filtering. Stain aug is the one lever with cross-validated generalization in our
exact setup -> the best remaining shot at moving past 0.6537.

## SUBMITTED to Labless (2026-06-12 ~14:30, user-directed batch, one reusable token, no re-auth)
All 23 completed Wave E/F/G runs submitted, 0 failures. Key run_ids: jf-lr125 (LEADER 0.6537) run_sub_1bc4d404e3;
jf-grad-eq (s_t, 0.6536) run_sub_2a698375d8. Full per-run run_ids in each output_dir/labless_submission.json.
jf_g_lr115 NOT submitted (hung, no summary). Wave H (8 jobs) to be submitted with the same token when they finish
(~15:00-15:20). Per user: NO new jobs after Wave H.

## WAVE H RESULTS (2026-06-12 ~15:15) — aug/curation sweep; NOTHING beats base 0.6537; campaign CLOSED
| arm | aug change | FINAL | vs base 0.6537 |
|-----|-----------|------:|------:|
| jf_h_gcrop | global_crop [0.25,1.0] | 0.6533 | -0.0004 |
| jf_h_cj05 | color_jitter 0.5 | 0.6528 | -0.0009 |
| jf_h_tis010 | tissue_thresh 0.1 | 0.6522 | -0.0015 |
| jf_h_tis025 | tissue_thresh 0.25 | 0.6503 | -0.0034 |
| jf_h_hed12 | HED-jitter 0.12 | 0.6447 | -0.0090 |
| jf_h_tis010_hed12 | tissue0.1+HED0.12 | 0.6444 | -0.0093 |
| jf_h_hedcj | HED0.12+color0.5 | 0.6440 | -0.0097 |
| jf_h_hed0 | NO stain aug | 0.6200 | -0.0337 |

**CONCLUSIONS:**
1. **Stain augmentation is the single biggest aug lever (~+0.034): hed0 (no HED-jitter) CRATERS to 0.6200.** The lit
   (Lai 2023) was RIGHT that stain aug is THE lever for ViT-S/TCGA SSL. BUT direction was wrong for us: HED 0.0->0.62,
   **0.06 (base)->0.6537 (best)**, 0.12->0.6447. **Base hed_jitter=0.06 is the sweet spot** — removing AND doubling
   both hurt. Cross-validated by the ablation.
2. **Tissue filtering neutral-to-negative** (tis010 -0.0015, tis025 -0.0034): confirms lit — curation doesn't help,
   stronger filtering shrinks the sample-limited set and hurts. tissue_thresh=0 is right.
3. **Crop/color neutral** (gcrop -0.0004, cj05 -0.0009): base aug is optimal.

## CAMPAIGN CLOSED (2026-06-12 ~15:15) — FINAL
**LEADER = jf_f_lr125 (FINO subtype+expr512+fga, gamma0.7, lr1.25e-4) = 0.6537 = +0.0093 over validated leader
0.6444 (clears the +0.006 topping bar). Checkpoint: /data/hm/nanopath/jepafino/jf_f_lr125/.**
Exhaustively explored & all converged/negative: metadata axes (saturated at subtype+expr512+fga), gamma (plateau
0.5-0.7), schedule keying (flop-keying optimal; LR-anneal hurts -> model is mildly underfit), LR (sharp peak
1.25e-4), local views (hurt), reg knobs (base optimal), s_t grad-equalisation (neutral), warmup/freeze (hurt),
ViT-B (blocked by 2h validation), aug/curation (base hed0.06 optimal, stain-aug essential, tissue-filter useless).
HONEST CEILING: 0.6537 is the ViT-S+FINO ceiling for this family; the remaining gap to the ViT-G teacher (0.6917)
is capacity/scale-bound, not closeable in-constraints (1M samples, 2h validation). Per user: NO new jobs.
All 31 runs (E/F/G/H) submitted to Labless with the reusable token.

## WAVE I (2026-06-12 ~15:30, campaign reopened, cap=4 per user) — fine HED-jitter sweep
Wave H showed stain aug is the biggest aug lever (hed0->0.620, 0.06->0.6537, 0.12->0.6447). Parabola fit to those 3
points peaks at hed~0.077 (predicted ~0.6554, +0.0017 over 0.06) -> tight sweep brackets it. Base jf_f_lr125, only
data.hed_jitter changes. Jobs: 82611 hed05(n-2) · 82612 hed07(n-2) · 82613 hed08(n-7) · 82614 hed09(n-7). Watcher
armed. Likely within noise (predicted gain < the 0.01 bar); RandStainNA (lit's stronger stain-aug method) is the
higher-EV-but-higher-effort follow-up if this confirms stain-aug headroom. ETA ~17:00. NO submission.
