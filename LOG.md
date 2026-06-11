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
