# Next Steps

## Deferred TCGA Sample-List Revision

The current txt-based TCGA loader is acceptable for now and should work.

Current state:

- the training configs read `/block/TCGA/sample_dataset_30.txt`
- each row is currently interpreted as `slide_path x y level`
- the effective patch MPP is fixed by the listed pyramid level for that row
- this means the loader can see different MPPs across rows, but it does **not** currently sample a fresh MPP uniformly in `[0.25, 0.6]` on every access

## What We Want Later

The desired future behavior is:

- all TCGA sample-list rows should correspond to one common base patch definition at `0.5` MPP
- training should then randomly resize on the fly to a sampled target MPP in `[0.25, 0.6]`
- this should keep one clean source of truth for TCGA patch centers while restoring random physical-scale variation during training

## Important Correction

The future txt file should **not** simply be:

- `slide_path center_x center_y 0.5`

That would be insufficient because the loader needs the slide's native root MPP in order to compute the correct level-0 crop size for a sampled target MPP.

The correct future row format is:

- `slide_path center_x center_y native_mpp`

This matches the logic used in the OpenMidnight random-MPP txt plan:

- sample `sampled_mpp` uniformly in `[0.25, 0.6]`
- compute `level0_size = round(224 * sampled_mpp / native_mpp)`
- read that crop from level 0 around the stored center
- resize back to `224x224`

## Why We Should Not Just Rewrite The Current File

It would be wrong to take the current `/block/TCGA/sample_dataset_30.txt` rows and only replace the final token with `0.5`.

Reason:

- the current file was generated as `slide_path x y level`
- those coordinates came from mixed pyramid levels
- so the rows do **not** all represent centers of true `0.5`-MPP patches

Therefore the correct future artifact must be a newly generated file:

- `/block/TCGA/sample_dataset_30_point5mpp.txt`

## Future Implementation Plan

1. Write a one-time builder that scans the current TCGA slides and generates `sample_dataset_30_point5mpp.txt`.
2. Define each output row as `slide_path center_x center_y native_mpp`.
3. Ensure each stored center corresponds to a valid `224x224` patch at `0.5` MPP.
4. Update this repo's loader to detect the `native_mpp` row format.
5. Sample target MPP uniformly in `[0.25, 0.6]` at runtime.
6. Read the corresponding level-0 crop and resize back to `224x224`.
7. Keep the existing JEPA global/local/latent augmentation path unchanged after patch extraction.
8. Bump `project.recipe_id` again once this new TCGA contract is active.

## Acceptance Criteria For That Later Change

- the new txt file exists at `/block/TCGA/sample_dataset_30_point5mpp.txt`
- the loader reads `path center_x center_y native_mpp`
- sampled MPP is random in `[0.25, 0.6]` during training
- validation remains deterministic per index
- the current `sample_dataset_30.txt` path can be retired from the checked-in configs once the new file is ready
