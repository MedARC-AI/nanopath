Project goals:
- Nanopath should be easy to share with collaborators so they can quickly try new training objectives, preprocessing choices, data curation ideas, and hyperparameters on a small model.
- The normal loop is: iterate fast on a single H100, validate promising changes with downstream probes, then reserve larger/full-node training for the best candidates.
- Keep the codebase hackable and nanochat-like: flat organization, few files, few lines, and minimal abstractions.

Before changing code:
- For broad or ambiguous tasks, read deeply enough into the current repo to understand the training/probing/data path before recommending changes. Look at every relevant source, config, script, and doc file rather than optimizing one file in isolation.
- Make a concrete multi-step plan for nontrivial work, then keep going through implementation, validation, and any needed doc/comment updates.
- Default to immediately implementing sensible recommendations and validating them rather than simply suggesting recommendations.

Coding guidelines:
- Use flat organization: as few folders, files, and lines of code as possible; functional elegance is the goal. If you make revisions where there are over a dozen new lines of code I am going to be highly skeptical you really tried your best to adhere to this guideline. A great revision should LOWER the total lines of code, not increase it. Don't play smart by opening subprocesses or other hacks to get around this limitation.
- Commenting is the exception to the line-count preference: add concise comments explaining "how and why" for functions and important (i.e., not plotting/logging) code blocks.
- Do not add defensive `try`/`except` blocks or fallbacks. If something is wrong, it should fail loudly. Don't bother with ValueError raises or other code guards.
- Prefer hard-coded constants over extra environment variables, modular options, or fallback paths, unless the value is meant to be frequently tuned.
- Prefer native PyTorch over Accelerate, Lightning, etc. Multi-GPU should use PyTorch DDP.
- Do not use `argparse`. Meaningful tunables should live in YAML config files, e.g. `cfg.train.lr`; if YAML does not define a variable used by a training script, it is fine for that to error. Only put variables in YAML when they are actually meant to be tuned often; otherwise hard-code them.
- Avoid tiny helper functions/classes that are only a handful of lines. Put the code directly where it is used.
- Follow [nanochat](https://github.com/karpathy/nanochat) as the model for a clean minimalist codebase, especially `train.py` and `model.py`.
- Do not create new files unless explicitly asked or truly necessary; prefer improving existing files. If you do create a new file, add a few commented out lines of code to the top of it to explain its purpose.
- If code changes make comments, docs, configs, or scripts inaccurate, update those too.

Experiment and benchmark discipline:
- Validate opinions experimentally whenever feasible. Run code, tests, probes, or short jobs that directly support the conclusion.
- Use downstream probing as the main comparison signal because objectives like JEPA, MAE, DINO, and iBOT may not have comparable validation losses.
- An improvement should only actually be considered an improvement mean_probe_score improves over .01, anything less is within random variance.
- Use wandb for logging, plotting, and utilization monitoring throughout pretraining. Log all metrics needed to validate training behavior (i.e., gradient norm).
- Aim for >80% GPU utilization during GPU runs; investigate and remedy code when utilization is poor.

Cluster and storage:
- The login node has no GPU access, and has a different /tmp folder than the compute nodes. For full training runs (or anything that would take more than a few minutes) you should submit SLURM jobs to H100 nodes (`n-#`) or CPU nodes (`c-1`). If it's a quick single-GPU assessment you can use `ssh n-#` directly (but only when an idle GPU is available!).
- Store large files, checkpoints, embeddings, caches, and pretrained models under `/data/$USER/nanopath` (configs use literal `$USER`, expanded by `train.py` at load time), not the repo.
- Treat SLURM wall-clock expiry as a clean finished run: save `latest.pt`, run downstream probes, write `summary.json`, and exit. Other kills/preemptions should keep using checkpoint auto-resume.
- Before launching a fresh run, delete any existing `project.output_dir` or choose a new one; otherwise `train.py` will auto-resume from `latest.pt` there, which is a problem unless resume was intended.
- Do not define `#SBATCH --cpus-per-task` in sbatch scripts and do not add unnecessary lines like `OMP_NUM_THREADS` or `MKL_NUM_THREADS` exports.

Workflow:
- Do not stop after the first small fix on a difficult ask. Continue through the adjacent tasks needed to make the change credible, such as config updates, probes, throughput checks, README notes, or cleanup.
- Use parallel agents, git worktrees, or independent jobs when they materially speed up exploration or experiments, but keep changes easy to review. Make sure to kill hanging or no longer useful sub-agents.
