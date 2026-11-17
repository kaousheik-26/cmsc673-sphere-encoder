# Follow-up experiments

Adaptive latent edits, direct probes, binary sky-mask controls, and measured-depth evaluation.

- [Findings](FOLLOWUP_FINDINGS.md)
- [Protocol](FOLLOWUP_PROTOCOL.md)
- [Metrics](FOLLOWUP_METRICS.csv)
- [Original run environment and source hashes](FOLLOWUP_ENVIRONMENT.json)

Run commands from the repository root with `python -m experiments.followup.<module> --help`.
The Slurm launcher uses the same package. Shared geometry, model-loading, and metric utilities
remain in `experiments.intrinsics`.

The environment manifest records the original execution sources before this folder move;
its hashes are historical provenance, not hashes of the reorganized files.
Existing generated outputs remain in `experiments/intrinsics/outputs/` and are ignored by Git.

Run follow-up tests with `python -m pytest experiments/followup`.
