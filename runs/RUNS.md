# Runs

One numbered run per submitted job setup. Numbers count up and are never
reused. Resubmitting the same script to finish a timed-out job keeps the same
number; changing code or settings gets a new one. Each run lives in
`runs/runN/` with its sbatch script, SLURM logs, and outputs.

Modes (`experiments.py --mode`): `rollout` (success rate only), `lens`
(rollouts + logit lens + hidden states), `probe` (ridge probes, CPU).
Before 2026-09-29 these were `--run A / B / C`.

| run | date | job id(s) | mode | scope | code | outcome |
|---|---|---|---|---|---|---|
| run0 | 2026-09-24 | 54679852 (failed: bad `env.sh` path), 54679900 | rollout, then lens, both `--dry_run` | libero_spatial task 0, init 0, 20 steps | experiments.py sha256 `4fd592d9…` (untracked; old `--run A/B` flags) | Pipeline check passed on V100S-32GB: lens matched the generated token on 140/140 action tokens. Episode "fail" only because capped at 20 steps. No success rate. |
| run1 | 2026-09-29 | 55215295 | lens | libero_spatial tasks 0-9 × inits 0-4 (50 episodes), with videos | experiments.py sha256 `db1bd915…` (before the tie-break fix) | **Failed** on task 0 init 0, step 0 (queued 7 h, ran 1.5 min). Exact fp16 tie between roll buckets 152 and 154: generate() picked 154 (first in token-id order), the lens picked 152 (first in bucket order), so the final-layer check stopped the run. No data saved. sacct showed COMPLETED because the script's last command succeeded. |
| run2 | 2026-09-30 | | lens | same as run1 | tie-break fix in `lens_for_step` (lens breaks ties in token-id order, like generate); sbatch now exits with Python's exit code | |
