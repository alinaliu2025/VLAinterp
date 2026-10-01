# Runs

One numbered run per submitted job setup. Numbers count up and are never
reused. Resubmitting the same script to finish a timed-out job keeps the same
number; changing code or settings gets a new one. Each run lives in
`runs/runN/` with its sbatch script, SLURM logs, and outputs.

Modes (`experiments.py --mode`): `rollout` (success rate only), `lens`
(rollouts + logit lens + hidden states), `probe` (ridge probes, CPU),
`patchlens` (`lens` plus a logit lens on the 256 image patches over the text
vocabulary).
Before 2026-09-29 these were `--run A / B / C`.

| run | date | job id(s) | mode | scope | code | outcome |
|---|---|---|---|---|---|---|
| run0 | 2026-09-24 | 54679852 (failed: bad `env.sh` path), 54679900 | rollout, then lens, both `--dry_run` | libero_spatial task 0, init 0, 20 steps | experiments.py sha256 `4fd592d9…` (untracked; old `--run A/B` flags) | Pipeline check passed on V100S-32GB: lens matched the generated token on 140/140 action tokens. Episode "fail" only because capped at 20 steps. No success rate. |
| run1 | 2026-09-29 | 55215295 | lens | libero_spatial tasks 0-9 × inits 0-4 (50 episodes), with videos | experiments.py sha256 `db1bd915…` (before the tie-break fix) | **Failed** on task 0 init 0, step 0 (queued 7 h, ran 1.5 min). Exact fp16 tie between roll buckets 152 and 154: generate() picked 154 (first in token-id order), the lens picked 152 (first in bucket order), so the final-layer check stopped the run. No data saved. sacct showed COMPLETED because the script's last command succeeded. |
| run2 | 2026-09-30 | 55231832 | lens | same as run1 | commit `2de255d`: tie-break fix in `lens_for_step` (lens breaks ties in token-id order, like generate); sbatch now exits with Python's exit code | **Success.** V100S-32GB, ran 13:57–14:40 (42 min). Success rate **42/50 = 84.0%** (OpenVLA reports 84.7%). Final-layer lens matched the generated token on all 45,220 action tokens. 6,460 env steps. Lens: gripper beats its most-common-value baseline (52%) at layer 19 and reaches 98% by layer 28; x/y/z/roll/pitch/yaw stay under 15% until layer 22 and reach 66–85% only at layer 31. Outputs on OSC: hidden/ 1.7 GB, lens/ 14 MB, videos/. Plots + episodes.csv in git. |
| run3 | 2026-10-01 | | patchlens | libero_spatial task 0 × inits 0-4 (5 episodes), with videos | adds `--mode patchlens`: logit lens on the 256 image-patch positions (1..256 in the sequence) over the text vocabulary; top-5 words + rank/log-prob of 22 candidate words per patch, layer and step; model input images; patch hidden states every 20 steps. New check: layer-0 states at the patch positions must equal the projector output. | |
