# OSC Job Commands: Queue, Track, Stop

A reference for running VLAinterp jobs on OSC (Pitzer, Cardinal, Ascend). OSC
uses **Slurm**, the job scheduler: you describe the resources you need (GPU,
memory, time), Slurm puts the job in a queue, and runs it on a compute node when
those resources free up. For the environment itself (`env.sh`, conda, weights),
see [VLAinterp_OSC_documentation.md](VLAinterp_OSC_documentation.md).

Replace `<jobid>` with the number `sbatch` prints (e.g. `55231832`).

---

## 1. Quick reference

| I want to... | Command |
|---|---|
| Queue a batch job | `sbatch runs/runN/runN.sbatch` |
| Queue B to start only if A succeeds | `jid=$(sbatch --parsable A.sbatch); sbatch --dependency=afterok:$jid B.sbatch` |
| Get an interactive GPU shell | `salloc --account=PAS2324 --nodes=1 --gres=gpu:v100-32g:1 --mem=64G --time=00:45:00` |
| See my jobs | `squeue -u $USER` |
| When will my pending job start? | `squeue -u $USER --start` |
| Everything about one job | `scontrol show job <jobid>` |
| Watch a running job's log | `tail -f runs/runN/runN_<jobid>.log` |
| Did a finished job succeed? | `sacct -j <jobid> -X --format=JobID,JobName,State,Elapsed,ExitCode` |
| Cancel one job | `scancel <jobid>` |
| Cancel all my jobs | `scancel -u $USER` |
| Pause a pending job (keep its place) | `scontrol hold <jobid>` / `scontrol release <jobid>` |
| Project hours used | `OSCusage` |
| Disk usage | `quota -s` (home), `du -sh /fs/ess/PAS2324/wiencek.16` (project) |

---

## 2. Two ways to run on a GPU

**Batch (`sbatch`)**: write the commands in a script, submit it, log out. The
job waits in the queue, runs on its own, and writes its output to a log file.
Use this for anything longer than a quick test, or when the queue is long. You
do not need to stay logged in.

**Interactive (`salloc`)**: ask for a node and get a shell on it, then type
commands yourself. Good for debugging, but you wait at the terminal until the
node is granted, and the session dies when time runs out or you disconnect.

```bash
salloc --account=PAS2324 --nodes=1 --gres=gpu:v100-32g:1 --mem=64G --time=00:45:00
# ...wait for "Granted job allocation <jobid>"; the prompt changes to a compute node
cd /fs/ess/PAS2324/wiencek.16/VLAinterp && source my_env.sh
python instruction_swap.py ...
exit          # give the node back as soon as you are done (stops billing)
```

**Never run the model on a login node** (`pitzer-login0x`). Login nodes are
shared, have no usable GPU, and kill heavy processes. `squeue`, `sbatch`,
`git`, editing files, and small CPU scripts are fine there.

---

## 3. Writing a batch script

A batch script is a bash script whose `#SBATCH` lines are read by Slurm before
anything runs. Template (this is what `runs/run4/run4.sbatch` looks like):

```bash
#!/bin/bash -l
#SBATCH --account=PAS2324          # project to bill (required at OSC)
#SBATCH --job-name=vla_run4        # name shown in squeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32g:1      # 1 GPU, the 32 GB V100 kind (Pitzer)
#SBATCH --mem=64G                  # CPU RAM, not GPU memory
#SBATCH --time=01:30:00            # wall-clock limit, HH:MM:SS; job is killed at this time
#SBATCH --mail-type=END,FAIL       # email when it ends or fails
#SBATCH -o runs/run4/run4_%j.log   # stdout+stderr go here; %j = job id
cd "$SLURM_SUBMIT_DIR"             # the folder you ran sbatch from
source ./my_env.sh
python instruction_swap.py ...
status=$?
echo "[job] python exit code ${status}"
exit ${status}                     # so sacct shows FAILED if Python crashed
```

`#!/bin/bash -l` (login shell) makes `module` and conda work inside the job.

### Common `#SBATCH` flags

| Flag | Meaning |
|---|---|
| `--account=PAS2324` | Project to charge. Required. |
| `--job-name=NAME` | Label in `squeue`/`sacct`. Use the run name. |
| `--time=HH:MM:SS` | Time limit. Ask for ~1.5x your estimate: too short and the job is killed (state `TIMEOUT`); much too long and it may wait longer in the queue. `D-HH:MM:SS` for days. |
| `--nodes=1` | Number of machines. Always 1 for this project. |
| `--gres=gpu:v100-32g:1` | Pitzer: one 32 GB V100. Plain `--gres=gpu:1` may give a 16 GB V100, too small for OpenVLA-7B. |
| `--gpus-per-node=1` | Any GPU on that cluster. Fine on Cardinal (H100) and Ascend (A100), but on those the model loads in bf16, not fp16 (not bit-comparable with run2). |
| `--mem=64G` | CPU RAM for the job. `OUT_OF_MEMORY` in `sacct` means raise it. |
| `--cpus-per-task=N` / `--ntasks=N` | CPU cores. The defaults are fine here. |
| `-o PATH` / `-e PATH` | Where stdout / stderr go. With only `-o`, both go to one file (easiest). |
| `--mail-type=BEGIN,END,FAIL` | When to email. `ALL` also works. |
| `--mail-user=you@osu.edu` | Email address, if the default is not the one you want. |
| `--dependency=afterok:<jobid>` | Wait for another job (see section 4). |
| `--begin=now+2hours` or `--begin=2026-10-14T08:00` | Don't start before this time. |

Any `#SBATCH` flag can also be given on the command line, where it overrides the
script: `sbatch --time=00:20:00 runs/run4/run4.sbatch`.

### Filename patterns for `-o`

| Pattern | Becomes |
|---|---|
| `%j` | job id |
| `%x` | job name |
| `%A` / `%a` | array master id / array index (section 4) |

### Variables available inside a job

| Variable | Value |
|---|---|
| `$SLURM_JOB_ID` | this job's id |
| `$SLURM_SUBMIT_DIR` | the directory you ran `sbatch` from |
| `$SLURM_JOB_NODELIST` | the node(s) the job runs on |
| `$SLURM_ARRAY_TASK_ID` | index in a job array |

---

## 4. Queueing

```bash
sbatch runs/run4/run4.sbatch          # prints "Submitted batch job <jobid>"
sbatch --parsable runs/run4/run4.sbatch   # prints only the id (for scripts)
sbatch --test-only runs/run4/run4.sbatch  # checks the script + estimates start; submits nothing
```

### Chaining jobs with dependencies

Queue several jobs now and let Slurm start each one when its condition is met:

```bash
jid=$(sbatch --parsable runs/run4/run4_test.sbatch)
sbatch --dependency=afterok:${jid} runs/run4/run4.sbatch
```

| Condition | Start when the earlier job... |
|---|---|
| `afterok:<id>` | finished with exit code 0 (success) |
| `afternotok:<id>` | failed, timed out, or was cancelled |
| `afterany:<id>` | finished in any way |
| `after:<id>` | started running |
| `singleton` | every earlier job with the same `--job-name` has finished |

Several ids: `afterok:111:222`. If the condition can never be met (the test
failed), the dependent job stays `PENDING` with reason
`DependencyNeverSatisfied` forever. Cancel it with `scancel`.

`afterok` only works if the script exits with Python's exit code (the
`exit ${status}` line). Without it, a script whose last line is `date` always
"succeeds" (this is what happened in run1).

### Job arrays (many similar jobs)

One script, many copies, each with a different `$SLURM_ARRAY_TASK_ID`:

```bash
#SBATCH --array=0-9               # 10 jobs, ids 0..9
#SBATCH --array=0-9%2             # same, at most 2 running at once
#SBATCH -o runs/runN/task_%a_%A.log
python instruction_swap.py --task ${SLURM_ARRAY_TASK_ID} ...
```

`scancel <arrayid>` cancels all of them; `scancel <arrayid>_3` cancels one.

---

## 5. Tracking

### `squeue`: jobs that are pending or running

```bash
squeue -u $USER                   # my jobs
squeue -u $USER --start           # estimated start times of pending jobs
squeue -j <jobid>                 # one job
squeue -u $USER -t PENDING        # only pending (or RUNNING)
squeue -A PAS2324                 # everyone's jobs on the project
watch -n 60 squeue -u $USER       # refresh every 60 s; Ctrl+C to quit
```

Wider, more readable columns:

```bash
squeue -u $USER -o "%.10i %.16j %.9T %.10M %.10l %.6D %R"
#                   id   name   state elapsed limit nodes reason/node
```

A finished job disappears from `squeue` within minutes. Use `sacct` for it.

The **ST** (state) column:

| Code | State | Meaning |
|---|---|---|
| `PD` | PENDING | waiting in the queue |
| `R` | RUNNING | running |
| `CG` | COMPLETING | finishing up |
| `CD` | COMPLETED | exited with code 0 |
| `F` | FAILED | exited with a nonzero code |
| `TO` | TIMEOUT | hit `--time` and was killed |
| `CA` | CANCELLED | `scancel`ed |
| `OOM` | OUT_OF_MEMORY | used more than `--mem` |
| `NF` | NODE_FAIL | the node died (not your fault; resubmit) |

The **NODELIST(REASON)** column, for pending jobs:

| Reason | Meaning |
|---|---|
| `Priority` | other jobs are ahead of you; just wait |
| `Resources` | you are next; waiting for a GPU to free up |
| `Dependency` | waiting for the job it depends on |
| `DependencyNeverSatisfied` | that job failed; this one will never start, so `scancel` it |
| `QOSMaxJobsPerUserLimit`, `QOSMaxGRESPerUser`, ... | you hit a per-user limit; it starts when your other jobs finish |
| `AssocGrp...` / `...Billing...` | a project-level limit (e.g. allocation used up); ask Dr. Zhu |
| `ReqNodeNotAvail` | maintenance is scheduled; the job would not finish before it |

### `scontrol show job`: everything about one job

```bash
scontrol show job <jobid>
```

Shows the full request, the node, `StartTime` (estimated if pending),
`WorkDir`, `StdOut` (where the log goes), and the `Reason`. Only works while the
job is in the queue or shortly after.

### Logs: what the job is printing

```bash
tail -f runs/run4/run4_<jobid>.log      # follow live; Ctrl+C to stop following
tail -n 50 runs/run4/run4_<jobid>.log   # last 50 lines
less runs/run4/run4_<jobid>.log         # scroll; q to quit, G to jump to the end
grep -n "Error\|Traceback" runs/run4/*.log
```

If the log is empty or behind while the job is clearly running, Python is
buffering its output. Print with `flush=True`, or run `python -u ...`, or set
`export PYTHONUNBUFFERED=1` in the script.

### Looking at the GPU of a running job

Open a second shell inside the job's allocation and run `nvidia-smi` (GPU memory
use and utilization):

```bash
srun --jobid=<jobid> --overlap --pty bash
nvidia-smi
exit
```

`ssh <nodename>` (the node from `squeue`) also works while you have a job
running on that node.

### `sacct`: finished jobs (and history)

```bash
sacct -j <jobid> -X --format=JobID,JobName,State,Elapsed,ExitCode
sacct -u $USER -X -S 2026-10-01 --format=JobID,JobName%20,State,Start,Elapsed,ExitCode
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,NodeList
```

- `-X` shows one line per job. Without it you also get the `.batch` and
  `.extern` steps, which is noisy but useful for `MaxRSS`.
- `-S` / `-E`: start and end of the date range. The default is today only.
- `ExitCode` is `code:signal`. `0:0` is success; `1:0` means Python raised an
  error (look at the log); a nonzero signal (e.g. `0:9`) means it was killed.
- `MaxRSS` (with the `.batch` step) is the peak CPU RAM used: compare it with
  `--mem` to see how much you can ask for next time.
- `JobName%20` widens that column to 20 characters.

`seff <jobid>`, if installed, gives a one-screen summary of CPU and memory
efficiency for a finished job.

---

## 6. Stopping and changing jobs

```bash
scancel <jobid>                    # cancel one job (pending or running)
scancel <jobid1> <jobid2>          # several
scancel -u $USER                   # ALL my jobs, on this cluster
scancel -u $USER -t PENDING        # only my pending jobs
scancel -n vla_run4                # by job name
```

Cancelling is immediate and cannot be undone; a running job loses whatever it
had not saved. (`save_npz_atomic` writes to a `.tmp` file and renames it at the
end, so a cancelled run never leaves a half-written `.npz` under the real name.
`instruction_swap.py` saves only once, at the very end, so a cancelled run4
saves nothing.)

```bash
scontrol hold <jobid>              # keep a pending job from starting
scontrol release <jobid>           # let it start again
scontrol update JobId=<jobid> TimeLimit=00:40:00   # change the limit (users can usually only lower it)
scontrol requeue <jobid>           # put a running/finished job back in the queue
```

To fix a mistake in a pending job's script: `scancel` it, fix the file, and
`sbatch` again. Editing the `.sbatch` file after submitting does **not** change
the queued job. Slurm copied the script at submit time. But the job reads your
**Python** files when it starts, so edits to `instruction_swap.py` made while the
job is pending **will** be used. Commit before you submit, and the
`git log` / `sha256sum` lines in the log record which code actually ran.

---

## 7. Cluster and account information

```bash
sinfo -s                           # partitions and how many nodes are busy/idle
sinfo -o "%P %G %D %t" | grep gpu  # GPU types (GRES) per partition
OSCusage                           # core-hours used by the project
quota -s                           # home directory usage
du -sh /fs/ess/PAS2324/wiencek.16/VLAinterp/runs/*   # size of each run's outputs
```

- Each cluster has its own queue. Log in to `pitzer.osc.edu`,
  `cardinal.osc.edu` or `ascend.osc.edu`; jobs submitted there run there, and
  `squeue`/`scancel` there only see that cluster's jobs. The project folder
  `/fs/ess/PAS2324/` is shared by all of them.
- **OSC OnDemand** (https://ondemand.osc.edu) has a web view of the same things:
  *Jobs > Active Jobs* lists and cancels jobs, and *Files* browses and downloads
  plots without `scp`.
- A job's hours are billed for the time it holds the node, not the time your
  code was busy. Exit `salloc` sessions as soon as you are done.

---

## 8. Gotchas

- **The `-o` folder must exist before the job starts.** If `runs/run5/` does not
  exist, Slurm cannot create the log, and the job fails with no log at all.
  `mkdir -p runs/run5` first (keeping the `.sbatch` in that folder does this).
- **Relative paths in `#SBATCH` lines are relative to where you ran `sbatch`.**
  Always submit from the repo root: `cd /fs/ess/PAS2324/wiencek.16/VLAinterp`.
- **Windows line endings.** A script saved on Windows may have CRLF endings, and
  `sbatch` rejects it ("DOS line breaks"). Fix with
  `sed -i 's/\r$//' runs/run4/run4.sbatch` (or `dos2unix`).
- **`COMPLETED` does not mean the code worked**, unless the script ends with
  `exit ${status}`. Always check the end of the log too.
- **Don't write in Alina's folder.** Outputs go in your clone; `my_env.sh` keeps
  the Hugging Face caches out of her folder.

---

## 9. This project's workflow

```bash
# on a login node
cd /fs/ess/PAS2324/wiencek.16/VLAinterp
git pull                                      # get the latest code
jid=$(sbatch --parsable runs/run4/run4_test.sbatch)
sbatch --dependency=afterok:${jid} runs/run4/run4.sbatch
squeue -u $USER                               # both jobs: one PD (Priority), one PD (Dependency)

# later
sacct -u $USER -X -S today --format=JobID,JobName%16,State,Elapsed,ExitCode
tail -n 30 runs/run4/run4_<jobid>.log         # ends with the summary and "exit code 0"
```

Then add a row to [`runs/RUNS.md`](../runs/RUNS.md): run number, date, job
id(s), what it did, and the outcome (including failures).
