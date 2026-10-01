# Run 2: What the Logit Lens Shows About OpenVLA

**Data:** run 2 (SLURM job 55231832, commit `2de255d`). OpenVLA-7B finetuned on LIBERO-Spatial, fp16 on a V100S-32GB. 10 tasks × 5 starting layouts = 50 episodes, 42 successful (84.0%; OpenVLA reports 84.7%). 6,460 simulator steps in total.

**Numbers in this doc come from:**
- `python experiments.py --mode lens --analyze --out_dir runs/run2`: the three plots in `runs/run2/plots/`
- `python lens_report.py runs/run2 --csv`: every other table here; CSVs in `runs/run2/plots/lens_report_*.csv`

---

## 1. Summary in plain words

1. **For the arm's movement (x, y, z, roll, pitch, yaw), the answer appears only in the last few layers.** In the first ~20 of the model's 32 layers, the lens almost never points at the bin the model will finally output (under 8%). The lens first does better than "always guess the most common value" at layer 26–28. It first matches the final answer at least half the time at layer 28–30. The typical step stops changing its mind at layer 28–31.
2. **The gripper (open or close) is decided earlier, around layers 19–24.** By layer 24 the lens already matches the final gripper decision 83% of the time, while the movement parts are at 6–18%.
3. **But the gripper's head start is mostly for "keep doing what you're doing."** On the 207 steps where the gripper actually switches (open→close or close→open), the lens at layer 24 matches only 47% of the time, against 84% on steady steps. The decision to *change* the gripper forms almost as late as the movement.
4. **Rotation looks slightly earlier than position, but that is mostly because rotation is usually "don't rotate."** After comparing to the "always guess the most common value" baseline, all six movement parts become readable at about the same depth (layers 26–28).
5. **Early layers are not "unsure." They are confidently pointing at the wrong thing.** Their predictions are concentrated on a few bins (low entropy) that have little to do with the final action. So the early layers are not producing a rough draft of the action. They are doing other work (likely processing the image and instruction) that the lens can't read as an action.

These are statements about what the **logit lens can read out**, not about what information exists in each layer. See section 6 for what that difference means.

---

## 2. What the logit lens does, concretely

OpenVLA outputs an action as **7 tokens**, one per action part, in this fixed order: x, y, z, roll, pitch, yaw, gripper. Each token is one of **256 bins** (bin 0 = most negative value, bin 255 = most positive value for that part).

At every simulator step, for each of those 7 tokens, we take the model's internal state after each layer and ask: *"If the model stopped here and went straight to its output layer, which bin would it pick?"* Concretely, the state after layer *L* goes through the model's own final normalization and output layer, giving a score for each of the 256 bins, which are turned into probabilities.

- **Layer numbering:** layer 0 = the input embedding (before any processing); layers 1–32 = after each of Llama-2's 32 layers; layer 32 = the model's actual output.
- **Built-in check:** at layer 32, the lens must pick exactly the bin the model output. It did on all 45,220 action tokens in run 2.
- **Order matters:** each token is predicted after the previous ones. So when the model predicts the gripper (token 7), it has already written x through yaw for this step. When it predicts x (token 1), it has only the image and instruction.
- **No memory between steps:** the model sees only the current camera image and the instruction, not its previous actions. Anything "steady" across steps has to come from what it sees.

---

## 3. The values used, and how to read each one

| Value | Exact definition | How to read it in words |
|---|---|---|
| **Agreement** (exact) | Fraction of steps where layer *L*'s top bin equals the model's final bin. | "How often this layer already has the exact final answer." 100% at layer 32 by definition. |
| **Within ±k bins** | Fraction of steps where layer *L*'s top bin is within *k* bins of the final bin. We use k = 5 (±2% of that part's range) and k = 25 (±10%). | "How often this layer is roughly right," giving credit for near misses. |
| **Bin distance** | Average of \|layer bin − final bin\|, in bins. | "How far off this layer is, on average." One bin = 1/255 of that part's range (see table in section 4). |
| **KL divergence** KL(P_final ‖ P_layer) | In nats, over the 256 action bins. 0 = the layer's probabilities are identical to the final layer's. | "How surprised this layer would be by the final answer." Above ~5 nats the layer gives the final answer almost no probability. Below ~1 nat it is close to the final distribution. |
| **Entropy** of P_layer | In nats. Uniform over 256 bins = 5.55; one bin with all probability = 0. | "How spread out this layer's guess is." **Not** the same as being right: a layer can be confident and wrong. |
| **Settle layer** | For one step and one action part: the first layer from which **every later layer** picks the final bin. | "The layer where the model stops changing its mind." Settle = 32 means only the very last layer had it. |
| **First-hit layer** | First layer whose top bin equals the final bin, even if later layers move away again. | "The first time the final answer shows up." Always ≤ the settle layer. |
| **Chance baseline** | 1/256 = 0.4%. | What random guessing would score on exact agreement. |
| **Most-common-value baseline** | Score of a guesser that always outputs the most common final bin for that part. | **The more honest comparison.** A layer has to beat this before we can say it carries action information. It matters most for the gripper (52%) and roll (54%). |

**Unit of analysis:** each number pools all 6,460 steps. Steps from the same episode are not independent, so treat small differences (a few percentage points) with caution. Only 50 episodes are behind them.

---

## 4. Background: what the actions look like

| Part | Bins actually used | Most common bin (share of steps) | 1 bin in action units | Range (1st–99th percentile) |
|---|---|---|---|---|
| x | 256 | 112 (23.2%) | 0.0066 | −0.745 … +0.938 |
| y | 254 | 109 (32.0%) | 0.0060 | −0.662 … +0.876 |
| z | 256 | 127 (17.5%) | 0.0073 | −0.938 … +0.932 |
| roll | 200 | 129 (53.6%) | 0.0008 | −0.107 … +0.104 |
| pitch | 241 | 137 (36.6%) | 0.0015 | −0.207 … +0.177 |
| yaw | 234 | 142 (45.0%) | 0.0013 | −0.184 … +0.146 |
| gripper | **2** | 127 = close (52.0%) | n/a | bin 127 = close, bin 255 = open |

- **Position (x, y, z)** uses the full range of bins and has no strongly dominant value. It's a genuinely varied, continuous output.
- **Rotation** mostly sits near the middle bin, which means "don't rotate." That makes the most-common-value baseline high (37–54%).
- **The gripper** only ever uses 2 of the 256 bins in practice: it is a yes/no decision. Closed on 52% of steps, open on 48%.

---

## 5. Findings, with the numbers

### 5.1 Layers 0–16: no readable action

Exact agreement for every movement part stays at **0.1–0.9%** through layer 16, which is at the 1-in-256 chance level. Even "roughly right" (±10% of the range) is only 9–24%.

| layer | x | y | z | roll | pitch | yaw | gripper |
|---|---|---|---|---|---|---|---|
| 0 | 0.4% | 0.2% | 0.3% | 0.1% | 0.1% | 0.1% | 0.0% |
| 8 | 0.2% | 0.3% | 0.6% | 0.1% | 0.1% | 0.1% | 0.2% |
| 12 | 0.4% | 0.6% | 0.3% | 0.2% | 0.2% | 0.1% | 7.1% |
| 16 | 0.9% | 0.8% | 0.8% | 0.4% | 0.5% | 0.4% | 29.5% |

KL stays at 6–10 nats here: these layers put almost no probability on the final answer.

**One artifact to ignore:** at layer 8, x is "roughly right" (±10%) 37.5% of the time. That's because layer 8 picks bin 107 on 57% of steps, which happens to sit near x's most common value (112). It's a default guess, not information.

### 5.2 Layers 20–32: the movement action forms late, and all at once

Exact agreement with the final bin:

| layer | x | y | z | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| 20 | 3.1% | 4.2% | 7.7% | 2.3% | 4.2% | 2.2% |
| 24 | 6.4% | 11.5% | 14.0% | 18.0% | 13.2% | 16.0% |
| 26 | 11.1% | 19.4% | 20.1% | 27.6% | 24.0% | 30.6% |
| 28 | 24.7% | 37.2% | 31.3% | 55.2% | 45.1% | 50.0% |
| 30 | 50.0% | 62.6% | 59.7% | 78.0% | 71.6% | 71.7% |
| 31 | 65.9% | 76.2% | 73.4% | 84.8% | 82.1% | 82.6% |

"Roughly right" doesn't come much earlier. Within ±2% of the range (±5 bins), layer 24 is right only 19–32% of the time, and layer 28 only 42–65%. So it's not that middle layers know the rough direction and later layers fine-tune the exact value. The approximate value also appears late.

**Settle layer** (where the model stops changing its mind):

| part | median | middle 50% of steps | settled by layer 24 | settled by layer 28 | only at layer 32 |
|---|---|---|---|---|---|
| x | 31 | 29–32 | 3.4% | 19.4% | 34.1% |
| y | 30 | 28–31 | 6.8% | 31.3% | 23.8% |
| z | 30 | 28–32 | 9.7% | 26.0% | 26.6% |
| roll | 28 | 27–30 | 12.1% | 51.2% | 15.2% |
| pitch | 29 | 27–31 | 9.2% | 40.4% | 17.9% |
| yaw | 29 | 26–31 | 11.2% | 45.1% | 17.4% |
| gripper | 21 | 17–24 | 80.1% | 97.5% | 0.7% |

**In words:** for x, a third of all steps change their answer at the very last layer. The final layer is still doing real work for position.

### 5.3 Comparing against the most-common-value baseline

This is the fairest test of "when does a layer know something about the action":

| part | baseline (always guess most common) | first layer that beats it | first layer at ≥ 50% agreement |
|---|---|---|---|
| x | 23.2% | 28 | 30 |
| y | 32.0% | 28 | 30 |
| z | 17.5% | 26 | 30 |
| roll | 53.6% | 28 | 28 |
| pitch | 36.6% | 28 | 29 |
| yaw | 45.0% | 28 | 29 |
| gripper | 52.0% | **19** | 18 |

Rotation *looks* earlier than position in the raw agreement plot. Against the baseline, all six movement parts become informative at the same depth, **layers 26–28**. Rotation's apparent head start comes mostly from "don't rotate" being an easy, common answer.

### 5.4 The gripper: earlier, but mostly for steady states

The gripper beats its baseline at layer 19 and reaches 83% agreement at layer 24 and 98% at layer 28. That's about 8 layers ahead of the movement parts.

Splitting steps by whether the gripper decision **changes** from the previous step:

| steps | count | median settle layer | agreement at 16 | at 20 | at 24 | at 28 |
|---|---|---|---|---|---|---|
| steady (same as last step) | 6,203 | 21 | 29.6% | 59.7% | 83.8% | 98.7% |
| switch (open↔close) | 207 | 26 | 17.4% | 39.1% | 46.9% | 73.4% |

**In words:** most of the gripper's early lead is the model saying "stay open" or "stay closed." The model has no memory of its last action, so this must come from what it sees, e.g. whether the gripper looks open in the image. Deciding to *grasp* or *release* forms around layers 24–28, close to when the movement forms.

**Also keep in mind:** the gripper is the 7th token, so when it is predicted, the six movement values for this step are already written into the model's input.

### 5.5 Early layers are confident, not uncertain

| layer | entropy, movement parts (nats) | entropy, gripper (nats) |
|---|---|---|
| 0 | 1.9–2.6 | 2.75 |
| 8 | 3.7–4.4 | 3.47 |
| 16 | 2.5–2.7 | 1.93 |
| 24 | 1.9–2.2 | 0.65 |
| 32 (final) | 0.3–0.7 | 0.01 |

Uniform uncertainty would be 5.55 nats. Early layers sit well below that while being almost always wrong. Each one concentrates its guess on a few bins that are mostly not the final answer. Example: at layer 0, x picks bin 173 on 100% of steps. That's expected: at layer 0 the "state" is just the embedding of the current input token, which is identical every step.

**So low entropy in early layers does not mean "the model already knows."** Use agreement and KL, not entropy, to judge when the action appears.

The **final layer's** own entropy is 0.3–0.7 nats for movement: the model often splits its probability between a few neighbouring bins. The gripper is near 0: always fully committed.

### 5.6 Does it depend on the situation?

Movement parts pooled (x through yaw):

| situation | steps | median settle layer | agreement at 28 | within ±5 bins at 24 |
|---|---|---|---|---|
| gripper open (reaching) | 3,098 | 29 | 46.2% | 29.1% |
| gripper closed (carrying) | 3,362 | 30 | 35.3% | 22.9% |
| episode time 0–25% | 1,635 | 29 | 47.0% | 30.5% |
| 25–50% | 1,607 | 30 | 41.6% | 25.3% |
| 50–75% | 1,622 | 30 | 38.0% | 24.4% |
| 75–100% | 1,596 | 30 | 35.6% | 23.0% |
| successful episodes | 4,700 | 29 | 42.2% | 26.6% |
| failed episodes | 1,760 | 30 | 36.2% | 23.8% |

**In words:** the action forms slightly earlier while reaching than while carrying, and slightly earlier early in an episode. The differences are small: about one layer in the median, and 6–11 points of agreement. With only 50 episodes, this is a lead worth following up, not a finding.

---

## 6. What this does and does not tell us

**It does tell us:**
- When the model's **own output layer** can read the action out of each layer's state.
- That, read this way, OpenVLA's movement commands are built in the last ~6 layers (26–32). The yes/no gripper decision is readable about 8 layers earlier, mostly when it isn't changing.

**It does not tell us:**
- **Whether the information is present in middle layers in a form the output layer can't read.** The logit lens uses the final layer's "reading glasses" on every layer. A middle layer could encode the action in a different format and look empty here. A **tuned lens** (a small learned translation per layer) or a **linear probe** would answer that.
- **Whether early layers do "perception."** The lens only reads action tokens. That early layers don't show actions is consistent with them doing perception, but it doesn't prove it.
- **Anything beyond this setup:** one model, one task suite (LIBERO-Spatial), fp16 on a V100, 50 episodes. Steps within an episode are correlated, so the effective sample size is closer to 50 than to 6,460.

---

## 7. Reproducing

```bash
# on OSC (needs the run's lens/ folder, which is not in git)
python experiments.py --mode lens --analyze --out_dir runs/run2
python lens_report.py runs/run2 --csv
```

- **Outputs in git:**
  - `runs/run2/plots/lens_agreement.png`, `lens_bucket_distance.png`, `lens_kl.png` and their CSVs
  - `lens_report_by_layer.csv`: every layer × action part, with agreement, ±5, ±25, distance, KL, entropy
  - `lens_report_settle.csv`
- **On OSC only:** `runs/run2/lens/*.npz` (14 MB) and `runs/run2/hidden/*.npz` (1.7 GB).
