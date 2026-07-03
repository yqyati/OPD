# OPD Experiment Records

## 2026-06-27: Top50 n=1 Ablation

### Setup

Common evaluation setting:

```text
tasks: AIME24, AIME25, AMC23
eval n: 16
eval max tokens: 31744
temperature: 0.7
top_p: 0.95
thinking: disabled
grader: rule-based
```

Training setting for the new run:

```text
script: /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/tmp_now.bash
base config: scripts/train/opd_top50.sh
method: token_reward_direct OPD
train data: datasets/opd_prompt_filter/opd_prompt_score_top50.parquet
student: /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/DeepSeek-R1-Distill-Qwen-1.5B
teacher: /mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/model/JustRL-DeepSeek-1.5B
train n: 1
max response length: 7168
top-k: 16
top-k strategy: only_stu
reward weight mode: student_p
lr: 1e-5
total epochs: 1
final checkpoint: global_step_139
```

Artifacts:

```text
checkpoint:
checkpoint/token_reward_direct_DAPO-Math-17k-TeacherAligned-Top50-n1_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B_7168-T_1.0-Tch_1.0-n_1-mbs_64-lr_1e-5-topk_16-topk_strategy_only_stu-rw_student_p-2026-06-27_18-43-27

merged model:
merged_models/opd_top50_n1_lr1e-5_step139

eval output:
outputs/eval/justrl_eval_outputs_31744/opd_top50_n1_lr1e-5_step139

grading:
outputs/eval/justrl_eval_outputs_31744/opd_top50_n1_lr1e-5_step139/grading_results.json

log:
logs/run_20260627_184317.log
```

### Results

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Student baseline | 0.283333 | 0.237500 | 0.732812 | 0.417882 |
| Teacher | 0.516667 | 0.372917 | 0.879687 | 0.589757 |
| Top50 OPD n=4 step139 | 0.462500 | 0.333333 | 0.845313 | 0.547049 |
| Top50 OPD n=1 step139 | 0.458333 | 0.345833 | 0.825000 | 0.543056 |
| Random50 seed1 OPD n=4 step139 | 0.506250 | 0.322917 | 0.845313 | 0.558160 |

### Deltas

Top50 `n=1` minus Top50 `n=4`:

| Metric | Delta |
| --- | ---: |
| AIME24 | -0.004167 |
| AIME25 | +0.012500 |
| AMC23 | -0.020313 |
| Avg | -0.003993 |

Top50 `n=1` minus Random50 seed1 `n=4`:

| Metric | Delta |
| --- | ---: |
| AIME24 | -0.047917 |
| AIME25 | +0.022917 |
| AMC23 | -0.020313 |
| Avg | -0.015104 |

### Notes

- Top50 `n=1` is close to Top50 `n=4`, but slightly lower on average.
- The average drop from Top50 `n=4` to Top50 `n=1` is small (`-0.003993`), so rollout count is probably not the main bottleneck for Top50.
- Top50 `n=1` still underperforms Random50 seed1 `n=4` on average (`-0.015104`).
- Current evidence still points to hard Top50 selection losing useful distribution coverage, rather than `n=4` rollout noise being the main issue.

## 2026-07-03 Teacher Context Follow-Up Directions

### Current Observation

The raw teacher-prefix experiments do not support continuing to longer fixed prefixes.

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Full OPD all-data `n=1` | 0.450000 | 0.337500 | 0.832813 | 0.540104 |
| TeacherPrefix128 + suffix OPD `n=1` | 0.439583 | 0.356250 | 0.834375 | 0.543403 |
| TeacherPrefix256 + suffix OPD `n=1` | 0.429167 | 0.331250 | 0.846875 | 0.535764 |
| Full OPD all-data `n=4` | 0.466667 | 0.318750 | 0.878125 | 0.554514 |

Key interpretation:

- TeacherPrefix128 is almost tied with full all-data `n=1`, but does not show a robust gain.
- TeacherPrefix256 is worse than both TeacherPrefix128 and full all-data `n=1`.
- Longer raw teacher prefix increases the risk of forcing the student to continue from a truncated, incomplete teacher reasoning state.
- This creates a train-test mismatch: training uses `prompt + teacher partial CoT`, while evaluation uses only the original prompt.
- Therefore, fixed raw teacher prefix length should not be extended to 512/1024 without changing the mechanism.

### Direction 1: Teacher Hint / Strategy Conditioning

Replace raw teacher CoT prefix with a short, complete strategy hint.

Instead of storing teacher's first 128/256 generated tokens, generate a concise planning hint:

```text
Given the math problem, write a concise strategy hint for solving it.
Do not solve the problem.
Do not include the final answer.
Use at most 2 sentences.
```

Training prompt format:

```text
Problem:
...

Teacher hint:
...

Now solve the problem step by step and put your final answer in \boxed{}.
```

Why this is better than raw prefix:

- The hint is a complete semantic unit, not a truncated partial CoT.
- The student still performs its own full rollout instead of continuing teacher's unfinished sentence.
- The teacher provides high-level problem representation or strategy rather than hard token-level continuation.
- Offline generation cost is paid once; rollout-time cost is almost unchanged.
- This is easier to explain as a paper method: teacher provides strategy conditioning, student keeps autonomous rollout, OPD trains on the student suffix.

Suggested experiment:

```text
TeacherHint64 + suffix OPD n=1
TeacherHint64 + 50% dropout + suffix OPD n=1
```

Expected benefit:

- Better than raw TeacherPrefix128/256 if the main issue is truncated CoT and continuation mismatch.
- Especially worth checking on AIME25, where TeacherPrefix256 showed increased format errors.

### Direction 2: Complete Segment Prefix Instead Of Fixed Token Prefix

If we still want teacher text as context, do not cut at a fixed token count. Generate a complete reasoning segment and stop at a semantic boundary.

Possible teacher instruction:

```text
Provide only the first complete reasoning step or setup.
Stop after the setup is complete.
Do not solve the full problem.
Do not give the final answer.
```

Alternative:

```text
Provide a concise plan in one paragraph.
Do not solve the problem fully.
```

Why this may help:

- It avoids cutting teacher CoT in the middle of a sentence, formula, or derivation.
- The context becomes a complete setup or plan, which is easier for the student to continue from.
- It keeps the spirit of teacher-prefix intervention while reducing prefix truncation artifacts.

Main risk:

- It still changes the training prompt distribution.
- If evaluation does not include the same teacher segment, the student may depend on context that is absent at test time.

Suggested experiment:

```text
TeacherSegmentSetup + suffix OPD n=1
TeacherSegmentPlan + suffix OPD n=1
```

This is lower priority than Teacher Hint because it still resembles teacher-context training and may keep train-test mismatch.

### Direction 3: Prefix / Hint Dropout And Curriculum

Do not apply teacher context to 100% of training examples. Mix original prompts and teacher-context prompts.

Simple dropout:

```text
50% original prompt
50% teacher hint or teacher segment prompt
```

Curriculum version:

```text
early training: higher teacher-context probability
late training: lower teacher-context probability
final stage: mostly or fully original prompt
```

Why this matters:

- Current raw prefix experiments used 100% teacher-prefix data.
- Evaluation uses no teacher prefix, so 100% teacher-context training can introduce dependency on unavailable context.
- Dropout forces the policy to remain competent under the original prompt distribution.

Suggested experiment:

```text
TeacherHint64 + 50% dropout + suffix OPD n=1
TeacherHint64 + linear dropout 0.5 -> 0.0 + suffix OPD n=1
```

This is likely the most practical fix if teacher hints help but introduce distribution shift.

### Direction 4: Teacher As Judge / Selector For `n>1` Rollouts

Use teacher as a critic or selector instead of a generator. The teacher does not provide prefix context. The student still performs normal rollouts from the original prompt.

Basic form:

```text
For each prompt:
  sample n student rollouts
  let teacher judge trajectory or segment quality
  use teacher score to select/reweight useful rollouts or tokens
  train OPD on the selected/reweighted student data
```

Possible teacher judgment targets:

- Whole trajectory quality: is this solution correct, recoverable, or clearly wrong?
- Early segment quality: does the first 512/1024 tokens contain a sound plan?
- Branch quality among `n` rollouts: which rollout is most promising?
- Token/segment-level veto: reject trajectories with obvious early mathematical mistakes.

Why this is promising:

- It uses `n>1` group information directly.
- It avoids contaminating student rollout distribution with teacher-generated context.
- It keeps training and evaluation closer: the student always starts from the original prompt.
- It can be framed as teacher-guided rollout selection or teacher-critic OPD.

Engineering considerations:

- More expensive than offline hint generation if teacher judges every rollout.
- Cost can be controlled by judging only short prefixes or only a subset of rollouts.
- A cheap first version can judge only the first 512/1024 generated tokens or only final answer plus concise solution.

Suggested experiment:

```text
Full OPD n=4 + teacher rank/select best rollout
Full OPD n=4 + teacher early-segment judge
Full OPD n=4 + teacher reject clearly bad trajectories
```

This is the most relevant direction if the goal is to exploit `n>1` group comparison for a paper.

### Recommended Priority

Current priority order:

| Priority | Direction | Reason |
| ---: | --- | --- |
| 1 | TeacherHint64 + dropout | Avoids truncated CoT and reduces train-test mismatch with low engineering cost |
| 2 | Teacher-as-judge for `n>1` | Most aligned with using group comparison and paper novelty |
| 3 | Complete segment prefix | Better than fixed raw prefix, but still has context mismatch risk |
| 4 | Raw fixed teacher prefix | Current 128/256 evidence is weak or negative |

Recommended next concrete run:

```text
TeacherHint64 + 50% dropout + suffix OPD n=1
```

If this does not beat full all-data `n=1`, stop teacher-context methods and move to teacher-as-judge for `n>1` rollouts.
