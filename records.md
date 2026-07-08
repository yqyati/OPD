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

## 2026-07-03 TeacherPrefix128 SFT-Prefix + Suffix OPD Result

Experiment:

```text
TeacherPrefix128 + SFT(prefix, coef=0.1) + suffix OPD n=1
train data: datasets/teacher_prefix/opd_prompt_all_teacher_prefix128.parquet
checkpoint: global_step_279
merged model: merged_models/opd_teacher_prefix128_sftprefix_suffix_opd_n1_lr1e-5_coef0.1_step279
eval: n=16, temperature=0.7, top_p=0.95, max_tokens=31744, disable_thinking
grading: rule-based only, no CompassVerifier
```

Results:

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Student baseline | 0.283333 | 0.237500 | 0.732812 | 0.417882 |
| Teacher | 0.516667 | 0.372917 | 0.879687 | 0.589757 |
| Full OPD all-data n=1 | 0.450000 | 0.337500 | 0.832813 | 0.540104 |
| Full OPD all-data n=4 | 0.466667 | 0.318750 | 0.878125 | 0.554514 |
| TeacherPrefix128 + suffix OPD n=1 | 0.439583 | 0.356250 | 0.834375 | 0.543403 |
| TeacherPrefix256 + suffix OPD n=1 | 0.429167 | 0.331250 | 0.846875 | 0.535764 |
| Our method: TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | 0.495833 | 0.325000 | 0.831250 | 0.550694 |

Detailed eval metrics for the SFT-prefix run:

| Task | mean_score | best_score | solve_none | solve_all | avg_output_length | format_error_rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIME24 | 0.495833 | 0.766667 | 7 | 6 | 10222.66 | 26 |
| AIME25 | 0.325000 | 0.500000 | 15 | 5 | 9919.36 | 53 |
| AMC23 | 0.831250 | 0.975000 | 1 | 20 | 6868.01 | 33 |

Teacher gap for `TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1`:

| Metric | Gap vs Teacher |
| --- | ---: |
| AIME24 | -0.020834 |
| AIME25 | -0.047917 |
| AMC23 | -0.048437 |
| Avg | -0.039063 |

Interpretation:

- Adding prefix SFT improves over `TeacherPrefix128 + suffix OPD n=1` by `+0.007291` average.
- It also improves over `Full OPD all-data n=1` by `+0.010590` average.
- It is close to, but still slightly below, `Full OPD all-data n=4` by `-0.003820` average.
- The gain is concentrated on AIME24: `0.495833`, higher than the listed OPD baselines and close to the teacher `0.516667`.
- AIME25 and AMC23 do not improve over the strongest corresponding baselines, so the effect is not uniform.
- Training was stable. The prefix SFT loss stayed bounded but did not show a clean monotonic decrease, suggesting coef `0.1` may be conservative.

Next run:

```text
TeacherPrefix128 + SFT(prefix, coef=0.2) + suffix OPD n=1
```

The script `/mnt/shared-storage-gpfs2/p1-shared-2/yangqingyu/tmp_now.bash` was updated to default `TEACHER_PREFIX_SFT_LOSS_COEF=0.2`.

## 2026-07-03 TeacherPrefix128 SFT-Prefix coef=0.2 Result

Experiment:

```text
TeacherPrefix128 + SFT(prefix, coef=0.2) + suffix OPD n=1
train data: datasets/teacher_prefix/opd_prompt_all_teacher_prefix128.parquet
checkpoint: global_step_279
merged model: merged_models/opd_teacher_prefix128_sftprefix_suffix_opd_n1_lr1e-5_coef0.2_step279
eval: n=16, temperature=0.7, top_p=0.95, max_tokens=31744, disable_thinking
grading: rule-based only, no CompassVerifier
```

Results:

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Full OPD all-data n=1 | 0.450000 | 0.337500 | 0.832813 | 0.540104 |
| Full OPD all-data n=4 | 0.466667 | 0.318750 | 0.878125 | 0.554514 |
| TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | 0.495833 | 0.325000 | 0.831250 | 0.550694 |
| TeacherPrefix128 + SFT(prefix, 0.2) + suffix OPD n=1 | 0.464583 | 0.339583 | 0.812500 | 0.538889 |

Detailed eval metrics for the coef `0.2` run:

| Task | mean_score | best_score | solve_none | solve_all | avg_output_length | format_error_rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIME24 | 0.464583 | 0.800000 | 6 | 3 | 10288.17 | 24 |
| AIME25 | 0.339583 | 0.533333 | 14 | 5 | 9621.23 | 44 |
| AMC23 | 0.812500 | 0.975000 | 1 | 19 | 7108.48 | 35 |

Interpretation:

- Increasing SFT coef from `0.1` to `0.2` hurts average score: `0.550694 -> 0.538889`.
- The only task-level gain is AIME25: `0.325000 -> 0.339583`.
- AIME24 drops clearly: `0.495833 -> 0.464583`.
- AMC23 also drops: `0.831250 -> 0.812500`.
- This suggests the useful range for prefix SFT is not larger than `0.1`; pushing harder on teacher-prefix imitation starts to interfere with suffix OPD behavior.

## 2026-07-03 TeacherPrefix256 SFT-Prefix coef=0.1 Result

Experiment:

```text
TeacherPrefix256 + SFT(prefix, coef=0.1) + suffix OPD n=1
train data: datasets/teacher_prefix/opd_prompt_all_teacher_prefix256.parquet
checkpoint: global_step_279
merged model: merged_models/opd_teacher_prefix256_sftprefix_suffix_opd_n1_lr1e-5_coef0.1_step279
eval: n=16, temperature=0.7, top_p=0.95, max_tokens=31744, disable_thinking
grading: rule-based only, no CompassVerifier
```

Results:

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Full OPD all-data n=1 | 0.450000 | 0.337500 | 0.832813 | 0.540104 |
| Full OPD all-data n=4 | 0.466667 | 0.318750 | 0.878125 | 0.554514 |
| TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | 0.495833 | 0.325000 | 0.831250 | 0.550694 |
| TeacherPrefix256 + SFT(prefix, 0.1) + suffix OPD n=1 | 0.431250 | 0.337500 | 0.850000 | 0.539583 |

Detailed eval metrics for the 256-prefix coef `0.1` run:

| Task | mean_score | best_score | solve_none | solve_all | avg_output_length | format_error_rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIME24 | 0.431250 | 0.733333 | 8 | 5 | 10213.44 | 31 |
| AIME25 | 0.337500 | 0.533333 | 14 | 6 | 9818.06 | 43 |
| AMC23 | 0.850000 | 0.975000 | 1 | 21 | 6973.64 | 26 |

Interpretation:

- `TeacherPrefix256 + SFT(prefix, 0.1)` does not improve over full OPD all-data n=1 on average: `0.539583` vs `0.540104`.
- It is lower than `TeacherPrefix128 + SFT(prefix, 0.1)` by `-0.011111` average.
- The longer prefix helps AMC23 (`0.850000`) but hurts AIME24 sharply (`0.431250`).
- This supports the earlier concern that a longer fixed teacher prefix can push the student into a less useful or partially truncated reasoning state.

## 2026-07-04 TeacherGuide128 SFT-Guide coef=0.05 Result

Experiment:

```text
TeacherGuide128 + SFT(guide, coef=0.05) + OPD n=1
train data: datasets/teacher_hint/opd_prompt_all_teacher_guide128.parquet
checkpoint: global_step_279
merged model: merged_models/opd_teacher_guide128_sftguide_opd_n1_lr1e-5_coef0.05_step279
eval: n=16, temperature=0.7, top_p=0.95, max_tokens=31744, disable_thinking
grading: rule-based only, no CompassVerifier
```

Results:

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Full OPD all-data n=1 | 0.450000 | 0.337500 | 0.832813 | 0.540104 |
| Full OPD all-data n=4 | 0.466667 | 0.318750 | 0.878125 | 0.554514 |
| TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | 0.495833 | 0.325000 | 0.831250 | 0.550694 |
| TeacherGuide128 + SFT(guide, 0.05) + OPD n=1 | 0.450000 | 0.339583 | 0.829688 | 0.539757 |

Detailed eval metrics:

| Task | mean_score | best_score | solve_none | solve_all | avg_output_length | format_error_rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIME24 | 0.450000 | 0.833333 | 5 | 4 | 10157.35 | 15 |
| AIME25 | 0.339583 | 0.433333 | 17 | 3 | 9691.36 | 45 |
| AMC23 | 0.829688 | 0.950000 | 2 | 18 | 7137.89 | 24 |

Guide-data diagnosis:

| Metric | Value |
| --- | ---: |
| rows | 17917 |
| mean guide tokens | 128.0 |
| median guide tokens | 128.0 |
| max guide tokens | 128 |
| finish_reason=length | 17917 / 17917 |

Interpretation:

- The score is essentially tied with full OPD all-data n=1: `0.539757` vs `0.540104`.
- It is below the strongest raw-prefix run, `TeacherPrefix128 + SFT(prefix, 0.1)`, by `-0.010937`.
- The generated "guide" data is not actually short guide data: every sample hit the `128` token cap and finished by length.
- Manual samples show the teacher often restates the problem or starts verbose CoT-like text, so this run should be interpreted as a failed guide-generation attempt rather than strong evidence against concise strategy guides.

## 2026-07-05 Qwen3 Teacher Ablation: GRPO Teacher vs No-Think Base Teacher

Experiment:

```text
Student: Qwen3-1.7B-Base
GRPO teacher: Qwen3-4B-Base-GRPO
No-think teacher: Qwen3-4B-Base with enable_thinking=False
train data for plain OPD: datasets/dapo-math-17k-teacher-aligned.parquet
train data for no-think prefix: datasets/teacher_prefix/qwen3_base_dapo_math_17k_teacher_prefix128.parquet
train data for GRPO prefix: datasets/teacher_prefix/qwen3_grpo_dapo_math_17k_teacher_prefix128.parquet
train data for GRPO pure SFT: datasets/sft/qwen3_grpo_teacher_prefix128_pure_sft.parquet
rollout n: 1
lr: 1e-5
prefix length: 128 tokens
prefix SFT loss coef: 0.1
eval: n=16, temperature=0.7, top_p=0.95, max_tokens=31744, disable_thinking
grading: rule-based only, no CompassVerifier
```

Results:

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| GRPO teacher plain OPD | 0.062500 | 0.056250 | 0.425000 | 0.181250 |
| GRPO teacher prefix128+sft0.1 | 0.118750 | 0.085417 | 0.468750 | 0.224306 |
| GRPO teacher prefix128 pure SFT | 0.052083 | 0.056250 | 0.350000 | 0.152778 |
| No-think teacher plain OPD | 0.052083 | 0.029167 | 0.239063 | 0.106771 |
| No-think teacher prefix128+sft0.1 | 0.060417 | 0.045833 | 0.350000 | 0.152083 |

No-think teacher internal gain:

| Comparison | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| No-think prefix128+sft0.1 - no-think plain OPD | +0.008334 | +0.016666 | +0.110937 | +0.045312 |

Detailed eval metrics:

| Run | Task | mean_score | best_score | solve_none | solve_all | avg_output_length | format_error_rollouts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO teacher plain OPD | AIME24 | 0.062500 | 0.166667 | 25 | 0 | 16013.6 | 188 |
| GRPO teacher plain OPD | AIME25 | 0.056250 | 0.200000 | 24 | 0 | 13681.7 | 166 |
| GRPO teacher plain OPD | AMC23 | 0.425000 | 0.800000 | 8 | 2 | 9790.6 | 164 |
| GRPO teacher prefix128+sft0.1 | AIME24 | 0.118750 | 0.200000 | 24 | 0 | 14022.6 | 166 |
| GRPO teacher prefix128+sft0.1 | AIME25 | 0.085417 | 0.333333 | 20 | 1 | 12030.5 | 132 |
| GRPO teacher prefix128+sft0.1 | AMC23 | 0.468750 | 0.800000 | 8 | 9 | 8694.4 | 127 |
| GRPO teacher prefix128 pure SFT | AIME24 | 0.052083 | 0.200000 | 24 | 0 | 10834.7 | 221 |
| GRPO teacher prefix128 pure SFT | AIME25 | 0.056250 | 0.200000 | 24 | 0 | 10094.7 | 186 |
| GRPO teacher prefix128 pure SFT | AMC23 | 0.350000 | 0.725000 | 11 | 1 | 6929.4 | 166 |
| No-think teacher plain OPD | AIME24 | 0.052083 | 0.233333 | 23 | 0 | 9762.2 | 175 |
| No-think teacher plain OPD | AIME25 | 0.029167 | 0.233333 | 23 | 0 | 8654.9 | 148 |
| No-think teacher plain OPD | AMC23 | 0.239063 | 0.625000 | 15 | 0 | 7029.1 | 167 |
| No-think teacher prefix128+sft0.1 | AIME24 | 0.060417 | 0.200000 | 24 | 0 | 9174.5 | 151 |
| No-think teacher prefix128+sft0.1 | AIME25 | 0.045833 | 0.200000 | 24 | 0 | 5699.5 | 83 |
| No-think teacher prefix128+sft0.1 | AMC23 | 0.350000 | 0.800000 | 8 | 2 | 4790.3 | 85 |

Output-format notes:

| Run | AIME24 Assistant prefix | AIME25 Assistant prefix | AMC23 Assistant prefix | AIME24 unicode prefix | AIME25 unicode prefix | AMC23 unicode prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No-think teacher plain OPD | 228 / 480 | 230 / 480 | 260 / 640 | 157 / 480 | 133 / 480 | 240 / 640 |
| No-think teacher prefix128+sft0.1 | 371 / 480 | 368 / 480 | 442 / 640 | 20 / 480 | 19 / 480 | 49 / 640 |
| GRPO teacher prefix128 pure SFT | 0 / 480 | 0 / 480 | 0 / 640 | 0 / 480 | 0 / 480 | 0 / 640 |

Interpretation:

- No-think teacher is substantially weaker than GRPO teacher in this OPD setting.
- Within the no-think teacher setting, `TeacherPrefix128 + SFTPrefix + SuffixOPD` still improves over plain OPD: average `0.152083` vs `0.106771`.
- GRPO teacher pure SFT is not enough by itself: average `0.152778`, below plain OPD `0.181250` and far below joint `GRPO teacher prefix128+sft0.1` `0.224306`.
- The pure SFT ablation removes obvious `Assistant:` and unicode/garble prefixes, but does not improve task accuracy. This supports the interpretation that SFT prefix stabilizes the beginning distribution, while suffix OPD is needed for answer quality.
- The no-think prefix run lowers output length and format errors relative to no-think plain OPD, especially on AIME25 and AMC23.
- However, no-think teacher runs show many `Assistant:` prefixes in generated outputs. This did not happen in the GRPO teacher prefix run, where obvious `Assistant:` and unicode/garble prefixes were previously observed as zero.
- Main-line Qwen3 result should remain `GRPO teacher prefix128+sft0.1`, with average `0.224306`.

## 2026-07-06 Cross-Experiment Summary Table

This table consolidates the completed DeepSeek/JustRL and Qwen3 ablations.

| Family | Student | Teacher | Method | SFT target | OPD target | AIME24 | AIME25 | AMC23 | Avg |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Student baseline | none | none | 0.283333 | 0.237500 | 0.732812 | 0.417882 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Teacher | none | none | 0.516667 | 0.372917 | 0.879687 | 0.589757 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Full OPD all-data n=1 | none | full student rollout | 0.450000 | 0.337500 | 0.832812 | 0.540104 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Full OPD all-data n=4 | none | full student rollout | 0.466667 | 0.318750 | 0.878125 | 0.554514 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix128 + suffix OPD n=1 | none | suffix after teacher prefix context | 0.439583 | 0.356250 | 0.834375 | 0.543403 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix256 + suffix OPD n=1 | none | suffix after teacher prefix context | 0.429167 | 0.331250 | 0.846875 | 0.535764 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix128 pure SFT | teacher prefix 128 | none | 0.279167 | 0.187500 | 0.595313 | 0.353993 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Full-response SFT | full teacher response | none | 0.089583 | 0.081250 | 0.428125 | 0.199653 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Filtered full-response SFT | correct full teacher response | none | 0.081250 | 0.068750 | 0.428125 | 0.192708 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix128 pure SFT init + plain OPD n=1 | teacher prefix 128 as init | full student rollout | 0.441667 | 0.327083 | 0.832812 | 0.533854 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | Batch mixed full-SFT + plain OPD | full teacher response on half batch | full student rollout on half batch | 0.445833 | 0.327083 | 0.820312 | 0.531076 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | teacher prefix 128 | suffix after teacher prefix context | 0.495833 | 0.325000 | 0.831250 | 0.550694 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix128 + SFT(prefix, 0.2) + suffix OPD n=1 | teacher prefix 128 | suffix after teacher prefix context | 0.464583 | 0.339583 | 0.812500 | 0.538889 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherPrefix256 + SFT(prefix, 0.1) + suffix OPD n=1 | teacher prefix 256 | suffix after teacher prefix context | 0.431250 | 0.337500 | 0.850000 | 0.539583 |
| DeepSeek/JustRL | DeepSeek-R1-Distill-Qwen-1.5B | JustRL-DeepSeek-1.5B | TeacherGuide128 + SFT(guide, 0.05) + OPD n=1 | generated guide 128 | student rollout | 0.450000 | 0.339583 | 0.829688 | 0.539757 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | Plain OPD n=1 | none | full student rollout | 0.062500 | 0.056250 | 0.425000 | 0.181250 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | TeacherPrefix128 pure SFT | teacher prefix 128 | none | 0.052083 | 0.056250 | 0.350000 | 0.152778 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base-GRPO | TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | teacher prefix 128 | suffix after teacher prefix context | 0.118750 | 0.085417 | 0.468750 | 0.224306 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | Plain OPD n=1 | none | full student rollout | 0.052083 | 0.029167 | 0.239063 | 0.106771 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | TeacherPrefix128 pure SFT | teacher prefix 128 | none | 0.050000 | 0.029167 | 0.282813 | 0.120660 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | Full-response SFT | full teacher response | none | 0.018750 | 0.006250 | 0.165625 | 0.063542 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | Filtered full-response SFT | correct full teacher response | none | 0.066667 | 0.025000 | 0.315625 | 0.135764 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | TeacherPrefix128 pure SFT init + plain OPD n=1 | teacher prefix 128 as init | full student rollout | 0.050000 | 0.031250 | 0.251563 | 0.110938 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | Batch mixed full-SFT + plain OPD | full teacher response on half batch | full student rollout on half batch | 0.045833 | 0.035417 | 0.245313 | 0.108854 |
| Qwen3 | Qwen3-1.7B-Base | Qwen3-4B-Base no-think | TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD n=1 | teacher prefix 128 | suffix after teacher prefix context | 0.060417 | 0.045833 | 0.350000 | 0.152083 |

### Split By Student-Teacher Pair

The original consolidated table above is kept unchanged. The same rows are split below by student-teacher pair for easier comparison within each setting.

#### DeepSeek-R1-Distill-Qwen-1.5B student + JustRL-DeepSeek-1.5B teacher

| Method | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Plain OPD | 0.450000 | 0.337500 | 0.832812 | 0.540104 |
| TeacherPrefix128 pure SFT | 0.279167 | 0.187500 | 0.595313 | 0.353993 |
| Full-response SFT | 0.089583 | 0.081250 | 0.428125 | 0.199653 |
| Filtered full-response SFT | 0.081250 | 0.068750 | 0.428125 | 0.192708 |
| TeacherPrefix128 pure SFT init + plain OPD | 0.441667 | 0.327083 | 0.832812 | 0.533854 |
| Batch mixed full-SFT + plain OPD | 0.445833 | 0.327083 | 0.820312 | 0.531076 |
| Our method: TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD | 0.495833 | 0.325000 | 0.831250 | 0.550694 |

#### Qwen3-1.7B-Base student + Qwen3-4B-Base-GRPO teacher

| Method | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Plain OPD | 0.062500 | 0.056250 | 0.425000 | 0.181250 |
| TeacherPrefix128 pure SFT | 0.052083 | 0.056250 | 0.350000 | 0.152778 |
| Our method: TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD | 0.118750 | 0.085417 | 0.468750 | 0.224306 |

#### Qwen3-1.7B-Base student + Qwen3-4B-Base no-think teacher

| Method | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Plain OPD | 0.052083 | 0.029167 | 0.239063 | 0.106771 |
| TeacherPrefix128 pure SFT | 0.050000 | 0.029167 | 0.282813 | 0.120660 |
| Full-response SFT | 0.018750 | 0.006250 | 0.165625 | 0.063542 |
| Filtered full-response SFT | 0.066667 | 0.025000 | 0.315625 | 0.135764 |
| TeacherPrefix128 pure SFT init + plain OPD | 0.050000 | 0.031250 | 0.251563 | 0.110938 |
| Batch mixed full-SFT + plain OPD | 0.045833 | 0.035417 | 0.245313 | 0.108854 |
| Our method: TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD | 0.060417 | 0.045833 | 0.350000 | 0.152083 |

Key comparisons:

| Family | Comparison | AIME24 | AIME25 | AMC23 | Avg |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek/JustRL | Prefix128 pure SFT - Full OPD n=1 | -0.170833 | -0.150000 | -0.237499 | -0.186111 |
| DeepSeek/JustRL | Full-response SFT - Full OPD n=1 | -0.360417 | -0.256250 | -0.404687 | -0.340451 |
| DeepSeek/JustRL | Prefix128 pure SFT init + plain OPD - Full OPD n=1 | -0.008333 | -0.010417 | +0.000000 | -0.006250 |
| DeepSeek/JustRL | Batch mixed full-SFT + plain OPD - Full OPD n=1 | -0.004167 | -0.010417 | -0.012500 | -0.009028 |
| DeepSeek/JustRL | Prefix128 SFT(0.1)+suffix OPD - Batch mixed full-SFT + plain OPD | +0.050000 | -0.002083 | +0.010938 | +0.019618 |
| DeepSeek/JustRL | Prefix128 SFT(0.1)+suffix OPD - Full OPD n=1 | +0.045833 | -0.012500 | -0.001562 | +0.010590 |
| DeepSeek/JustRL | Prefix128 SFT(0.1)+suffix OPD - Prefix128 pure SFT | +0.216666 | +0.137500 | +0.235937 | +0.196701 |
| Qwen3 GRPO teacher | Prefix128 pure SFT - Plain OPD | -0.010417 | +0.000000 | -0.075000 | -0.028472 |
| Qwen3 GRPO teacher | Prefix128 SFT(0.1)+suffix OPD - Plain OPD | +0.056250 | +0.029167 | +0.043750 | +0.043056 |
| Qwen3 GRPO teacher | Prefix128 SFT(0.1)+suffix OPD - Prefix128 pure SFT | +0.066667 | +0.029167 | +0.118750 | +0.071528 |
| Qwen3 no-think teacher | Prefix128 pure SFT - Plain OPD | -0.002083 | +0.000000 | +0.043750 | +0.013889 |
| Qwen3 no-think teacher | Full-response SFT - Plain OPD | -0.033333 | -0.022917 | -0.073438 | -0.043229 |
| Qwen3 no-think teacher | Prefix128 pure SFT init + plain OPD - Plain OPD | -0.002083 | +0.002083 | +0.012500 | +0.004167 |
| Qwen3 no-think teacher | Batch mixed full-SFT + plain OPD - Plain OPD | -0.006250 | +0.006250 | +0.006250 | +0.002083 |
| Qwen3 no-think teacher | Prefix128 SFT(0.1)+suffix OPD - Batch mixed full-SFT + plain OPD | +0.014584 | +0.010416 | +0.104687 | +0.043229 |
| Qwen3 no-think teacher | Prefix128 SFT(0.1)+suffix OPD - Plain OPD | +0.008334 | +0.016666 | +0.110937 | +0.045312 |
| Qwen3 no-think teacher | Prefix128 SFT(0.1)+suffix OPD - Prefix128 pure SFT | +0.010417 | +0.016666 | +0.067187 | +0.031423 |

Summary:

- Prefix-only SFT is not sufficient in either model family.
- Full-response SFT is unstable and underperforms in both families under the current setup.
- Prefix-SFT initialization followed by plain OPD recovers most of full OPD on DeepSeek/JustRL, but does not beat full OPD or the joint method.
- Prefix-SFT initialization followed by plain OPD provides only a small gain on Qwen3 no-think and remains far below the joint method.
- The joint setting, `TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD`, is the strongest n=1 variant in both families.
- DeepSeek/JustRL remains much stronger in absolute score than Qwen3 in this setup.
- Qwen3 GRPO teacher is substantially stronger than Qwen3 no-think base teacher.
- The current best DeepSeek/JustRL n=1 result is `TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD`, Avg `0.550694`.
- The current best Qwen3 result is `Qwen3-4B-Base-GRPO TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD`, Avg `0.224306`.

### Mechanism Insight: Raw Top-K Overlap Is Not Enough

The Qwen3 no-think setting provides the clearest mechanism case study. Prior OPD analysis emphasizes top-k overlap as a condition for effective distillation: teacher guidance should lie in the student's locally reachable support. Our diagnostics suggest a sharper condition:

`Top-k overlap measures whether the teacher signal is locally accessible, but intersection probability mass measures whether it is actually learnable.`

In the Qwen3 no-think run, plain OPD has higher raw top-k overlap than our method, but much lower probability mass on the overlapped support:

| Method | Top-k overlap last | Student p-mass on intersection last | Teacher p-mass on intersection last | Eval Avg |
| --- | ---: | ---: | ---: | ---: |
| Plain OPD | 0.7927 | 0.3465 | 0.3625 | 0.106771 |
| Our method: TeacherPrefix128 + SFT(prefix, 0.1) + suffix OPD | 0.5782 | 0.9021 | 0.9041 | 0.152083 |

Interpretation:

- Raw top-k overlap is a set-level metric: it asks how many candidate tokens appear in both the student and teacher top-k lists.
- Intersection p-mass is a probability-mass metric: it asks how much probability both models assign to the shared candidate region.
- Plain OPD has many overlapping candidate tokens, but those overlapping tokens carry only about 35% of each model's probability mass. This means the apparent support overlap is not where either model places most of its confidence.
- Our method has lower raw overlap, but the overlapping support carries about 90% of both student and teacher probability mass. This indicates a smaller but much more important shared teachable region.
- Therefore, the gain is not simply from increasing teacher-student top-k overlap. Teacher-prefix conditioning reshapes the student suffix trajectory so that the high-probability regions of teacher and student concentrate on a shared teachable support.

This extends the overlap insight in Rethinking OPD:

`OPD is effective when teacher and student concentrate probability mass on a shared high-probability support, not merely when their top-k supports overlap.`

Suggested paper wording:

`Prior work has emphasized top-k overlap as a prerequisite for effective OPD, arguing that teacher guidance must lie within the student's locally reachable support. We find that this condition is not sufficient. In our Qwen3 no-think setting, plain OPD exhibits higher raw top-k overlap than our method, yet assigns only a small fraction of both student and teacher probability mass to the overlapped support. In contrast, teacher-prefix conditioning yields lower raw overlap but concentrates over 90% of both policies' probability mass on the shared support, leading to substantially better downstream accuracy. This suggests that the key determinant is not raw support overlap, but shared high-probability support.`

Related reward-curve figure:

`analysis_plots/qwen3_nothink_reward_curves.png`
