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
