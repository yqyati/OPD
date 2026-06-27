# Soft Teachability Sampling for On-Policy Distillation

## Motivation

On-policy distillation (OPD) requires more than high-quality data. A useful OPD
sample should provide a meaningful teacher-student learning signal while still
preserving broad coverage of the original training distribution.

Our current observations suggest a tension between these two requirements:

- **Random50** preserves coverage and is already a strong baseline.
- **Top50 by prompt teachability score** over-concentrates on high-score samples
  and does not outperform Random50.
- **Full step140** is weaker than both Top50 and Random50, likely because
  `data.shuffle=False` makes it a prefix of the parquet order rather than a
  representative half-dataset.

This motivates a selection method that interpolates between random sampling and
hard top-score selection.

## Core Idea

Random sampling ignores teacher-student teachability. Hard top-score sampling
ignores distribution coverage. We propose **Soft Teachability Sampling (STS)**:
a temperature-controlled, quantile-stratified sampler that softly biases data
selection toward higher-teachability samples while keeping coverage across all
teachability regions.

In one sentence:

> STS preserves coverage like random sampling, but assigns more probability to
> samples with stronger OPD teachability signal.

## Teachability Score

Let each training prompt `x_i` have a cheap, precomputed teachability score
`s_i`. For the current prompt-only prototype, we use:

```text
s_i = overlap_i * max(H_student(x_i) - H_teacher(x_i), 0)
```

where:

- `H_student(x_i)` is the student entropy on selected prompt tokens.
- `H_teacher(x_i)` is the teacher entropy on the same prompt tokens.
- `overlap_i` is the student-teacher top-k overlap on prompt tokens.

This score should not be used as a hard top-k ranking criterion. Instead, STS
uses it only to softly bias sampling.

## Method

Given a dataset `D = {x_i}_{i=1}^N`, target subset size `M`, number of score
bins `K`, and temperature/bias parameter `alpha`:

1. Sort samples by teachability score `s_i`.
2. Split samples into `K` equal-size quantile bins:

```text
B_1, B_2, ..., B_K
```

where `B_1` contains the lowest-score samples and `B_K` contains the
highest-score samples.

3. Allocate sample counts to bins using a soft exponential bias:

```text
w_k = exp(alpha * k)
m_k = round(M * w_k / sum_j w_j)
```

4. Uniformly sample `m_k` examples from each bin `B_k`.
5. Concatenate all sampled examples as the selected subset:

```text
D_STS = union_k Sample(B_k, m_k)
```

## Interpolation View

STS naturally interpolates between random sampling and hard top-score selection:

```text
alpha = 0       -> stratified random sampling
alpha moderate  -> soft teachability-biased sampling
alpha -> large  -> approximates top-score selection
```

This gives a clean ablation axis:

```text
Random50      : alpha = 0
STS50         : alpha = 0.5 or 1.0
Top50         : hard top-score selection
```

## Why This Is More Principled Than Top50

Hard top-score selection assumes that larger teachability score is always
better. Our Random50 result contradicts this assumption. High-score-only
selection can lose important coverage and overfit to a narrow region of the
training distribution.

STS makes a weaker and more defensible assumption:

> Higher-teachability samples should be sampled more often, but all
> teachability regions should remain represented.

This is aligned with OPD, where effective training requires both:

- useful teacher-student token-level discrepancy;
- diverse reasoning patterns and problem types.

## Proposed Experimental Matrix

All runs should use the same training/evaluation configuration:

```text
student: DeepSeek-R1-Distill-Qwen-1.5B
teacher: JustRL-DeepSeek-1.5B
estimator: token_reward_direct
n: 4
lr: 1e-5
max_response_length: 7168
final_eval_max_tokens: 31744
eval_n: 16
enable_thinking: False
```

Primary comparison:

| Run | Data size | Selection | Purpose |
| --- | ---: | --- | --- |
| Full step140 | ~50% steps | parquet prefix | step-matched full baseline |
| Random50 seed1/2 | 8958 | uniform random | strong coverage baseline |
| Top50 | 8958 | hard top score | pure teachability baseline |
| STS50 alpha=0.5 | 8958 | soft teachability | main method |
| STS50 alpha=1.0 | 8958 | stronger soft bias | sensitivity check |
| Full final | 17917 | full data | upper training-budget baseline |

Current known results:

| Run | AIME24 | AIME25 | AMC23 | Avg |
| --- | ---: | ---: | ---: | ---: |
| Student baseline | 0.283333 | 0.237500 | 0.732812 | 0.417882 |
| Teacher | 0.516667 | 0.372917 | 0.879687 | 0.589757 |
| Full OPD lr=1e-5 step140 | 0.429167 | 0.341667 | 0.842187 | 0.537674 |
| Top50 OPD lr=1e-5 step139 | 0.462500 | 0.333333 | 0.845313 | 0.547049 |
| Random50 seed1 lr=1e-5 step139 | 0.506250 | 0.322917 | 0.845313 | 0.558160 |
| Full OPD lr=1e-5 step260 | 0.466667 | 0.318750 | 0.878125 | 0.554514 |

## Expected Outcomes

The target result is:

```text
STS50 > Random50 mean >= Top50
```

or, more conservatively:

```text
STS50 has better average performance or lower seed variance than Random50.
```

Interpretation:

- If `STS50 > Random50`, the teachability score is useful when applied softly.
- If `STS50 ~= Random50`, prompt-level teachability is weak, and future methods
  should use response-level or online training signals.
- If `STS50 < Random50`, the current prompt score is harmful even under soft
  sampling, and should not be used for OPD data selection.

## Paper Framing

The central claim should not be that prompt entropy alone identifies the best
OPD samples. The stronger and cleaner claim is:

> OPD data selection requires balancing teacher-student teachability with
> distribution coverage. Hard selection by teachability over-concentrates the
> dataset, while soft teachability sampling preserves coverage and improves data
> efficiency.

Possible method names:

- Soft Teachability Sampling (STS)
- Coverage-Preserving Teachability Sampling (CPTS)
- Temperature-Controlled Teachability Sampling

Preferred name:

```text
Soft Teachability Sampling (STS)
```

## Next Implementation Steps

1. Use the existing `opd_prompt_scores.parquet`.
2. Generate STS subsets:

```text
datasets/opd_prompt_filter/opd_prompt_score_sts50_alpha0.5_seed1.parquet
datasets/opd_prompt_filter/opd_prompt_score_sts50_alpha1.0_seed1.parquet
```

3. Create matching training scripts:

```text
opd_sts50_alpha0.5_seed1.sh
opd_sts50_alpha1.0_seed1.sh
```

4. Keep all training/evaluation settings identical to Top50 and Random50.
5. Compare against Random50 seed1/seed2 before making claims.

