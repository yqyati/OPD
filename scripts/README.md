# Script Layout

This directory groups runnable project scripts by purpose.

- `train/`: OPD, GRPO, top50, random50, and other training launch scripts.
- `validate/`: checkpoint/model validation launch scripts.
- `data_filter/`: prompt scoring, shard merging, and data selection utilities.
- `infer/`: standalone inference and rollout utilities.
- `val/`: JustRL-style evaluation data and grading/generation code.

Root-level training scripts were moved here to keep the project root focused on
source, datasets, outputs, and documentation.

Validation uses two shared entry points:

- `validate/eval_model.sh`: evaluate an existing HuggingFace-format model.
- `validate/eval_checkpoint.sh`: merge a verl FSDP actor checkpoint, then
  evaluate the merged model.

Only one preset wrapper is kept for the current mainline run:
`validate/validate_opd_top50_31744_4gpu.sh`. Use the shared entry points
directly for other models or checkpoints.
