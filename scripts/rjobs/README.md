# Rjob Script Layout

The `/yangqingyu` root is reserved for current, directly invoked rjob entry
points. Completed experiment launchers are retained here as reproducibility
snapshots instead of remaining in one flat directory.

## Canonical templates

New MOPD pipelines must start from the templates in `scripts/mopd/templates/`:

- `plain_mopd_train_merge_eval.bash.template`
- `prefix_mopd_train_merge_eval.bash.template`
- `rollout_full_sft_plain_mopd_eval.bash.template`

Do not create another launcher by copying an arbitrary historical experiment.
Keep one public train-to-eval entry point and put reusable implementation code
under `workspace/OPD/scripts/`.

## Archive

- `archive/base17b/`: completed Qwen3-1.7B Base OPD ablations.
- `archive/pragmatic/`: completed pragmatic 8B plain/prefix/Full-SFT MOPD runs.
- `archive/completed/`: one-off completion and evaluation launchers.
- `archive/other/`: scripts belonging to other experiment families.

Archived files retain their original basenames. They are not current launch
commands, but their internal launcher references are kept valid.

## Protected current entry

The current same-origin Full-SFT pipeline remains at its stable path:

```text
/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/rjob_q8b_sameorigin_full7168sft_then_plain_mopd_r7168_train_eval_8gpu.bash
```

Its basename and absolute path must not be changed.
