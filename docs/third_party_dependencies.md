# Third-party dependencies

The repository intentionally does **not** version `third_party/`.  It contains
locally cloned repositories and downloader caches rather than OPD source code.

Recreate the dependencies required by the IF-RLVR code with:

```bash
bash scripts/setup/bootstrap_third_party.sh
```

The bootstrap script pins Open-Instruct exactly to:

```text
repository: https://github.com/allenai/open-instruct.git
commit:     1049dde2fdf36fec9d220bde57f42df15c02e029
directory:  third_party/open-instruct-ifrlvr
```

It also downloads only the NLTK resources currently needed by the project into
`third_party/nltk_data`:

```text
punkt
punkt_tab
```

The existing IF-RLVR launchers set `NLTK_DATA` to that directory automatically.

`third_party/ifeval` is an old local copy that is not referenced by the current
training or evaluation scripts.  It is intentionally not recreated by the
bootstrap script: its nested Git remote points to this OPD repository rather
than a standalone upstream source.  The project’s current IFEval evaluation
uses the held-out data under `datasets/eval/IFEval` and the project-local
evaluator instead.
