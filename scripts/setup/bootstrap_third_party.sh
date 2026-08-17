#!/usr/bin/env bash
# Recreate the locally ignored third_party dependencies used by OPD/IF-RLVR.
#
# This script is deliberately conservative: it pins the external source and
# refuses to alter an existing clone at a different revision.  It never
# installs Python packages and never downloads model/data training corpora.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
THIRD_PARTY_DIR="${ROOT}/third_party"
OPEN_INSTRUCT_DIR="${THIRD_PARTY_DIR}/open-instruct-ifrlvr"
NLTK_DIR="${THIRD_PARTY_DIR}/nltk_data"

OPEN_INSTRUCT_REPO="https://github.com/allenai/open-instruct.git"
OPEN_INSTRUCT_COMMIT="1049dde2fdf36fec9d220bde57f42df15c02e029"
PYTHON_BIN=${PYTHON_BIN:-python3}

usage() {
    cat <<'EOF'
Usage: bash scripts/setup/bootstrap_third_party.sh [--skip-nltk]

Recreates the ignored third_party dependencies required by the IF-RLVR code:
  * open-instruct-ifrlvr at a pinned allenai/open-instruct commit
  * NLTK punkt and punkt_tab under third_party/nltk_data

The command needs outbound access to GitHub and the NLTK package host. It does
not install pip packages; use the project's existing Python environment.
EOF
}

SKIP_NLTK=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-nltk) SKIP_NLTK=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -e "${OPEN_INSTRUCT_DIR}" ]]; then
    [[ -d "${OPEN_INSTRUCT_DIR}/.git" ]] || {
        echo "Expected a Git clone at ${OPEN_INSTRUCT_DIR}, but it is not one." >&2
        exit 1
    }
    CURRENT_COMMIT=$(git -C "${OPEN_INSTRUCT_DIR}" rev-parse HEAD)
    [[ "${CURRENT_COMMIT}" == "${OPEN_INSTRUCT_COMMIT}" ]] || {
        echo "Refusing to modify existing open-instruct clone." >&2
        echo "expected=${OPEN_INSTRUCT_COMMIT}" >&2
        echo "actual=${CURRENT_COMMIT}" >&2
        exit 1
    }
    echo "Reusing pinned Open-Instruct clone: ${OPEN_INSTRUCT_DIR} @ ${CURRENT_COMMIT}"
else
    echo "Cloning Open-Instruct at pinned commit ${OPEN_INSTRUCT_COMMIT}"
    git clone "${OPEN_INSTRUCT_REPO}" "${OPEN_INSTRUCT_DIR}"
    git -C "${OPEN_INSTRUCT_DIR}" checkout --detach "${OPEN_INSTRUCT_COMMIT}"
fi

if [[ "${SKIP_NLTK}" == true ]]; then
    echo "Skipping NLTK data bootstrap by request."
    exit 0
fi

echo "Ensuring NLTK punkt and punkt_tab exist under ${NLTK_DIR}"
"${PYTHON_BIN}" - "${NLTK_DIR}" <<'PY'
import sys
from pathlib import Path

import nltk

target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
for package in ("punkt", "punkt_tab"):
    ok = nltk.download(package, download_dir=str(target), quiet=False, raise_on_error=True)
    if not ok:
        raise RuntimeError(f"NLTK downloader returned false for {package}")
print(f"NLTK data ready: {target}")
PY

echo "Bootstrap complete."
echo "For IF-RLVR launchers, export NLTK_DATA=${NLTK_DIR} (existing launchers already do this)."
