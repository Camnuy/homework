#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-homework2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Updating existing conda env: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_diffusion.txt

echo
echo "Done."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Read next: docs/CODEX_SERVER_HANDOFF_zh.md"
