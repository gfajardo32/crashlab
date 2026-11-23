#!/bin/bash
#SBATCH --job-name=crashlab-llm-v32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=crashlab-llm-%x-%j.out
#SBATCH --error=crashlab-llm-%x-%j.err

set -euo pipefail

: "${CRASHLAB_VENV:?Set CRASHLAB_VENV to the absolute path of the GPU Python virtual environment}"
cd "${CRASHLAB_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this job from the crashlab directory}}"

if [[ -n "${CRASHLAB_PYTHON_MODULE:-}" ]]; then
    module load "$CRASHLAB_PYTHON_MODULE"
fi
if [[ -n "${CRASHLAB_CUDA_MODULE:-}" ]]; then
    module load "$CRASHLAB_CUDA_MODULE"
fi

# Use the caller's existing CUDA-enabled Unsloth / TRL environment.
source "$CRASHLAB_VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
nvidia-smi

case "${1:-}" in
    cv|final) exec python -m llm_arm.train "$@" ;;
    held-out) shift; exec python -m evaluate_heldout "$@" ;;
    *) echo 'Usage: sbatch --account ALLOCATION llm_arm/train.slurm {cv|final|held-out} ...' >&2; exit 2 ;;
esac
