#!/bin/bash
# train_llm.slurm - submit the QLoRA fine-tune to SuperMike-III with: sbatch train_llm.slurm
#
# FILL IN THREE THINGS before submitting (marked <<< ... >>>):
#   1. your allocation name (-A)      -> Elias's Startup allocation
#   2. the module names               -> confirm with `module avail python` / `module avail cuda`
#   3. the paths                       -> your env and your project folder
#
# ONE-TIME SETUP (do this once on a login node, NOT in this script):
#   module load <python module>
#   python -m venv ~/crashenv
#   source ~/crashenv/bin/activate
#   pip install unsloth
# Then submit this job.

#SBATCH -J crashllm                 # job name
#SBATCH -A <<<your_allocation>>>     # allocation to charge (required)
#SBATCH -p gpu                      # GPU partition on SuperMike-III
#SBATCH -N 1                        # one node
#SBATCH -n 8                        # 8 CPU cores  (~32 GB RAM at 4 GB/core; bump if OOM)
#SBATCH --gres=gpu:1                # one A100
#SBATCH -t 04:00:00                 # walltime (training is fast; confirm gpu-partition max)
#SBATCH -o crashllm-%j.out          # stdout -> file (%j = job id)
#SBATCH -e crashllm-%j.err          # stderr -> file

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"

# --- environment ---
module purge
module load <<<python_module>>>     # e.g. python/3.11-anaconda  (module avail python)
module load <<<cuda_module>>>       # e.g. cuda/12.4             (module avail cuda)

source ~/crashenv/bin/activate      # the venv you made in one-time setup

cd <<<path/to/4700Project/crashlab>>>   # where train_llm.py and llm_data/ live

nvidia-smi                          # log which GPU we got, sanity check

# --- run the fine-tune ---
python train_llm.py

echo "finished $(date)"
