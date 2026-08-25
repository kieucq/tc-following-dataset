#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J WRF-TC-Prep
#SBATCH -p general
#SBATCH -A r00043
#SBATCH --mem=64G
module load python/gpu/3.12.5
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$script_dir" || exit 1
set -x

# ------------------------------------------------------------------------------
# May need to set up a virtual environment if you haven't already. Uncomment the 
# following lines and set VIRTUAL_ENV_PATH if needed.
# ------------------------------------------------------------------------------
#venv_dir="${VIRTUAL_ENV_PATH:-$script_dir/../.venv}"
#if [ ! -f "$venv_dir/bin/activate" ]; then
#    echo "Python environment not found: $venv_dir" >&2
#    echo "Create it as described in README.md or set VIRTUAL_ENV_PATH." >&2
#    exit 1
#fi
#source "$venv_dir/bin/activate"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
config_file="${CONFIG_FILE:-$script_dir/config.yaml}"
#pipeline="${PIPELINE:-cmip6}"
pipeline="idealize"
indir="/N/project/Typhoon-deep-learning/data/tc-wrf/"
outdir="/N/u/ckieu/BigRed200/codex/tc-following-dataset/output/idealize/"
case "$pipeline" in
    cmip5)
        step1_script="step1_cropping_cmip5.py"
        step2_script="step2_merging_cmip5.py"
        ;;
    cmip6)
        step1_script="step1_cropping_cmip6.py"
        step2_script="step2_merging_cmip6.py"
        ;;
    idealize)
        step1_script="step1_cropping_idealized.py"
        step2_script="step2_merging_idealized.py"
        ;;
    *)
        echo "Unknown PIPELINE '$pipeline'; use cmip5, cmip6, or idealize." >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# Step toggles: set to 1 to run, 0 to skip
# ------------------------------------------------------------------------------
run_step1=0
run_step2=1

# ------------------------------------------------------------------------------
# 1) Crop raw WRF outputs
# ------------------------------------------------------------------------------
if [ "$run_step1" -eq 1 ]; then
    python "$step1_script" \
        --config "$config_file" \
        --data_dir "$data_dir" --workdir "${outdir}/level_1_data/"
fi

# ------------------------------------------------------------------------------
# 2) Process cropped .nc files into NumPy arrays
# ------------------------------------------------------------------------------
if [ "$run_step2" -eq 1 ]; then
    python "$step2_script" \
        --config "$config_file" \
        --outdir "${outdir}/level_2_data"
fi

echo "All requested steps completed."
