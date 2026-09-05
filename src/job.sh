#!/bin/bash
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -J WRF-TC-Prep
#SBATCH -p general
#SBATCH -A r00043
#SBATCH --mem=64G
conda deactivate >& /dev/null
module load python/gpu/3.12.5 >& /dev/null
repo_dir="/N/u/ckieu/BigRed200/codex/tc-following-dataset/"
script_dir="${repo_dir}/src"
cd "$script_dir" || exit 1

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
# Step toggles for workflow: set to 1 to run, 0 to skip
# ------------------------------------------------------------------------------
run_step1=1
run_step2=1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
config_file="${CONFIG_FILE:-$script_dir/config.yaml}"
pipeline="$1"
if [ -z "$pipeline" ]; then
    echo "Error: Missing required argument <pipeline>" >&2
    echo "Usage: $0 <pipeline>, where <pipeline>: cmip5, cmip6, or idealize" >&2
    exit 1
fi
echo "Using pipeline: $pipeline"
set -x
case "$pipeline" in
    cmip5)
        step1_script="step1_cropping_cmip5.py"
        step2_script="step2_merging_cmip5.py"
        default_track_file="$repo_dir/input/best_track/baseline_track.txt"
        default_data_dir="$repo_dir/input/cmip5"
        default_level_1_dir="$repo_dir/output/cmip5/level_1_data"
        default_level_2_dir="$repo_dir/output/cmip5/level_2_data"
        uses_track_file=1
        ;;
    cmip6)
        step1_script="step1_cropping_cmip6.py"
        step2_script="step2_merging_cmip6.py"
        default_track_file="$repo_dir/input/best_track/baseline_track.txt"
        default_data_dir="$repo_dir/input/cmip6"
        default_level_1_dir="$repo_dir/output/cmip6/level_1_data"
        default_level_2_dir="$repo_dir/output/cmip6/level_2_data"
        uses_track_file=1
        ;;
    idealize)
        step1_script="step1_cropping_idealized.py"
        step2_script="step2_merging_idealized.py"
        default_track_file=""
        default_data_dir="$repo_dir/input/wrf"
        default_level_1_dir="$repo_dir/output/idealize/level_1_data"
        default_level_2_dir="$repo_dir/output/idealize/level_2_data"
        uses_track_file=0
        ;;
    *)
        echo "Unknown PIPELINE '$pipeline'; use cmip5, cmip6, or idealize." >&2
        exit 1
        ;;
esac

# Dataset paths live here rather than in config.yaml. These environment
# variables allow a submission to override any path without editing this file.
track_file="${TRACK_FILE:-$default_track_file}"
data_dir="${DATA_DIR:-$default_data_dir}"
level_1_dir="${LEVEL1_DIR:-$default_level_1_dir}"
level_2_dir="${LEVEL2_DIR:-$default_level_2_dir}"

# ------------------------------------------------------------------------------
# 1) Crop raw WRF outputs
# ------------------------------------------------------------------------------
if [ "$run_step1" -eq 1 ]; then
    step1_args=(
        --config "$config_file"
        --data_dir "$data_dir"
        --workdir "$level_1_dir"
    )
    if [ "$uses_track_file" -eq 1 ]; then
        step1_args+=(--track_file "$track_file")
    fi
    python "$step1_script" "${step1_args[@]}"
fi

# ------------------------------------------------------------------------------
# 2) Process cropped .nc files into NumPy arrays
# ------------------------------------------------------------------------------
if [ "$run_step2" -eq 1 ]; then
    python "$step2_script" \
        --config "$config_file" \
        --indir "$level_1_dir" \
        --outdir "$level_2_dir"
fi

echo "All requested steps completed."
