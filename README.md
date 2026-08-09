# Tropical Cyclone Following Dataset

## Description

This repository converts dynamically downscaled CMIP6 Weather Research and
Forecasting (WRF) output and tropical-cyclone (TC) tracks into storm-following,
machine-learning-ready datasets. The workflow preserves the evolving
three-dimensional storm environment while removing most of the unrelated WRF
domain.

The preprocessing pipeline has two stages:

1. **Storm-centered cropping (L1).** Daily detected TC positions are linearly
   interpolated to the WRF output interval. Each full-domain WRF file is
   destaggered onto the mass grid, cropped around the nearest grid point to the
   storm center, and grouped by storm ID in NetCDF format. Local **x**/**y**
   indices describe the moving crop; the true grid latitude/longitude and
   storm-center coordinates are retained.
2. **Temporal sample generation (L2).** Selected surface and vertical-level
   variables are read from L1 files and assembled into fixed-length sequences.
   The primary **X** array has shape *(samples, frames, height, width,
   channels)*. The metadata-aware CMIP and idealized mergers write per-frame
   **Z** arrays plus storm/experiment identifiers, lifetimes, and chunk starts.

The reference configuration uses 100 × 100 crops, five consecutive six-hourly
frames (24 hours) and a four-timestep sample stride. CMIP and idealized variable
lists are configured independently. Selecting
**PH** or **PHB** at a vertical level reconstructs total geopotential as
**PH + PHB**.

## Directory structure

~~~text
.
├── docs/
│   └── draft.tex             # Dataset manuscript and scientific description
├── input/
│   ├── best_track/           # TC track ASCII files
│   ├── cmip6/                # Raw CMIP6-forced WRF output used by Stage 1
│   └── wrf/                  # Idealized WRF experiments
├── output/
│   ├── cmip6/                # CMIP L1/L2 links or directories
│   └── idealized/            # Idealized L1/L2 links or directories
├── src/
│   ├── config.yaml           # Paths and all two-stage processing parameters
│   ├── config_utils.py       # YAML loading and path resolution
│   ├── extractor.py          # Extractor for the idealized WRF experiments
│   ├── job_slurm.sh          # Selectable CMIP/idealized SLURM launcher
│   ├── step1_cropping_cmip5.py
│   ├── step1_cropping_cmip6.py
│   ├── step1_cropping_idealized.py
│   ├── step2_merging_cmip5.py
│   ├── step2_merging_cmip6.py
│   └── step2_merging_idealized.py
├── requirements.txt
└── README.md
~~~

The checked-in **input/** directories are data locations rather than bundled
datasets. On the Indiana University system, the two output entries may be links
to project storage. They can also be replaced with ordinary local directories.

## How to install

Python 3.11 is recommended and is loaded by the supplied Big Red 200 SLURM job.
Replace **REPOSITORY_URL** below with this repository's published URL:

~~~bash
git clone REPOSITORY_URL tc-following-dataset
cd tc-following-dataset

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The core dependencies are NumPy, xarray, netCDF4, and PyYAML. A SLURM workload
manager is needed only for batch submission; both Python stages can be run
directly on any Linux system with access to the input data.

On Big Red 200, create the same environment with the cluster module used by the
job:

~~~bash
module load python/3.11.13
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

## Configuration

Edit **src/config.yaml** before running. CMIP scripts read **cmip.step1** or
**cmip.step2**; idealized scripts read **idealize.step1** or
**idealize.step2**. Relative paths are resolved from the configuration file.

| Section | Setting | Meaning |
| --- | --- | --- |
| cmip.step1 | track_file | ASCII TC track file |
| both step1 blocks | data_dir, workdir | Raw input and L1 output directories |
| both step1 blocks | imsize_x, imsize_y | Crop dimensions in grid points |
| cmip.step1 | interpolation_steps, timestep_hours | Track interpolation and WRF interval |
| idealize.step1 | experiment_wrf, x_resolution | Experiment folders and filename token |
| both step1 blocks | output_filename_template | L1 name with {storm_id} or {experiment_id} |
| both step2 blocks | indir, outdir | L1 input and L2 output directories |
| both step2 blocks | frames, sample_stride, var_levels | Temporal and channel selection |
| both step2 blocks | *_filename | Explicit name for each generated NumPy array |

Stage-2 filenames may contain **{frames}** and **{variables}**. For example,
**cmip_{frames}f_{variables}v_X.npy** becomes **cmip_5f_14v_X.npy**. Names must
be unique and may not contain directory components.

Variable codes use a two-digit zero-based model-level index, such as **U05**,
**T23**, or **QVAPOR10**. Codes ending in **m**, such as **U10m**, select the WRF
surface variable after removing the suffix (**U10**). Other surface variables,
such as **SST** and **LANDMASK**, are written without a level. Names must exist
in the L1 WRF datasets.

The track reader expects whitespace-separated numeric rows with at least:

~~~text
storm_id  year  day_of_year  center_longitude  center_latitude
~~~

Rows are grouped by **storm_id**. The default four interpolation steps and
six-hour interval turn adjacent daily track points into 00, 06, 12, and 18 UTC
positions.

## How to run

### Run locally

After configuring the paths, run the desired pipeline from the repository root.
For the metadata-aware CMIP workflow:

~~~bash
python src/step1_cropping_cmip6.py --config src/config.yaml
python src/step2_merging_cmip6.py --config src/config.yaml
~~~

For the idealized workflow:

~~~bash
python src/step1_cropping_idealized.py --config src/config.yaml
python src/step2_merging_idealized.py --config src/config.yaml
~~~

Each YAML value can be overridden for a one-off run. For example:

~~~bash
python src/step2_merging_cmip6.py --config src/config.yaml --frames 9 --x_filename my_cmip_X.npy
~~~

Stage 1 writes one configured NetCDF name per storm or experiment. Stage 2 uses
the explicit X, Z, ID, lifetime, and chunk-start filenames from its section.

### Run with SLURM

Update the **#SBATCH** account, partition, memory, and wall-time directives in
**src/job_slurm.sh** for the target cluster. The script currently loads
**python/3.11.13**. Set **run_step1** and **run_step2** near the top of the
script to 1 or 0, then submit:

~~~bash
sbatch src/job_slurm.sh
~~~

By default, the supplied job selects **cmip6**, skips Stage 1, and runs Stage 2.
Select another script family with **PIPELINE**:

~~~bash
PIPELINE=idealize sbatch src/job_slurm.sh
PIPELINE=cmip5 sbatch src/job_slurm.sh
~~~

To use another configuration without editing the job file:

~~~bash
CONFIG_FILE=/absolute/path/to/config.yaml sbatch src/job_slurm.sh
~~~

The job activates **.venv** at the repository root. Set **VIRTUAL_ENV_PATH** if
the environment is elsewhere:

~~~bash
VIRTUAL_ENV_PATH=/absolute/path/to/venv sbatch src/job_slurm.sh
~~~

## Reference

The scientific motivation, data construction, variable selection, and
validation are described in **docs/draft.tex**:

> Luong, K., and C. Kieu, 2026: *High-Resolution Temporal Dataset of Tropical
> Cyclone Three-Dimensional Structure for Limited-Area Machine Learning Model
> Applications* (manuscript in preparation).

The manuscript citation, publication dates, and some WRF/CMIP6 experiment
details are still marked as pending in the draft. Please use the final published
citation when it becomes available.

## Contact

**Chanh Kieu**  
Department of Earth and Atmospheric Sciences, Indiana University Bloomington  
[ckieu@iu.edu](mailto:ckieu@iu.edu)
