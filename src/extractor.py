import glob
import re
import os
import numpy as np
import xarray as xr
import argparse

print('Starting')

def parse_args():
    parser = argparse.ArgumentParser(description='Extract WRF data without eid and m_id.')
    parser.add_argument('-ix', '--imsize_variables', type=int, nargs=2, default=[100, 100],
                        help='Image size for variable extraction (width height)')
    parser.add_argument('-r', '--root', type=str, default='/N/slate/kmluong/PROJECT2/WRF',
                        help='Output directory')
    parser.add_argument('-b', '--wrf_base', type=str, default="/N/project/Typhoon-deep-learning/data/tc-wrf/",
                        help='Base directory to read from')
    parser.add_argument('-vl', '--var_levels', type=str, nargs='+', 
                        default=["U10m", "V10m", "SST", "LANDMASK",
                                 "U14", "V14", "U03", "V03",
                                 "T12", "QVAPOR05", "PHB05"],
                        help=("List of variable-level codes. Format e.g. U01, V02, QVAPOR03 "
                              "or PSFC for surface. Levels are parsed and halved (ceil)."))
    parser.add_argument('-ew', '--experiment_wrf', type=str, nargs='+', 
                        default=["exp_02km_m01", "exp_02km_m02", "exp_02km_m03", "exp_02km_m04",
                                 "exp_02km_m05", "exp_02km_m06", "exp_02km_m07", "exp_02km_m08",
                                 "exp_02km_m09", "exp_02km_m10"],
                        help='WRF experiment folders to process (inputs)')
    parser.add_argument('-xd', '--X_resolution', type=str, default='d01', 
                        help='X resolution string in filename (e.g. d01)')
    return parser.parse_args()


def extract_input_variables(ds1, imsize1=(64, 64), var_levels=None):
    """
    Extract core variables from ds1 only (inputs).
    """
    if var_levels is None:
        var_levels = [('U', 1), ('U', 2), ('U', 3),
                      ('V', 1), ('V', 2), ('V', 3),
                      ('T', 1), ('T', 2), ('T', 3),
                      ('QVAPOR', 1), ('QVAPOR', 2), ('QVAPOR', 3),
                      ('PSFC', None)]

    mid_x1 = ds1.sizes['west_east'] // 2
    mid_y1 = ds1.sizes['south_north'] // 2
    start_x1 = mid_x1 - imsize1[0] // 2
    end_x1   = mid_x1 + imsize1[0] // 2
    start_y1 = mid_y1 - imsize1[1] // 2
    end_y1   = mid_y1 + imsize1[1] // 2

    def select_variable(ds, var_name, lev):
        try:
            if lev is not None and 'bottom_top' in ds[var_name].dims:
                selected = ds[var_name].isel(bottom_top=lev)
            elif lev is not None and 'bottom_top_stag' in ds[var_name].dims:
                selected = ds[var_name].isel(bottom_top_stag=lev)
            elif lev is not None and 'lev' in ds[var_name].coords:
                selected = ds[var_name].sel(lev=lev)
            else:
                selected = ds[var_name]
        except Exception:
            selected = ds[var_name]

        if 'south_north' in selected.dims:
            selected = selected.isel(south_north=slice(start_y1, end_y1))
        elif 'south_north_stag' in selected.dims:
            selected = selected.isel(south_north_stag=slice(start_y1, end_y1))
        if 'west_east_stag' in selected.dims:
            selected = selected.isel(west_east_stag=slice(start_x1, end_x1))
        elif 'west_east' in selected.dims:
            selected = selected.isel(west_east=slice(start_x1, end_x1))
        return selected

    arrays = []
    for var, lev in var_levels:
        if var in ('PH', 'PHB'):
            try:
                ph = select_variable(ds1, 'PH', lev)
                phb = select_variable(ds1, 'PHB', lev)
                selected_data = ph + phb
            except Exception:
                selected_data = select_variable(ds1, var, lev)
        else:
            selected_data = select_variable(ds1, var, lev)

        if 'Time' not in selected_data.dims:
            selected_data = selected_data.expand_dims(Time=[0])
        if selected_data.dims[0] != 'Time':
            selected_data = selected_data.transpose('Time', ...)

        arr = np.squeeze(selected_data.values, axis=0)
        arrays.append(arr)

    if arrays:
        min_h = min(a.shape[-2] for a in arrays)
        min_w = min(a.shape[-1] for a in arrays)
        if min_h <= 0 or min_w <= 0:
            raise ValueError("Invalid spatial dimensions after slicing.")
        arrays = [a[..., :min_h, :min_w] for a in arrays]
    final_result = np.stack(arrays, axis=0)
    final_result = final_result[np.newaxis, ...]  # shape: (1, channels, height, width)
    return final_result

def natural_sort_key(s):
    """
    Produce a sort key that sorts strings naturally.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def collect_x_files(exp_folder, x_res):
    """
    Given an experiment folder, collect all files whose names contain the X resolution substring.
    Returns a list of x files sorted naturally by basename (chronological order in filename).
    """
    all_files = glob.glob(os.path.join(exp_folder, "*"))
    x_files = [f for f in all_files if x_res in os.path.basename(f)]
    x_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
    return x_files


def process_experiments(exp_list, base_path, imsize_x, root, x_res, var_levels):
    """
    Process a list of experiment folders. For each folder, collect X files by looking for the X resolution
    substring in the filename. Process each file and save the concatenated results using the original folder name.
    """
    for exp in exp_list:
        exp_folder1 = os.path.join(base_path, exp)
        x_files = collect_x_files(exp_folder1, x_res)
        if not x_files:
            print(f"No X files found in {exp} for X resolution '{x_res}'.")
            continue
        
        results = []
        for x_file in x_files:
            ds1 = xr.open_dataset(x_file)
            
            # Convert var_levels strings (e.g. "U01") to tuples: ("U", 1) or ("PSFC", None)
            levels = []
            for v in var_levels:
                if v.endswith('m'):
                    levels.append((v[:-1], None))
                    continue
                match = re.match(r'^([A-Za-z_]+)(\d+)$', v)
                if match:
                    levels.append((match.group(1), int(match.group(2))))
                else:
                    levels.append((v, None))
            result = extract_input_variables(ds1, imsize1=imsize_x, var_levels=levels)
            
            results.append(result)
        
        if results:
            concatenated_result = np.concatenate(results, axis=0)
            x_filename = f"x_{x_res}_{imsize_x[0]}x{imsize_x[1]}_{exp}.npy"

            np.save(os.path.join(root, x_filename), concatenated_result)
            print(f"Saved concatenated {x_filename} in {root}.")
        else:
            print(f"No valid files processed for folder {exp}.")
    print("All experiment folders have been processed and saved with concatenated data.")


if __name__ == '__main__':
    args = parse_args()
    imsize_x = args.imsize_variables
    root = os.path.join(args.root, 'wrf_data')
    base_path = args.wrf_base
    var_levels = args.var_levels

    os.makedirs(root, exist_ok=True)

    if args.experiment_wrf:
        print("Processing experiments:")
        process_experiments(args.experiment_wrf, base_path, imsize_x, root,
                            x_res=args.X_resolution, var_levels=var_levels)
