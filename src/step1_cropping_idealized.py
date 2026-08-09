import argparse
import glob
import os
import re

import numpy as np
import xarray as xr

from config_utils import (
    DEFAULT_CONFIG_PATH,
    apply_config_defaults,
    load_stage_config,
    render_output_filename,
)

STEP1_SETTINGS = (
    "data_dir",
    "workdir",
    "imsize_x",
    "imsize_y",
    "experiment_wrf",
    "x_resolution",
    "output_filename_template",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop idealized WRF outputs and write one NetCDF per storm folder"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration file (default: src/config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        "-dd",
        "-b",
        "--wrf_base",
        type=str,
        default=None,
        help="Override idealize.step1.data_dir",
    )
    parser.add_argument(
        "--workdir",
        "-wd",
        type=str,
        default=None,
        help="Override idealize.step1.workdir",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default=None,
        help="Legacy output root; if --workdir is omitted, output is <root>/wrf_data",
    )
    parser.add_argument(
        "--imsize_x",
        "-x",
        type=int,
        default=None,
        help="Crop width",
    )
    parser.add_argument(
        "--imsize_y",
        "-y",
        type=int,
        default=None,
        help="Crop height",
    )
    parser.add_argument(
        "-ix",
        "--imsize_variables",
        type=int,
        nargs=2,
        default=None,
        help="Legacy crop size [x y], used when --imsize_x/--imsize_y are not set",
    )
    parser.add_argument(
        "-vl",
        "--var_levels",
        type=str,
        nargs="+",
        default=None,
        help="Accepted for compatibility; step1 writes cropped fields, step2 selects variables",
    )
    parser.add_argument(
        "-ew",
        "--experiment_wrf",
        type=str,
        nargs="+",
        default=None,
        help="Override idealize.step1.experiment_wrf",
    )
    parser.add_argument(
        "--x_resolution",
        "-xd",
        "--X_resolution",
        type=str,
        default=None,
        help="Override idealize.step1.x_resolution",
    )
    parser.add_argument(
        "--output_filename_template",
        default=None,
        help="Override the NetCDF filename template; supports '{experiment_id}'",
    )
    return parser.parse_args()


def resolve_imsize(args):
    if args.imsize_x is not None or args.imsize_y is not None:
        if args.imsize_x is None or args.imsize_y is None:
            raise ValueError("Both --imsize_x and --imsize_y must be provided together")
        imsize = [int(args.imsize_x), int(args.imsize_y)]
    else:
        imsize = [int(args.imsize_variables[0]), int(args.imsize_variables[1])]
    if imsize[0] <= 0 or imsize[1] <= 0:
        raise ValueError(f"Invalid image size: {imsize}")
    return imsize


def natural_sort_key(text):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"([0-9]+)", text)]


def collect_x_files(exp_folder, x_res):
    all_files = glob.glob(os.path.join(exp_folder, "*"))
    x_files = [path for path in all_files if x_res in os.path.basename(path)]
    x_files.sort(key=lambda path: natural_sort_key(os.path.basename(path)))
    return x_files


def get_coord_names(ds):
    lat_candidates = ("XLAT", "XLAT_M")
    lon_candidates = ("XLONG", "XLONG_M")
    lat_name = next((name for name in lat_candidates if name in ds), None)
    lon_name = next((name for name in lon_candidates if name in ds), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not find latitude/longitude fields (e.g., XLAT/XLONG).")
    return lat_name, lon_name


def decode_times_from_wrf(ds, n_time):
    if "Times" in ds:
        raw = ds["Times"].values
        strings = []
        if raw.ndim == 2:
            for row in raw:
                chars = []
                for val in row:
                    if isinstance(val, (bytes, np.bytes_)):
                        chars.append(val.decode("utf-8", errors="ignore"))
                    else:
                        chars.append(str(val))
                strings.append("".join(chars).strip())
        elif raw.ndim == 1:
            for val in raw:
                if isinstance(val, (bytes, np.bytes_)):
                    strings.append(val.decode("utf-8", errors="ignore").strip())
                else:
                    strings.append(str(val).strip())
        if strings:
            parsed = []
            for s in strings:
                try:
                    parsed.append(np.datetime64(s.replace("_", "T")))
                except Exception:
                    parsed = []
                    break
            if len(parsed) == n_time:
                return np.array(parsed, dtype="datetime64[ns]")
    base = np.datetime64("2000-01-01T00:00:00")
    return base + np.arange(n_time).astype("timedelta64[h]")


def resolve_time_values(ds_sub, ds_raw):
    n_time = int(ds_sub.sizes["Time"])
    time_coord = ds_sub.coords.get("Time")
    if time_coord is not None and np.issubdtype(time_coord.dtype, np.datetime64):
        return np.array(time_coord.values, dtype="datetime64[ns]")

    if "XTIME" in ds_sub.coords:
        xtime = ds_sub["XTIME"].values
        if np.issubdtype(np.array(xtime).dtype, np.datetime64):
            return np.array(xtime, dtype="datetime64[ns]")

    if "XTIME" in ds_raw.coords:
        xtime = ds_raw["XTIME"].values
        if np.issubdtype(np.array(xtime).dtype, np.datetime64):
            return np.array(xtime, dtype="datetime64[ns]")

    return decode_times_from_wrf(ds_raw, n_time)


def interpolate_staggered_to_center(ds, sn_dim, we_dim):
    sn_stag = f"{sn_dim}_stag"
    we_stag = f"{we_dim}_stag"

    for name in list(ds.data_vars):
        var = ds[name]
        dims = var.dims
        if sn_stag not in dims and we_stag not in dims:
            continue

        var_c = var
        if sn_stag in dims:
            var_c = 0.5 * (
                var_c.isel({sn_stag: slice(0, ds.sizes[sn_dim])})
                + var_c.isel({sn_stag: slice(1, ds.sizes[sn_stag])})
            )
            var_c = var_c.rename({sn_stag: sn_dim})

        if we_stag in var_c.dims:
            var_c = 0.5 * (
                var_c.isel({we_stag: slice(0, ds.sizes[we_dim])})
                + var_c.isel({we_stag: slice(1, ds.sizes[we_stag])})
            )
            var_c = var_c.rename({we_stag: we_dim})

        ds[name] = var_c
    return ds


def compute_center_crop(size, crop):
    center = size // 2
    start = max(center - crop // 2, 0)
    end = start + crop
    if end > size:
        end = size
        start = max(0, end - crop)
    return start, end


def extract_data_centered(ds, imsize_x, imsize_y):
    lat_name, lon_name = get_coord_names(ds)
    grid_lat = ds[lat_name].isel(Time=0)
    grid_lon = ds[lon_name].isel(Time=0)
    sn_dim, we_dim = "south_north", "west_east"

    ds_work = interpolate_staggered_to_center(ds.copy(deep=False), sn_dim, we_dim)

    sni, eni = compute_center_crop(ds_work.sizes[sn_dim], imsize_y)
    swi, ewi = compute_center_crop(ds_work.sizes[we_dim], imsize_x)
    ds_sub = ds_work.isel({sn_dim: slice(sni, eni), we_dim: slice(swi, ewi)})

    rename_map = {}
    if sn_dim != "y":
        rename_map[sn_dim] = "y"
    if we_dim != "x":
        rename_map[we_dim] = "x"
    if rename_map:
        ds_sub = ds_sub.rename(rename_map)

    y_size = int(ds_sub.sizes["y"])
    x_size = int(ds_sub.sizes["x"])
    ds_sub = ds_sub.assign_coords({"y": np.arange(y_size), "x": np.arange(x_size)})

    new_lat = ds[lat_name].isel({sn_dim: slice(sni, eni), we_dim: slice(swi, ewi)}).rename(
        {sn_dim: "y", we_dim: "x"}
    )
    new_lon = ds[lon_name].isel({sn_dim: slice(sni, eni), we_dim: slice(swi, ewi)}).rename(
        {sn_dim: "y", we_dim: "x"}
    )
    ds_sub = ds_sub.assign_coords({"TRUE_LAT": new_lat, "TRUE_LONG": new_lon})

    center_i = ds_work.sizes[sn_dim] // 2
    center_j = ds_work.sizes[we_dim] // 2
    cen_lat_val = float(grid_lat.isel({sn_dim: center_i, we_dim: center_j}).values)
    cen_lon_val = float(grid_lon.isel({sn_dim: center_i, we_dim: center_j}).values)

    time_values = resolve_time_values(ds_sub, ds)
    ds_sub = ds_sub.assign_coords(Time=("Time", time_values))
    n_time = int(ds_sub.sizes["Time"])
    ds_sub = ds_sub.assign_coords(
        cen_lat=("Time", np.full(n_time, cen_lat_val)),
        cen_lon=("Time", np.full(n_time, cen_lon_val)),
    )

    drop_vars = [
        lat_name,
        lon_name,
        f"{lat_name}_U",
        f"{lon_name}_U",
        f"{lat_name}_V",
        f"{lon_name}_V",
        "south_north_stag",
        "west_east_stag",
        "Times",
        "XTIME",
    ]
    ds_sub = ds_sub.drop_vars([name for name in drop_vars if name in ds_sub])
    return ds_sub.load()


def process_experiments(
    exp_list,
    base_path,
    imsize_x,
    workdir,
    x_res,
    output_filename_template,
):
    output_filenames = {
        exp: render_output_filename(
            output_filename_template,
            experiment_id=exp,
        )
        for exp in exp_list
    }
    if len(set(output_filenames.values())) != len(output_filenames):
        raise ValueError(
            "output_filename_template must produce a unique name for every experiment"
        )
    for exp in exp_list:
        exp_folder = os.path.join(base_path, exp)
        x_files = collect_x_files(exp_folder, x_res)
        if not x_files:
            print(f"No files found in {exp} for resolution token '{x_res}'.")
            continue

        print(f"=== Starting storm folder {exp} ===")
        ds_list = []
        skip_storm = False
        for x_file in x_files:
            try:
                with xr.open_dataset(x_file) as ds:
                    ds_sub = extract_data_centered(ds, imsize_x[0], imsize_x[1])
                ds_list.append(ds_sub)
                print(f"  Added {os.path.basename(x_file)}")
            except Exception as exc:
                print(f"  Error processing {x_file}: {exc!r}")
                print(f"  Skipping storm folder {exp}")
                skip_storm = True
                break

        if skip_storm or not ds_list:
            continue

        try:
            ds_merged = xr.concat(ds_list, dim="Time").sortby("Time")
            experiment_id = str(exp)
            ds_merged = ds_merged.assign_coords(
                experiment_id=(
                    "Time",
                    np.full(
                        ds_merged.sizes["Time"],
                        experiment_id,
                        dtype=f"<U{max(1, len(experiment_id))}",
                    ),
                )
            )
            out_fn = os.path.join(workdir, output_filenames[exp])
            ds_merged.to_netcdf(out_fn)
            print(f"Saved {out_fn}\n")
        except Exception as exc:
            print(f"Failed to save {exp}: {exc!r}\n")

    print("All storm folders processed.")


if __name__ == "__main__":
    args = parse_args()
    if args.workdir is None and args.root is not None:
        args.workdir = os.path.join(args.root, "wrf_data")
    if (
        args.imsize_x is None
        and args.imsize_y is None
        and args.imsize_variables is not None
    ):
        args.imsize_x, args.imsize_y = args.imsize_variables
    settings = load_stage_config(args.config, "idealize", "step1")
    args = apply_config_defaults(args, settings, STEP1_SETTINGS)
    imsize = resolve_imsize(args)
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    if args.experiment_wrf:
        process_experiments(
            exp_list=args.experiment_wrf,
            base_path=args.data_dir,
            imsize_x=imsize,
            workdir=workdir,
            x_res=args.x_resolution,
            output_filename_template=args.output_filename_template,
        )
