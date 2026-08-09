import argparse
import os

import numpy as np
import xarray as xr

from config_utils import (
    DEFAULT_CONFIG_PATH,
    apply_config_defaults,
    load_stage_config,
    render_output_filename,
)


STEP2_SETTINGS = (
    "indir",
    "outdir",
    "frames",
    "sample_stride",
    "var_levels",
    "x_filename",
    "z_filename",
    "experiment_ids_filename",
    "storm_lifetimes_filename",
    "chunk_start_frames_filename",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process idealized WRF NetCDF files into X/Z NumPy arrays (CMIP6-style step2)"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration file (default: src/config.yaml)",
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        default=None,
        help="Override idealize.step2.indir",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        help="Override idealize.step2.outdir",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        default=None,
        help="Override idealize.step2.frames",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=None,
        help="Override idealize.step2.sample_stride",
    )
    parser.add_argument(
        "-vl",
        "--var_levels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of variable-level codes. "
            "Format e.g. U01, V02, QVAPOR03 or PSFC for surface."
        ),
    )
    parser.add_argument("--x_filename", default=None, help="Override the X filename")
    parser.add_argument("--z_filename", default=None, help="Override the Z filename")
    parser.add_argument(
        "--experiment_ids_filename",
        default=None,
        help="Override the experiment-ID filename",
    )
    parser.add_argument(
        "--storm_lifetimes_filename",
        default=None,
        help="Override the storm-lifetime filename",
    )
    parser.add_argument(
        "--chunk_start_frames_filename",
        default=None,
        help="Override the chunk-start-frame filename",
    )
    return parser.parse_args()


def var_extract(ds: xr.Dataset, var_levels, frames: int, sample_stride: int):
    t = ds["Time"]
    doy = t.dt.dayofyear.values
    is_leap = t.dt.is_leap_year.values
    yearlen = np.where(is_leap, 366, 365)
    theta = 2 * np.pi * doy / yearlen
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)

    lon_arr = ds["cen_lon"].values
    lat_arr = ds["cen_lat"].values
    year_arr = t.dt.year.values
    file_year = int(year_arr[0])

    var_das, var_names = [], []
    for name, lvl in var_levels:
        da = ds[name]
        if lvl is not None:
            lvl_dim = next(d for d in da.dims if "bottom_top" in d)
            if name == "PHB":
                da = da.isel({lvl_dim: lvl}) + ds["PH"].isel({lvl_dim: lvl})
            elif name == "PH":
                da = da.isel({lvl_dim: lvl}) + ds["PHB"].isel({lvl_dim: lvl})
            else:
                da = da.isel({lvl_dim: lvl})
        da = da.reset_coords(drop=True)
        var_das.append(da)
        var_names.append(f"{name}_{lvl}" if lvl is not None else name)

    n_time = ds.sizes["Time"]
    n_time = int(ds.sizes["Time"])
    F = int(frames)     # 5
    S = int(sample_stride)
    
    # discard ONLY from the beginning so that the last chunk ends at the final frame
    head = (n_time - F) % S          # 0..3
    last_base = n_time - F
    
    bases = np.arange(head, last_base + 1, S)

    x_list, z_list = [], []
    chunk_start_frames, storm_lifetimes = [], []

    for base in bases:
        idx_hist = base + np.arange(frames)
        if idx_hist.max() >= n_time:
            continue

        hist_vars = []
        for da in var_das:
            h = da.isel(Time=idx_hist).rename({"Time": "frame"})
            hist_vars.append(h)
        sample_x = xr.concat(hist_vars, dim="var", coords="minimal", compat="override")
        sample_x = sample_x.assign_coords(var=var_names)
        sample_x = sample_x.transpose("frame", "var", "y", "x")
        sample_x = sample_x.assign_coords(frame=np.arange(sample_x.sizes["frame"]))
        x_list.append(sample_x.expand_dims({"sample": [base]}))

        chunk_start_frame = int(base) + 1
        storm_lifetime = int(n_time)
        chunk_start_frames.append(chunk_start_frame)
        storm_lifetimes.append(storm_lifetime)

        zarr = np.stack(
            [
                lon_arr[idx_hist],
                lat_arr[idx_hist],
                sin_theta[idx_hist],
                cos_theta[idx_hist],
                np.full(frames, storm_lifetime, dtype=np.float64),
                np.full(frames, chunk_start_frame, dtype=np.float64),
            ],
            axis=-1,
        )
        sample_z = xr.DataArray(
            zarr,
            dims=("frame", "feature"),
            coords={
                "frame": np.arange(frames),
                "feature": ["lon", "lat", "sin", "cos", "storm_lifetime", "chunk_start_frame"],
            },
        )
        z_list.append(sample_z.expand_dims({"sample": [base]}))

    x = xr.concat(x_list, dim="sample").values
    x = np.transpose(x, (0, 1, 3, 4, 2))
    z = xr.concat(z_list, dim="sample").values
    return (
        x,
        z,
        file_year,
        np.asarray(storm_lifetimes, dtype=np.int64),
        np.asarray(chunk_start_frames, dtype=np.int64),
    )


def process_data(
    indir,
    outdir,
    var_levels,
    frames: int,
    sample_stride: int,
    x_filename: str,
    z_filename: str,
    experiment_ids_filename: str,
    storm_lifetimes_filename: str,
    chunk_start_frames_filename: str,
):
    os.makedirs(outdir, exist_ok=True)
    filename_values = {"frames": frames, "variables": len(var_levels)}
    filename_templates = [
        x_filename,
        z_filename,
        experiment_ids_filename,
        storm_lifetimes_filename,
        chunk_start_frames_filename,
    ]
    output_filenames = [
        render_output_filename(template, **filename_values)
        for template in filename_templates
    ]
    if len(set(output_filenames)) != len(output_filenames):
        raise ValueError("Configured Stage-2 output filenames must be unique")
    x_list, z_list, exp_id_list = [], [], []
    storm_lifetime_list, chunk_start_frame_list = [], []

    for fname in sorted(os.listdir(indir)):
        if not fname.endswith(".nc"):
            continue
        fpath = os.path.join(indir, fname)
        with xr.open_dataset(fpath, decode_times=True) as ds:
            #print(ds['Time'])
            n_time = int(ds.sizes["Time"])
            if n_time < frames:
                print(f"Skipping {fname}: Time len {n_time} < frames {frames}")
                continue
            x, z, _, storm_lifetimes, chunk_start_frames = var_extract(
                ds,
                var_levels,
                frames,
                sample_stride,
            )
            if "experiment_id" in ds:
                experiment_values = np.asarray(ds["experiment_id"].values).reshape(-1)
                sample_exp_id = str(experiment_values[0].item())
            else:
                sample_exp_id = os.path.splitext(fname)[0]
        x_list.append(x)
        z_list.append(z)
        exp_id_dtype = f"<U{max(1, len(sample_exp_id))}"
        exp_id_list.append(np.full(x.shape[0], sample_exp_id, dtype=exp_id_dtype))
        storm_lifetime_list.append(storm_lifetimes)
        chunk_start_frame_list.append(chunk_start_frames)

    if not x_list:
        raise RuntimeError(f"No valid .nc files found in {indir}")

    x_all = np.concatenate(x_list, axis=0)
    z_all = np.concatenate(z_list, axis=0)
    exp_ids_all = np.concatenate(exp_id_list, axis=0).astype(str)
    storm_lifetimes_all = np.concatenate(storm_lifetime_list, axis=0).astype(np.int64)
    chunk_start_frames_all = np.concatenate(chunk_start_frame_list, axis=0).astype(np.int64)

    output_arrays = [
        x_all,
        z_all,
        exp_ids_all,
        storm_lifetimes_all,
        chunk_start_frames_all,
    ]
    for filename, array in zip(output_filenames, output_arrays):
        np.save(os.path.join(outdir, filename), array)


if __name__ == "__main__":
    args = parse_args()
    settings = load_stage_config(args.config, "idealize", "step2")
    args = apply_config_defaults(args, settings, STEP2_SETTINGS)
    if args.frames <= 0 or args.sample_stride <= 0:
        raise ValueError("frames and sample_stride must be positive")
    if not isinstance(args.var_levels, list) or not args.var_levels:
        raise ValueError("var_levels must contain at least one variable code")

    var_levels = [
        (v[:-2], int(v[-2:])) if v[-2:].isdigit() else ((v[:-1], None) if v[-1:] == "m" else (v, None))
        for v in args.var_levels
    ]
    process_data(
        indir=args.indir,
        outdir=args.outdir,
        var_levels=var_levels,
        frames=args.frames,
        sample_stride=args.sample_stride,
        x_filename=args.x_filename,
        z_filename=args.z_filename,
        experiment_ids_filename=args.experiment_ids_filename,
        storm_lifetimes_filename=args.storm_lifetimes_filename,
        chunk_start_frames_filename=args.chunk_start_frames_filename,
    )
