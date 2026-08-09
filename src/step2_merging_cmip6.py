import os
import numpy as np
import xarray as xr
import argparse

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
    "storm_ids_filename",
    "storm_lifetimes_filename",
    "chunk_start_frames_filename",
)


def _storm_id_from_filename(fname: str) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0]
    prefix = "WRF_STORMID_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _first_value(ds: xr.Dataset, name: str, default=None):
    if name not in ds:
        return default
    values = np.asarray(ds[name].values).reshape(-1)
    if values.size == 0:
        return default
    return values[0].item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process WRF NetCDF files into NumPy arrays (X, Z, storm IDs)"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration file (default: src/config.yaml)",
    )
    parser.add_argument(
        "-i", "--indir",
        type=str,
        default=None,
        help="Override cmip.step2.indir"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=None,
        help="Override cmip.step2.outdir"
    )
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=None,
        help="Override cmip.step2.frames"
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=None,
        help="Override cmip.step2.sample_stride",
    )
    parser.add_argument(
        "-vl", "--var_levels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of variable-level codes. "
            "Format e.g. U01, V02, QVAPOR03 or PSFC for surface. "
            "This will be parsed into (name, level) pairs."
        )
    )
    parser.add_argument("--x_filename", default=None, help="Override the X filename")
    parser.add_argument("--z_filename", default=None, help="Override the Z filename")
    parser.add_argument(
        "--storm_ids_filename",
        default=None,
        help="Override the storm-ID filename",
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

def var_extract(
    ds: xr.Dataset,
    var_levels,
    frames: int,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build X and Z arrays from a WRF-style Dataset.

    Returns
    -------
    X : np.ndarray
        shape = (ns, nf, nh, nw, nc)
    Z : np.ndarray
        shape = (ns, nf, 6)
        [lon, lat, sin, cos, storm_lifetime, chunk_start_frame] for each frame.
        chunk_start_frame is 1-based within the storm lifetime.
    file_year : int
        Year of the first time step in this dataset.
    """
    # time-of-year features
    t = ds['Time']
    doy     = t.dt.dayofyear.values
    is_leap = t.dt.is_leap_year.values
    yearlen = np.where(is_leap, 366, 365)
    theta   = 2 * np.pi * doy / yearlen
    sinθ, cosθ = np.sin(theta), np.cos(theta)

    lon_arr  = ds['cen_lon'].values
    lat_arr  = ds['cen_lat'].values
    lifetime_arr = (
        ds['storm_lifetime'].values
        if 'storm_lifetime' in ds
        else np.full(ds.sizes['Time'], ds.sizes['Time'], dtype=np.int64)
    )
    lifetime_frame_arr = (
        ds['lifetime_frame'].values
        if 'lifetime_frame' in ds
        else np.arange(1, ds.sizes['Time'] + 1, dtype=np.int64)
    )
    year_arr = t.dt.year.values
    file_year = int(year_arr[0])

    # extract each variable as (Time, y, x)
    var_das, var_names = [], []
    for name, lvl in var_levels:
        da = ds[name]
        if lvl is not None:
            lvl_dim = next(d for d in da.dims if 'bottom_top' in d)
            if name == 'PHB':
                da = da.isel({lvl_dim: lvl}) + ds['PH'].isel({lvl_dim: lvl})
            elif name == 'PH':
                da = da.isel({lvl_dim: lvl}) + ds['PHB'].isel({lvl_dim: lvl})
            else:
                da = da.isel({lvl_dim: lvl})
        var_das.append(da)
        var_names.append(f"{name}_{lvl}" if lvl is not None else name)

    # sample indices: configured stride, then frames at unit spacing
    n_time = ds.sizes['Time']
    bases = np.arange(0, n_time, sample_stride)
    X_list, Z_list = [], []
    chunk_start_frames, storm_lifetimes = [], []

    for base in bases:
        idx_hist = base + np.arange(frames)
        if idx_hist.max() >= n_time:
            continue

        # build X: (frames, vars, y, x)
        hist_vars = []
        for da in var_das:
            h = da.isel(Time=idx_hist).rename({'Time': 'frame'})
            hist_vars.append(h)
        sample_X = xr.concat(hist_vars, dim='var')
        sample_X = sample_X.assign_coords(var=var_names)
        sample_X = sample_X.transpose('frame', 'var', 'y', 'x')
        # reset the frame‐coordinate to a simple 0..frames-1 index
        sample_X = sample_X.assign_coords(frame=np.arange(sample_X.sizes['frame']))
        X_list.append(sample_X.expand_dims({'sample': [base]}))

        chunk_start_frame = int(np.asarray(lifetime_frame_arr)[base])
        storm_lifetime = int(np.asarray(lifetime_arr)[base])
        chunk_start_frames.append(chunk_start_frame)
        storm_lifetimes.append(storm_lifetime)

        # build Z per frame: [lon, lat, sin, cos, storm_lifetime, chunk_start_frame]
        zarr = np.stack(
            [
                lon_arr[idx_hist],
                lat_arr[idx_hist],
                sinθ[idx_hist],
                cosθ[idx_hist],
                np.full(frames, storm_lifetime, dtype=np.float64),
                np.full(frames, chunk_start_frame, dtype=np.float64),
            ],
            axis=-1,
        )
        sample_Z = xr.DataArray(
            zarr,
            dims=('frame', 'feature'),
            coords={
                'frame': np.arange(frames),
                'feature': ['lon', 'lat', 'sin', 'cos', 'storm_lifetime', 'chunk_start_frame'],
            },
        )
        Z_list.append(sample_Z.expand_dims({'sample': [base]}))

    # concatenate and convert to NumPy
    X = xr.concat(X_list, dim='sample').values  # (ns, nf, nc, nh, nw)
    X = np.transpose(X, (0, 1, 3, 4, 2))        # → (ns, nf, nh, nw, nc)
    Z = xr.concat(Z_list, dim='sample').values  # (ns, nf, 6)

    return (
        X,
        Z,
        file_year,
        np.asarray(storm_lifetimes, dtype=np.int64),
        np.asarray(chunk_start_frames, dtype=np.int64),
    )

def process_data(
    indir: str,
    outdir: str,
    var_levels,
    frames: int,
    sample_stride: int,
    x_filename: str,
    z_filename: str,
    storm_ids_filename: str,
    storm_lifetimes_filename: str,
    chunk_start_frames_filename: str,
):
    """
    Loop over all .nc files in indir, extract X/Z,
    concatenate arrays and save:
      - configured X and per-frame Z files
      - configured storm-ID, storm-lifetime, and chunk-start-frame files
    """
    os.makedirs(outdir, exist_ok=True)
    filename_values = {"frames": frames, "variables": len(var_levels)}
    filename_templates = [
        x_filename,
        z_filename,
        storm_ids_filename,
        storm_lifetimes_filename,
        chunk_start_frames_filename,
    ]
    output_filenames = [
        render_output_filename(template, **filename_values)
        for template in filename_templates
    ]
    if len(set(output_filenames)) != len(output_filenames):
        raise ValueError("Configured Stage-2 output filenames must be unique")
    X_list, Z_list, storm_id_list = [], [], []
    storm_lifetime_list, chunk_start_frame_list = [], []

    for fname in sorted(os.listdir(indir)):
        if not fname.endswith('.nc'):
            continue
        fpath = os.path.join(indir, fname)
        with xr.open_dataset(fpath) as ds:
            n_time = int(ds.sizes['Time'])
            if n_time < frames:
                print(f"Skipping {fname}: Time len {n_time} < frames {frames}")
                continue
            X, Z, _, storm_lifetimes, chunk_start_frames = var_extract(
                ds,
                var_levels,
                frames,
                sample_stride,
            )
            storm_id = str(_first_value(ds, "storm_id", _storm_id_from_filename(fname)))
        X_list.append(X)
        Z_list.append(Z)
        storm_id_dtype = f"<U{max(1, len(storm_id))}"
        storm_id_list.append(np.full(X.shape[0], storm_id, dtype=storm_id_dtype))
        storm_lifetime_list.append(storm_lifetimes)
        chunk_start_frame_list.append(chunk_start_frames)

    if not X_list:
        raise RuntimeError(f"No valid .nc files found in {indir}")

    X_all = np.concatenate(X_list, axis=0)
    Z_all = np.concatenate(Z_list, axis=0)
    storm_ids_all = np.concatenate(storm_id_list, axis=0).astype(str)
    storm_lifetimes_all = np.concatenate(storm_lifetime_list, axis=0).astype(np.int64)
    chunk_start_frames_all = np.concatenate(chunk_start_frame_list, axis=0).astype(np.int64)

    output_arrays = [
        X_all,
        Z_all,
        storm_ids_all,
        storm_lifetimes_all,
        chunk_start_frames_all,
    ]
    for filename, array in zip(output_filenames, output_arrays):
        np.save(os.path.join(outdir, filename), array)

if __name__ == "__main__":
    args = parse_args()
    settings = load_stage_config(args.config, "cmip", "step2")
    args = apply_config_defaults(args, settings, STEP2_SETTINGS)
    if args.frames <= 0 or args.sample_stride <= 0:
        raise ValueError("frames and sample_stride must be positive")
    if not isinstance(args.var_levels, list) or not args.var_levels:
        raise ValueError("var_levels must contain at least one variable code")

    # parse var_levels strings into (name, level) pairs
    var_levels = [
        (v[:-2], int(v[-2:])) if v[-2:].isdigit() else ((v[:-1], None) if v[-1:]=='m' else (v, None))
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
        storm_ids_filename=args.storm_ids_filename,
        storm_lifetimes_filename=args.storm_lifetimes_filename,
        chunk_start_frames_filename=args.chunk_start_frames_filename,
    )
