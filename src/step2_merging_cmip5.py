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
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process WRF NetCDF files into two big NumPy arrays"
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
        help="Override the indir value in the YAML configuration"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=None,
        help="Override the outdir value in the YAML configuration"
    )
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=None,
        help="Override the number of consecutive frames in the YAML configuration"
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=None,
        help="Override the number of time steps between sample starts",
    )
    parser.add_argument(
        "-vl", "--var_levels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override the YAML list of variable-level codes. "
            "Format e.g. U01, V02, QVAPOR03 or PSFC for surface. "
            "This will be parsed into (name, level) pairs."
        )
    )
    parser.add_argument("--x_filename", default=None, help="Override the X filename")
    parser.add_argument("--z_filename", default=None, help="Override the Z filename")
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
        shape = (ns, 4)  # [lon, lat, sin, cos]
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

        # build Z: [lon, lat, sin, cos]
        zarr = np.array([lon_arr[base], lat_arr[base], sinθ[base], cosθ[base]])
        sample_Z = xr.DataArray(
            zarr,
            dims=('feature',),
            coords={'feature': ['lon', 'lat', 'sin', 'cos']}
        )
        Z_list.append(sample_Z.expand_dims({'sample': [base]}))

    # concatenate and convert to NumPy
    X = xr.concat(X_list, dim='sample').values  # (ns, nf, nc, nh, nw)
    X = np.transpose(X, (0, 1, 3, 4, 2))        # → (ns, nf, nh, nw, nc)
    Z = xr.concat(Z_list, dim='sample').values  # (ns, 4)

    return X, Z, file_year

def process_data(
    indir: str,
    outdir: str,
    var_levels,
    frames: int,
    x_filename: str,
    z_filename: str,
    sample_stride: int,
):
    """
    Loop over all .nc files in indir, extract X/Z,
    concatenate into two big arrays and save them.
    """
    os.makedirs(outdir, exist_ok=True)
    filename_values = {"frames": frames, "variables": len(var_levels)}
    output_filenames = [
        render_output_filename(x_filename, **filename_values),
        render_output_filename(z_filename, **filename_values),
    ]
    if len(set(output_filenames)) != len(output_filenames):
        raise ValueError("Configured Stage-2 output filenames must be unique")
    X_list, Z_list = [], []

    for fname in sorted(os.listdir(indir)):
        if not fname.endswith('.nc'):
            continue
        ds = xr.open_dataset(os.path.join(indir, fname))
        n_time = int(ds.sizes['Time'])
        if n_time < frames:
            print(f"Skipping {fname}: Time len {n_time} < frames {frames}")
            continue 
        X, Z, _ = var_extract(ds, var_levels, frames, sample_stride)
        X_list.append(X)
        Z_list.append(Z)

    if not X_list:
        raise ValueError(f"No usable NetCDF files found in {indir}")

    X_all = np.concatenate(X_list, axis=0)
    Z_all = np.concatenate(Z_list, axis=0)

    np.save(os.path.join(outdir, output_filenames[0]), X_all)
    np.save(os.path.join(outdir, output_filenames[1]), Z_all)

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
        x_filename=args.x_filename,
        z_filename=args.z_filename,
        sample_stride=args.sample_stride,
    )
