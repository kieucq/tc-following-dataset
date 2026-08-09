import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from collections import defaultdict
import argparse

from config_utils import (
    DEFAULT_CONFIG_PATH,
    apply_config_defaults,
    load_stage_config,
    render_output_filename,
)


STEP1_SETTINGS = (
    "track_file",
    "data_dir",
    "workdir",
    "imsize_x",
    "imsize_y",
    "interpolation_steps",
    "timestep_hours",
    "output_filename_template",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop raw WRF outputs around TC tracks"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration file (default: src/config.yaml)",
    )
    parser.add_argument(
        "--track_file", "-td",
        type=str,
        default=None,
        help="Override cmip.step1.track_file"
    )
    parser.add_argument(
        "--data_dir", "-dd",
        type=str,
        default=None,
        help="Override cmip.step1.data_dir"
    )
    parser.add_argument(
        "--workdir", "-wd",
        default=None,
        help="Override cmip.step1.workdir"
    )
    parser.add_argument(
        "--imsize_x", "-x",
        type=int,
        default=None,
        help="Override cmip.step1.imsize_x"
    )
    parser.add_argument(
        "--imsize_y", "-y",
        type=int,
        default=None,
        help="Override cmip.step1.imsize_y"
    )
    parser.add_argument(
        "--interpolation_steps",
        type=int,
        default=None,
        help="Override cmip.step1.interpolation_steps",
    )
    parser.add_argument(
        "--timestep_hours",
        type=int,
        default=None,
        help="Override cmip.step1.timestep_hours",
    )
    parser.add_argument(
        "--output_filename_template",
        default=None,
        help="Override the NetCDF filename template; supports '{storm_id}'",
    )
    return parser.parse_args()


def interpolate(val_start, val_end, frac):
    """Helper function for linear interpolation."""
    return val_start + frac * (val_end - val_start)


def track_reader(file_path, interpolation_steps):
    grouped_data = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                key = parts[0]
                numbers = list(map(float, parts))
                grouped_data[key].append(numbers)
    expanded_groups = defaultdict(list)
    for key, group in grouped_data.items():
        for i in range(len(group) - 1):
            current, next_row = group[i], group[i+1]
            for step in range(interpolation_steps):
                frac = step / float(interpolation_steps)
                interp4 = interpolate(current[3], next_row[3], frac)
                interp5 = interpolate(current[4], next_row[4], frac)
                new_row = current[:3] + [step] + [interp4, interp5] + current[5:]
                expanded_groups[key].append(new_row)
        last = group[-1]
        last_row = last[:3] + [0] + last[3:]
        expanded_groups[key].append(last_row)
    return expanded_groups


def get_coord_names(var):
    return "XLAT", "XLONG"


def extract_data(
    ds,
    clat,
    clon,
    imsize_x,
    imsize_y,
    storm_id=None,
    storm_lifetime=None,
    lifetime_frame=None,
):
    lat_name, lon_name = get_coord_names(None)
    grid_lat = ds[lat_name].isel(Time=0)
    grid_lon = ds[lon_name].isel(Time=0)
    sn_dim, we_dim = 'south_north', 'west_east'
    dist = np.sqrt((grid_lat - clat)**2 + (grid_lon - clon)**2)
    ci, cj = np.unravel_index(int(dist.argmin().values), dist.shape)
    half_y, half_x = imsize_y//2, imsize_x//2
    sni, swi = max(ci-half_y,0), max(cj-half_x,0)
    eni, ewi = sni+imsize_y, swi+imsize_x
    # ---- 1) INTERPOLATE ALL STAGGERED VARS ONTO CENTER GRID ----
    sn_stag = sn_dim + '_stag'
    we_stag = we_dim + '_stag'
    for name, var in ds.data_vars.items():
        dims = var.dims
        if sn_stag in dims or we_stag in dims:
            var_c = var
            # U‐style (south_north_stag → south_north)
            if sn_stag in dims:
                var_c = 0.5 * (
                    var_c.isel({sn_stag: slice(0, ds.sizes[sn_dim])}) +
                    var_c.isel({sn_stag: slice(1, ds.sizes[sn_stag])})
                )
                var_c = var_c.rename({sn_stag: sn_dim})
            # V‐style (west_east_stag → west_east)
            if we_stag in dims:
                var_c = 0.5 * (
                    var_c.isel({we_stag: slice(0, ds.sizes[we_dim])}) +
                    var_c.isel({we_stag: slice(1, ds.sizes[we_stag])})
                )
                var_c = var_c.rename({we_stag: we_dim})
            # override original var with center‐grid version
            ds[name] = var_c
    # ---------------------------------------------------------------
    # subset full dataset
    ds_sub = ds.isel({sn_dim: slice(sni, eni), we_dim: slice(swi, ewi)})
    # rename dims to x/y
    ds_sub = ds_sub.rename({sn_dim: 'y', we_dim: 'x'})
    # assign 1D coords
    ds_sub = ds_sub.assign_coords({'y': np.arange(imsize_y), 'x': np.arange(imsize_x)})

    # crop & attach true lat/lon
    new_lat = ds[lat_name].isel({
        'Time': slice(None), sn_dim: slice(sni, eni), we_dim: slice(swi, ewi)
    }).rename({sn_dim: 'y', we_dim: 'x'})
    new_lon = ds[lon_name].isel({
        'Time': slice(None), sn_dim: slice(sni, eni), we_dim: slice(swi, ewi)
    }).rename({sn_dim: 'y', we_dim: 'x'})
    ds_sub = ds_sub.assign_coords({'TRUE_LAT': new_lat, 'TRUE_LONG': new_lon})

    # drop original staggered coords & raw lat/lon
    drop_vars = [lat_name, lon_name,
                 f"{lat_name}_U", f"{lon_name}_U",
                 f"{lat_name}_V", f"{lon_name}_V"]
    ds_sub = ds_sub.drop_vars([v for v in drop_vars if v in ds_sub])

    nT = ds_sub.sizes['Time']
    cen_lat = xr.DataArray(
        np.full(nT, clat),
        dims=('Time',),
        coords={'Time': ds_sub.Time},
        name='cen_lat'
    )
    cen_lon = xr.DataArray(
        np.full(nT, clon),
        dims=('Time',),
        coords={'Time': ds_sub.Time},
        name='cen_lon'
    )
    ds_sub = ds_sub.assign_coords(cen_lat=cen_lat, cen_lon=cen_lon)

    if storm_id is not None:
        storm_id = str(storm_id)
        ds_sub = ds_sub.assign_coords(
            storm_id=(
                "Time",
                np.full(nT, storm_id, dtype=f"<U{max(1, len(storm_id))}"),
            )
        )
    if storm_lifetime is not None:
        ds_sub = ds_sub.assign_coords(
            storm_lifetime=("Time", np.full(nT, int(storm_lifetime), dtype=np.int64))
        )
    if lifetime_frame is not None:
        ds_sub = ds_sub.assign_coords(
            lifetime_frame=("Time", np.full(nT, int(lifetime_frame), dtype=np.int64))
        )

    ds_sub = (
        ds_sub
        .swap_dims({'Time':'XTIME'})        # 1) make XTIME your only time‐dim
        .reset_coords('Time', drop=True)     # 2) drop the dummy “Time” coord
        .rename({'XTIME':'Time'})            # 3) rename the XTIME dim & coord → “Time”
    )
    return ds_sub


def process_groups(
    groups,
    data_dir,
    workdir,
    imsize_x,
    imsize_y,
    timestep_hours,
    output_filename_template,
):
    os.makedirs(workdir, exist_ok=True)
    output_filenames = {
        group_id: render_output_filename(
            output_filename_template,
            storm_id=group_id,
        )
        for group_id in groups
    }
    if len(set(output_filenames.values())) != len(output_filenames):
        raise ValueError(
            "output_filename_template must produce a unique name for every storm"
        )
    for group_id, rows in groups.items():
        print(f"=== Starting group {group_id} ===")
        ds_list = []
        skip_group = False
        storm_lifetime = len(rows)

        for lifetime_frame, row in enumerate(rows, start=1):
            try:
                # Parse row values
                year = int(row[1])
                doy = int(row[2])
                tcode = int(row[3])
                clon, clat = row[4], row[5]
                # Build filename and path
                date_obj = datetime(year, 1, 1) + timedelta(days=doy - 1)
                hour = tcode * timestep_hours
                fname = f"raw_wrfout_d01_{date_obj:%Y-%m-%d}_{hour:02d}:00:00"
                fpath = os.path.join(data_dir, fname)

                print(f"Processing {group_id}, file: {fpath}")
                # Open and extract
                ds = xr.open_dataset(fpath)
                ds_sub = extract_data(
                    ds,
                    clat,
                    clon,
                    imsize_x,
                    imsize_y,
                    storm_id=group_id,
                    storm_lifetime=storm_lifetime,
                    lifetime_frame=lifetime_frame,
                )
                ds_list.append(ds_sub)

            except Exception as e:
                print(f"  Error processing trial {row}: {e!r}")
                print(f"  Skipping entire group {group_id}")
                skip_group = True
                break

        if skip_group:
            print(f"=== Group {group_id} skipped ===\n")
            continue

        if not ds_list:
            print(f"No successful trials in group {group_id}, nothing to save.\n")
            continue

        try:
            ds_merged = xr.concat(ds_list, dim='Time')
            out_fn = os.path.join(workdir, output_filenames[group_id])
            ds_merged.to_netcdf(out_fn)
            print(f"Saved {out_fn}\n")
        except Exception as e:
            print(f"Failed to save group {group_id}: {e!r}\n")
if __name__ == '__main__':
    args = parse_args()
    settings = load_stage_config(args.config, "cmip", "step1")
    args = apply_config_defaults(args, settings, STEP1_SETTINGS)
    if args.imsize_x <= 0 or args.imsize_y <= 0:
        raise ValueError("imsize_x and imsize_y must be positive")
    if args.interpolation_steps <= 0 or args.timestep_hours <= 0:
        raise ValueError("interpolation_steps and timestep_hours must be positive")

    groups = track_reader(args.track_file, args.interpolation_steps)
    process_groups(
        groups,
        data_dir=args.data_dir,
        workdir=args.workdir,
        imsize_x=args.imsize_x,
        imsize_y=args.imsize_y,
        timestep_hours=args.timestep_hours,
        output_filename_template=args.output_filename_template,
    )
