"""
Archive the 00Z ICON-CH2-EPS ensemble (all 21 members) to zarr.

MeteoSwiss's OGD API keeps each run for ~24h with no backfill, so this runs
every 2h from a systemd timer rather than once at a fixed time. If the run is
already archived completely it exits after a local metadata check, without
touching the network. An incomplete run is fetched again in full.

Usage:
    python -m solarcast.archive.icon_ch2_eps --run-hour 00
    python -m solarcast.archive.icon_ch2_eps --run-hour 00 --date 2026-09-08

Design notes:
  - The reference time is fixed once and passed explicitly on every request,
    otherwise with "latest" it would resolve to different runs while MeteoSwiss is
    still publishing and would mix runs.
  - Each (variable, horizon) takes two requests: perturbed=True returns the
    20 perturbed members, perturbed=False the control.
  - A failed (variable, horizon) is logged and NaN-padded rather than aborting
    the run. Padding keeps lead_time the same length for every variable, which
    appending requires.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from meteodatalab import ogd_api

from solarcast import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLLECTION = "ogd-forecasting-icon-ch2"

OUT_ROOT = paths.ICON_CH2_EPS
SCRATCH_ROOT = paths.DATA / ".earthkit-scratch"


def _configure_earthkit_cache() -> None:
    """Keep earthkit-data's temporary download cache under data/ (Downloads regularly exceed 5 GB)"""
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    from earthkit.data import config as ekd_config

    ekd_config.set("cache-policy", "temporary")
    ekd_config.set("temporary-cache-directory-root", str(SCRATCH_ROOT.resolve()))
    log.info("earthkit-data cache directory root set to %s", SCRATCH_ROOT.resolve())


def _silence_earthkit_download_progress_bars() -> None:
    """
    Disable earthkit-data's per-file download progress bar, which has no config
    switch and prints a line per chunk when output is piped
    """
    from tqdm import tqdm
    import earthkit.data.sources.url as ekd_url

    def _disabled_progress_bar(*, total=None, iterable=None, initial=0, desc=None):
        return tqdm(iterable=iterable, total=total, initial=initial, disable=True, leave=False, desc=desc)

    ekd_url.progress_bar = _disabled_progress_bar

# Inputs to the PV models
VARIABLES = [
    "ASWDIR_S",  # direct downward shortwave at the surface, mean since model start
    "ASWDIFD_S",  # diffuse downward shortwave at the surface, mean since model start
    "T_2M",  # 2m temperature
    "CLCT",  # total cloud cover
    "H_SNOW",  # snow depth
    "U_10M",  # 10m wind
    "V_10M",
]


def build_horizons(max_hours: int = 120, dense_to: int = 48, coarse_step: int = 3) -> list[str]:
    """
    Build the list of horizons (ISO 8601 strings) once and reuse it for every variable, so the archive
    has the same lead_time coordinates for every variable
    """
    hours = list(range(0, dense_to + 1, 1)) + list(range(dense_to + coarse_step, max_hours + 1, coarse_step))
    horizons = []
    for h in hours:
        d, r = divmod(h, 24)
        horizons.append(f"P{d}DT{r}H")
    return horizons


_HORIZON_RE = re.compile(r"^P(\d+)DT(\d+)H$")


def horizon_to_hours(horizon: str) -> int:
    """Inverse of the P{d}DT{h}H construction in build_horizons()"""
    m = _HORIZON_RE.match(horizon)
    if not m:
        raise ValueError(f"Unrecognised horizon format: {horizon!r}")
    days, hours = int(m.group(1)), int(m.group(2))
    return days * 24 + hours


def make_nan_placeholder(template: xr.DataArray, horizon: str) -> xr.DataArray:
    """
    NaN array shaped like `template`, with lead_time set to `horizon`, standing in
    for a failed request
    """
    hours = horizon_to_hours(horizon)
    data = np.full_like(np.asarray(template.values), np.nan, dtype="float32")
    return xr.DataArray(
        data,
        dims=template.dims,
        coords={"lead_time": [np.timedelta64(hours, "h")]},
    )


def _existing_archive_is_complete(out_path: Path) -> bool:
    """
    True if out_path holds a store that archive_run() marked complete (every
    request succeeded, nothing NaN-padded). Incomplete stores aren't patched,
    the whole run is fetched and written again.
    """
    if not out_path.exists():
        return False
    import zarr

    try:
        root = zarr.open(str(out_path), mode="r")
        return bool(root.attrs.get("archive_complete", False))
    except Exception:
        # Unreadable -> refetch and overwrite.
        return False


def resolve_reference_datetime(run_hour: str, date: str | None) -> str:
    """
    Reference time of the run to archive `date` at `run_hour`.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{date}T{run_hour}:00:00Z"


def fetch_variable_horizon(variable: str, horizon: str, reference_datetime: str) -> xr.DataArray | None:
    """
    The control and perturbed members for one (variable, horizon) of a run (21 members total).

    Returns None unless both requests succeed, so every horizon of a variable has the same eps size.
    """
    perturbed_req = ogd_api.Request(
        collection=COLLECTION,
        variable=variable,
        reference_datetime=reference_datetime,
        perturbed=True,
        horizon=horizon,
    )
    control_req = ogd_api.Request(
        collection=COLLECTION,
        variable=variable,
        reference_datetime=reference_datetime,
        perturbed=False,
        horizon=horizon,
    )
    try:
        perturbed_da = ogd_api.get_from_ogd(perturbed_req)
        control_da = ogd_api.get_from_ogd(control_req)
    except Exception as exc:  # On any failure process later (log, skip, NaN-pad)
        log.warning("Failed to fetch %s @ %s: %s", variable, horizon, exc)
        return None

    if "eps" not in control_da.dims:
        # A single member can come back without an eps dim at all
        control_da = control_da.expand_dims(eps=[0])

    da = xr.concat([control_da, perturbed_da], dim="eps")

    # Warn rather than fail: an unexpected member count needs a look, but the data is still worth keeping.
    member_dims = [d for d in da.dims if d == "eps" or "member" in d.lower() or "perturbation" in d.lower() or "realization" in d.lower()]
    if not member_dims:
        log.warning(
            "%s @ %s: no obvious ensemble-member dimension found in dims=%s "
            "-- got a single member instead of the full ensemble? Inspect "
            "before trusting this archive.",
            variable, horizon, da.dims,
        )
    elif da.sizes.get("eps") != 21:
        log.warning(
            "%s @ %s: expected 21 ensemble members (1 control + 20 "
            "perturbed), got %d -- inspect before trusting this archive.",
            variable, horizon, da.sizes.get("eps"),
        )
    return da


def archive_run(run_hour: str, date: str | None) -> Path:
    # Fast path for the common case of the run already being complete
    candidate_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    candidate_out_path = OUT_ROOT / candidate_date / f"{run_hour}z.zarr"
    if _existing_archive_is_complete(candidate_out_path):
        log.info(
            "%s already fully archived (%s) -- nothing to do.",
            candidate_out_path, run_hour,
        )
        return candidate_out_path

    _configure_earthkit_cache()
    _silence_earthkit_download_progress_bars()
    reference_datetime = resolve_reference_datetime(run_hour, date)
    run_date = reference_datetime[:10]

    out_dir = OUT_ROOT / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_hour}z.zarr"

    horizons = build_horizons()
    log.info(
        "Archiving %s, ref_time=%s, %d variables x %d horizons = %d requests",
        COLLECTION, reference_datetime, len(VARIABLES), len(horizons), len(VARIABLES) * len(horizons),
    )

    n_ok, n_fail = 0, 0
    n_written = 0
    store_initialized = False

    # Write each variable as soon as it's fetched. One variable is 1-2 GB in
    # memory (283,876 cells x 21 members x 73 leads x 4 B)
    for variable in VARIABLES:
        fetched: dict[str, xr.DataArray] = {}
        for horizon in horizons:
            da = fetch_variable_horizon(variable, horizon, reference_datetime)
            if da is None:
                n_fail += 1
                continue
            fetched[horizon] = da
            n_ok += 1

        if not fetched:
            log.warning("No data at all for %s this run -- omitted from archive.", variable)
            continue

        if len(fetched) < len(horizons):
            missing = [h for h in horizons if h not in fetched]
            log.warning(
                "%s: %d/%d horizons missing (%s) -- padding with NaN so the "
                "zarr store's lead_time dimension stays consistent across "
                "variables. Real data for those horizons is gone (24h window) "
                "and will need re-fetching next run if it's still needed.",
                variable, len(missing), len(horizons), ", ".join(missing),
            )

        # NaN-fill failed horizons so every variable has len(horizons) lead times
        template = next(iter(fetched.values()))
        # Not `fetched.get(h) or ...`: bool() of a multi-element DataArray raises.
        arrays = [fetched[h] if h in fetched else make_nan_placeholder(template, h) for h in horizons]

        # Concatenate along the existing length-1 lead_time dim. Stacking along a
        # new dim would first outer-join every array to all 73 lead times, a ~73x
        # memory blow-up that gets the process OOM-killed. join="exact" turns any
        # coordinate mismatch into an immediate error.
        var_da = xr.concat(arrays, dim="lead_time", join="exact")
        del arrays, fetched

        ds = xr.Dataset({variable: var_da})
        del var_da

        if not store_initialized:
            # Root attrs are set after the loop, not via ds.attrs: every to_zarr()
            # call, mode="a" included, replaces the root attrs wholesale
            log.info("Writing zarr store: %s (variable=%s)", out_path, variable)
            # No consolidated=True: not part of Zarr v3, it only adds a warning
            ds.to_zarr(out_path, mode="w")
            store_initialized = True
        else:
            log.info("Appending to zarr store: %s (variable=%s)", out_path, variable)
            ds.to_zarr(out_path, mode="a")
        del ds
        n_written += 1

    if n_written == 0:
        raise RuntimeError(
            f"Archived nothing for run {reference_datetime} ({n_fail} failures, 0 successes). "
            "Check credentials/connectivity/variable names before assuming the day is lost."
        )

    # Provenance and completeness, written once through zarr directly so no later to_zarr() call can replace them
    import zarr

    root = zarr.open(str(out_path), mode="a")
    root.attrs["reference_datetime"] = reference_datetime
    root.attrs["collection"] = COLLECTION
    root.attrs["archived_at"] = datetime.now(timezone.utc).isoformat()
    # Read by _existing_archive_is_complete on the next invocation
    root.attrs["archive_complete"] = n_fail == 0 and n_written == len(VARIABLES)
    root.attrs["n_ok"] = n_ok
    root.attrs["n_fail"] = n_fail
    root.attrs["n_written"] = n_written

    log.info(
        "Done. %d/%d (variable, horizon) requests succeeded, %d/%d variables written.",
        n_ok, n_ok + n_fail, n_written, len(VARIABLES),
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hour", default="00", choices=["00", "06", "12", "18"])
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD of the run to archive (default: today, UTC). "
        "Only runs from the last ~24h are available.",
    )
    args = parser.parse_args()
    archive_run(run_hour=args.run_hour, date=args.date)
