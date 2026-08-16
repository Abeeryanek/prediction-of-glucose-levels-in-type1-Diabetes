"""
Glucdict dataset loader for blood glucose forecasting.

Glucdict provides ~10 days of simultaneous CGM and wearable sensor data
(Dexcom G6 at 5-min resolution + smartwatch IMU / heart rate at high
frequency) from 13 participants in our working copy. The dataset's published
source characterizes a 12-participant version as a general-population
cohort (mostly non-diabetic students; only 1 had Type 1 diabetes).

Reference
---------
Glucdict Dataset — available at the project data directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ALL_USERS: list[str] = [
    "User1", "User3", "User4", "User5", "User6", "User7",
    "User8", "User9", "User10", "User12", "User13", "User14", "User15",
]

GLUCDICT_BASE: Path = Path(
    r"C:\Users\mefte\Desktop\Studium\Bachleor Projekt"
    r"\prediction-of-glucose-levels-in-type1-Diabetes"
    r"\Glucdict Dataset\Glucdict Dataset"
)

# Dexcom 'Low' / Hebrew 'נמוך' sentinel — device reports <40 mg/dL as text
_LOW_FILL: float = 39.0


# ── Watch sensor helpers ───────────────────────────────────────────────────────

def _load_watch_sensors(watch_dir: Path) -> tuple[pd.Series, pd.Series]:
    """
    Read watch CSV files in 500k-row chunks, filter to sensor types 1 and 21,
    and return (acc_5min, hr_5min) resampled at 5-minute resolution.

    File priority: SensorReader*.csv (consolidated) over Firebase_*.csv (daily).
    Watch timestamps are Unix-milliseconds (UTC), stored as naive datetimes.
    CGM timestamps are naive local time (UTC+2 for this cohort). The 2-hour
    offset shifts watch features relative to the true CGM moment but does not
    cause NaN gaps — both signals share the same naive-datetime range.

    Sensor type legend
    ------------------
    1  — Accelerometer (x, y, z in m/s²) → acc_magnitude = sqrt(x²+y²+z²)
    21 — Heart rate / PPG  (BPM in the x column)
    """
    sensor_files = sorted(watch_dir.glob("*SensorReader*.csv"))
    if not sensor_files:
        sensor_files = sorted(watch_dir.glob("Firebase_*.csv"))
    if not sensor_files:
        empty: pd.Series = pd.Series(dtype=float)
        return empty.rename("acc_magnitude"), empty.rename("heartrate")

    acc_parts: list[pd.DataFrame] = []
    hr_parts:  list[pd.DataFrame] = []

    for path in sensor_files:
        reader = pd.read_csv(
            path,
            header=None,
            names=["sensor_type", "ts_ms", "x", "y", "z"],
            low_memory=False,
            chunksize=500_000,
        )
        for chunk in reader:
            chunk = chunk[chunk["sensor_type"].isin([1, 21])].copy()
            if chunk.empty:
                continue

            chunk["ts_ms"] = pd.to_numeric(chunk["ts_ms"], errors="coerce")
            chunk = chunk.dropna(subset=["ts_ms"])
            # Convert Unix-ms to naive datetime (UTC reference, tz stripped)
            chunk["dt"] = pd.to_datetime(
                chunk["ts_ms"].astype(np.int64), unit="ms"
            )

            acc = chunk[chunk["sensor_type"] == 1]
            hr  = chunk[chunk["sensor_type"] == 21]

            if not acc.empty:
                x = pd.to_numeric(acc["x"], errors="coerce").to_numpy()
                y = pd.to_numeric(acc["y"], errors="coerce").to_numpy()
                z = pd.to_numeric(acc["z"], errors="coerce").to_numpy()
                mag = np.sqrt(x**2 + y**2 + z**2)
                acc_parts.append(
                    pd.DataFrame({"dt": acc["dt"].to_numpy(), "magnitude": mag})
                )

            if not hr.empty:
                bpm = pd.to_numeric(hr["x"], errors="coerce").to_numpy()
                hr_parts.append(
                    pd.DataFrame({"dt": hr["dt"].to_numpy(), "bpm": bpm})
                )

    if acc_parts:
        acc_df  = pd.concat(acc_parts, ignore_index=True).dropna()
        acc_5min = (
            acc_df.set_index("dt").sort_index()["magnitude"]
            .resample("5min").mean()
            .rename("acc_magnitude")
        )
    else:
        acc_5min = pd.Series(dtype=float, name="acc_magnitude")

    if hr_parts:
        hr_df   = pd.concat(hr_parts, ignore_index=True).dropna()
        hr_5min = (
            hr_df.set_index("dt").sort_index()["bpm"]
            .resample("5min").mean()
            .rename("heartrate")
        )
    else:
        hr_5min = pd.Series(dtype=float, name="heartrate")

    return acc_5min, hr_5min


# ── Public API ─────────────────────────────────────────────────────────────────

def load_patient(user_dir: Path | str) -> pd.DataFrame:
    """
    Load all Glucdict signals for one user into a 5-minute aligned DataFrame.

    Parameters
    ----------
    user_dir : path to a user directory, e.g. ``GLUCDICT_BASE / 'User1'``

    Returns
    -------
    pd.DataFrame
        Columns: ts, glucose, heartrate, acc_magnitude, eat_event, drink_event

        - ``glucose``       : CGM reading in mg/dL; 'Low'/'נמוך' → 39.0
        - ``heartrate``     : watch heart rate in BPM (NaN between measurements)
        - ``acc_magnitude`` : watch accelerometer magnitude sqrt(x²+y²+z²) in m/s²
        - ``eat_event``     : 1 if an eating activity started in this 5-min bin
        - ``drink_event``   : 1 if a drinking activity started in this 5-min bin
    """
    user_dir = Path(user_dir)
    user_id  = user_dir.name  # e.g. 'User1'

    # ── a. Load CGM ────────────────────────────────────────────────────────────
    cgm_path = user_dir / "Glucose" / f"CGM_{user_id}.csv"
    cgm_raw  = pd.read_csv(cgm_path)
    cgm_raw["ts"] = pd.to_datetime(cgm_raw["Timestamp (YYYY-MM-DDThh:mm:ss)"])
    cgm_raw["glucose"] = (
        pd.to_numeric(cgm_raw["Glucose Value (mg/dL)"], errors="coerce")
        .fillna(_LOW_FILL)
    )
    cgm = (
        cgm_raw.set_index("ts")[["glucose"]]
        .resample("5min").mean()
    )

    # ── b. Load watch sensors ──────────────────────────────────────────────────
    watch_dir = user_dir / "Watch"
    if watch_dir.is_dir():
        acc_5min, hr_5min = _load_watch_sensors(watch_dir)
    else:
        acc_5min = pd.Series(dtype=float, name="acc_magnitude")
        hr_5min  = pd.Series(dtype=float, name="heartrate")

    # ── c. Load activities ─────────────────────────────────────────────────────
    act_path = user_dir / "Phone" / "Activities" / "Activities.csv"
    cgm_index = cgm.index

    if act_path.exists():
        act = pd.read_csv(
            act_path, header=None, names=["id", "type", "start", "end"]
        )
        act["start"] = pd.to_datetime(act["start"])

        def _make_event(activity_type: str) -> pd.Series:
            starts  = act.loc[act["type"] == activity_type, "start"]
            floored = starts.dt.floor("5min")
            counts  = floored.value_counts()
            # clip(upper=1): multiple events in same 5-min bin → still 1
            return (
                counts.reindex(cgm_index).fillna(0).clip(upper=1).astype(np.int8)
            )

        eat_event   = _make_event("eat").rename("eat_event")
        drink_event = _make_event("drink").rename("drink_event")
    else:
        eat_event   = pd.Series(0, index=cgm_index, name="eat_event",   dtype=np.int8)
        drink_event = pd.Series(0, index=cgm_index, name="drink_event", dtype=np.int8)

    # ── d-e. Merge all signals onto the 5-min CGM grid ────────────────────────
    grid = cgm.join(hr_5min,  how="left")
    grid = grid.join(acc_5min, how="left")
    grid["eat_event"]   = eat_event
    grid["drink_event"] = drink_event

    # ── f. Reset index so 'ts' is a regular column ────────────────────────────
    return grid.reset_index()


def load_all_users(base_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all 13 Glucdict users.

    Parameters
    ----------
    base_dir : root of the Glucdict dataset; defaults to ``GLUCDICT_BASE``

    Returns
    -------
    dict mapping user_id (e.g. ``'User1'``) to its 5-min DataFrame
    """
    base_dir = Path(base_dir) if base_dir is not None else GLUCDICT_BASE
    result: dict[str, pd.DataFrame] = {}

    for user_id in ALL_USERS:
        user_path = base_dir / user_id
        try:
            result[user_id] = load_patient(user_path)
            print(f"  {user_id}: {len(result[user_id])} rows loaded")
        except Exception as exc:
            print(f"  {user_id}: ERROR — {exc}")

    return result
