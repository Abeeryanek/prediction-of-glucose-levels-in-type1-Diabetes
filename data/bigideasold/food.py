"""
fix_food_features.py

Replaces old windowed food columns (calorie_2h, calorie_8h, etc.)
with plain 5-min resampled columns (calorie, total_carb, etc.).

NaN = no food logged in that 5-min window (intentional, not an error).
Multiple food items at the same timestamp are summed into one row.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
INPUT_DIR    = "data/bigideas"
OUTPUT_DIR   = Path("bigideanew")
FOOD_LOG_DIR = "data/bigideas"

PATIENT_IDS = [f"{i:03d}" for i in range(1, 17)]  # "001" .. "016"

FOOD_COLS = [
    "calorie", "total_carb", "dietary_fiber",
    "sugar", "protein", "total_fat",
]

# Only these suffixes mark the OLD windowed columns to remove
OLD_SUFFIXES = ["_2h", "_8h", "_24h"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STEP 1 — drop old windowed food columns
# ============================================================================
def drop_old_food_columns(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    """
    Drop columns like calorie_2h, calorie_8h, calorie_24h, protein_2h …
    Does NOT touch plain columns like 'calorie' if already present.
    """
    targets = {
        f"{fc}{sfx}"
        for fc in FOOD_COLS
        for sfx in OLD_SUFFIXES
    }
    to_drop = [c for c in df.columns if c in targets]

    if to_drop:
        print(f"  [{pid}] Dropping {len(to_drop)} windowed cols: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print(f"  [{pid}] No windowed food columns found.")

    return df


# ============================================================================
# STEP 2 — load food log and resample to 5-min grid
# ============================================================================
def load_and_resample_food_log(pid: str) -> pd.DataFrame | None:
    """
    Reads Food_Log_{pid}.csv.

    Your data has time_begin as a full datetime string, e.g.:
        2020-03-22 10:32:00

    Multiple items logged at the same minute are summed (correct behaviour —
    e.g. the lunch at 12:30 with 9 different food items).

    Returns a DataFrame indexed on a 5-min grid with columns = FOOD_COLS,
    or None if the file is missing / unreadable.

    NaN in the result means "no food was logged in this 5-min window".
    """
    path = os.path.join(FOOD_LOG_DIR, f"Food_Log_{pid}.csv")
    if not os.path.exists(path):
        print(f"  [{pid}] ✗ Food_Log_{pid}.csv not found → food cols will be NaN")
        return None

    # ------------------------------------------------------------------ load
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ------------------------------------------------------------------ parse time_begin
    # Your real data: time_begin is already "2020-03-22 10:32:00"
    if "time_begin" not in df.columns:
        print(f"  [{pid}] ✗ 'time_begin' column missing → food cols will be NaN")
        return None

    df["time_begin"] = pd.to_datetime(df["time_begin"], errors="coerce")

    # Drop timezone if present (parquet is tz-naive)
    if df["time_begin"].dt.tz is not None:
        df["time_begin"] = df["time_begin"].dt.tz_localize(None)

    # Report and remove rows with unparseable timestamps
    bad_ts = df["time_begin"].isna().sum()
    if bad_ts > 0:
        print(f"  [{pid}] ⚠ Dropping {bad_ts} food rows with unparseable time_begin")
        df = df.dropna(subset=["time_begin"])

    if df.empty:
        print(f"  [{pid}] ✗ Food log empty after timestamp cleaning")
        return None

    df = df.set_index("time_begin").sort_index()

    # ------------------------------------------------------------------ coerce numerics
    for col in FOOD_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"  [{pid}] ⚠ '{col}' not in food log → will be NaN")

    # ------------------------------------------------------------------ resample to 5-min sums
    # min_count=1 → a bin with NO food rows stays NaN (not 0)
    # This handles multiple items at the same timestamp correctly:
    #   e.g. 9 foods at 12:30 all fall in the 12:30 bin and get summed
    available = [c for c in FOOD_COLS if c in df.columns]
    food_5min = df[available].resample("5min").sum(min_count=1)

    # Add any missing FOOD_COLS as NaN columns so schema stays consistent
    for col in FOOD_COLS:
        if col not in food_5min.columns:
            food_5min[col] = np.nan

    food_5min = food_5min[FOOD_COLS]  # enforce column order

    print(
        f"  [{pid}] Food log loaded: {len(df)} raw rows → "
        f"{food_5min.notna().any(axis=1).sum()} non-empty 5-min bins "
        f"(out of {len(food_5min)} total bins)"
    )
    return food_5min


# ============================================================================
# STEP 3 — alignment check (informational only, never blocks the merge)
# ============================================================================
def check_alignment(food_5min: pd.DataFrame | None,
                    parquet_ts: pd.Series,
                    pid: str) -> None:
    if food_5min is None or food_5min.empty:
        return

    f_start, f_end = food_5min.index.min(), food_5min.index.max()
    p_start, p_end = parquet_ts.min(),       parquet_ts.max()

    overlap_start = max(f_start, p_start)
    overlap_end   = min(f_end,   p_end)

    if overlap_start > overlap_end:
        print(
            f"  [{pid}] ⚠ WARNING: food log ({f_start} → {f_end}) "
            f"does NOT overlap parquet ({p_start} → {p_end}). "
            f"All food columns will be NaN."
        )
        return

    # Check parquet timestamps form a clean 5-min grid
    diffs = parquet_ts.sort_values().diff().dropna()
    irregular = diffs[diffs != pd.Timedelta(minutes=5)]

    if irregular.empty:
        print(
            f"  [{pid}] ✓ Clean 5-min parquet grid. "
            f"Food overlap: {overlap_start} → {overlap_end}"
        )
    else:
        print(
            f"  [{pid}] ⚠ Parquet has {len(irregular)} non-5-min gaps "
            f"(max gap = {irregular.max()}). "
            f"Merge still works — non-matching timestamps get NaN food values."
        )


# ============================================================================
# STEP 4 — merge food onto parquet rows
# ============================================================================
def merge_food(df_p: pd.DataFrame,
               food_5min: pd.DataFrame | None,
               pid: str) -> pd.DataFrame:
    """
    Join food_5min onto df_p using Timestamp as the key.

    We floor each parquet Timestamp to the nearest 5-min boundary so it
    lands on the same grid as the resampled food index.

    Example
    -------
    Parquet Timestamp  17:32:00  →  floored  17:30:00  →  looks up food at 17:30
    Parquet Timestamp  17:30:00  →  floored  17:30:00  →  looks up food at 17:30

    In your sample data the parquet is already on exact 5-min marks
    (17:20, 17:25 …) so floor() is a no-op — but it makes the code
    robust to any future data with sub-5-min resolution.
    """
    if food_5min is not None and not food_5min.empty:
        # Floor parquet timestamps to 5-min so they match the food index
        floored = df_p["Timestamp"].dt.floor("5min")

        # Build lookup Series for each food column keyed by floored timestamp
        for col in FOOD_COLS:
            # .map() returns NaN automatically for timestamps not in food_5min
            df_p[col] = floored.map(food_5min[col]).astype("float64")

        non_null_rows = df_p[FOOD_COLS].notna().any(axis=1).sum()
        print(
            f"  [{pid}] ✓ Food merged — "
            f"{non_null_rows:,} parquet rows have at least one food value "
            f"({non_null_rows / len(df_p) * 100:.1f}%)"
        )
    else:
        # No food log available — add all columns as NaN
        for col in FOOD_COLS:
            df_p[col] = np.nan
        print(f"  [{pid}] Food columns added as NaN (no food log)")

    # Enforce float64 on every food column for consistent parquet schema
    for col in FOOD_COLS:
        df_p[col] = df_p[col].astype("float64")

    return df_p


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    separator = "=" * 65
    print(f"\n{separator}")
    print(f"FOOD COLUMN FIX — {len(PATIENT_IDS)} patients")
    print(separator)

    results = []

    for pid in PATIENT_IDS:
        in_path = os.path.join(INPUT_DIR, f"clean_patient_{pid}.parquet")

        if not os.path.exists(in_path):
            print(f"\n[{pid}] ✗ Skipped — {in_path} not found")
            results.append({"pid": pid, "status": "skipped"})
            continue

        print(f"\n[{pid}] Loading {in_path}")
        df_p = pd.read_parquet(in_path)

        # Timestamp hygiene
        df_p["Timestamp"] = pd.to_datetime(df_p["Timestamp"])
        if df_p["Timestamp"].dt.tz is not None:
            df_p["Timestamp"] = df_p["Timestamp"].dt.tz_localize(None)
        df_p = df_p.sort_values("Timestamp").reset_index(drop=True)

        print(f"  [{pid}] {df_p.shape[0]:,} rows × {df_p.shape[1]} cols loaded")

        # 1. Remove old windowed food columns
        df_p = drop_old_food_columns(df_p, pid)

        # 2. Load + resample food log to 5-min grid
        food_5min = load_and_resample_food_log(pid)

        # 3. Alignment diagnostics
        check_alignment(food_5min, df_p["Timestamp"], pid)

        # 4. Merge food onto parquet
        df_p = merge_food(df_p, food_5min, pid)

        # 5. Save
        out_path = OUTPUT_DIR / f"clean_patient_{pid}.parquet"
        df_p.to_parquet(out_path, index=False)

        # Per-column NaN summary
        nan_rates = {
            col: f"{df_p[col].isna().mean() * 100:.1f}%"
            for col in FOOD_COLS
        }
        print(f"  [{pid}] Saved → {out_path}")
        print(f"  [{pid}] Shape: {df_p.shape[0]:,} rows × {df_p.shape[1]} cols")
        print(f"  [{pid}] Food NaN rates: {nan_rates}")

        results.append({
            "pid":    pid,
            "status": "ok",
            "rows":   df_p.shape[0],
            "cols":   df_p.shape[1],
        })

    # Final summary
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)
    for r in results:
        print(r)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()