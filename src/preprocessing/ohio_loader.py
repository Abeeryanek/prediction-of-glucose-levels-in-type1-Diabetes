"""
Load and merge OhioT1DM XML files into tidy 5-minute-resolution DataFrames.

The glucose column is always named 'glucose' so that downstream pipeline code
works identically across OhioT1DM and BigIDEAS data.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

_TS_FMT = "%d-%m-%Y %H:%M:%S"
_RESAMPLE_FREQ = "5min"

COHORT_2018 = ["559", "563", "570", "575", "588", "591"]
COHORT_2020 = ["540", "544", "552", "567", "584", "596"]


def _parse_value_events(root: ET.Element, tag: str, value_attr: str) -> pd.DataFrame:
    """Extract (ts, value_attr) pairs from an XML subtree of <event> elements."""
    node = root.find(tag)
    if node is None:
        return pd.DataFrame(columns=["ts", value_attr])
    rows = []
    for e in node.findall("event"):
        rows.append(
            {"ts": e.get("ts"), value_attr: float(e.get(value_attr) or "nan")}
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], format=_TS_FMT)
    return df


def _parse_bolus(root: ET.Element) -> pd.DataFrame:
    """Extract bolus dose events."""
    node = root.find("bolus")
    if node is None:
        return pd.DataFrame(columns=["ts", "bolus"])
    rows = []
    for e in node.findall("event"):
        rows.append({"ts": e.get("ts_begin"), "bolus": float(e.get("dose") or "nan")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], format=_TS_FMT)
    return df


def load_patient(xml_path: Path | str) -> pd.DataFrame:
    """
    Parse one OhioT1DM XML file and return a 5-minute-indexed DataFrame.

    Columns always present
    ----------------------
    ts        : datetime
    glucose   : CGM reading [mg/dL]; NaN where sensor was unavailable
    carbs     : meal carbohydrates [g]; NaN where no meal was logged
    bolus     : insulin bolus dose [U]; NaN where no bolus was delivered
    gsr       : galvanic skin response; NaN when device was off
    skin_temp : skin temperature;       NaN when device was off

    2018 cohort only  (file contains <basis_heart_rate>)
    -----------------------------------------------------
    heartrate : Basis wristband heart rate
    steps     : Basis wristband step count

    2020 cohort only  (file contains <acceleration>)
    -------------------------------------------------
    acceleration : Empatica Embrace accelerometer reading

    Event NaN (carbs, bolus) are filled with 0 by the pipeline.
    Sensor NaN rows are dropped by the pipeline — never interpolated.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ── Always-present signals ───────────────────────────────────────────────
    glucose = _parse_value_events(root, "glucose_level", "value").rename(
        columns={"value": "glucose"}
    )
    meal  = _parse_value_events(root, "meal", "carbs")
    bolus = _parse_bolus(root)
    gsr = _parse_value_events(root, "basis_gsr", "value").rename(
        columns={"value": "gsr"}
    )
    skin_temp = _parse_value_events(root, "basis_skin_temperature", "value").rename(
        columns={"value": "skin_temp"}
    )

    # ── Build 5-minute grid anchored on CGM timestamps ───────────────────────
    # resample().mean() tolerates sub-second timing jitter in CGM timestamps;
    # reindex() would miss readings offset by even 1 second, causing ~85% NaN.
    glucose_s = glucose.sort_values("ts").set_index("ts")["glucose"]
    glucose_resampled = glucose_s.resample(_RESAMPLE_FREQ).mean()
    df = glucose_resampled.to_frame()
    df.index.name = "ts"
    idx = df.index

    def _merge(events: pd.DataFrame, col: str) -> None:
        """Merge event-type signals: sum duplicates within the same 5-min bin."""
        if events.empty:
            df[col] = float("nan")
            return
        ev = events.sort_values("ts").set_index("ts")
        ev = ev.groupby(level=0)[col].sum()
        df[col] = ev.reindex(idx)

    def _merge_sensor(aux: pd.DataFrame, col: str) -> None:
        """Merge continuous sensor signals: reindex onto the 5-min grid."""
        if not aux.empty:
            df[col] = aux.sort_values("ts").set_index("ts")[col].reindex(idx)
        else:
            df[col] = float("nan")

    _merge(meal, "carbs")
    _merge(bolus, "bolus")
    _merge_sensor(gsr, "gsr")
    _merge_sensor(skin_temp, "skin_temp")

    # ── Cohort-specific signals ──────────────────────────────────────────────
    if root.find("basis_heart_rate") is not None:
        # 2018 cohort — Basis wristband
        hr = _parse_value_events(root, "basis_heart_rate", "value").rename(
            columns={"value": "heartrate"}
        )
        steps = _parse_value_events(root, "basis_steps", "value").rename(
            columns={"value": "steps"}
        )
        _merge_sensor(hr, "heartrate")
        _merge_sensor(steps, "steps")
    elif root.find("acceleration") is not None:
        # 2020 cohort — Empatica Embrace
        accel = _parse_value_events(root, "acceleration", "value").rename(
            columns={"value": "acceleration"}
        )
        _merge_sensor(accel, "acceleration")

    return df.reset_index()


def load_split(split_dir: Path | str) -> dict[str, pd.DataFrame]:
    """
    Load all XML files in a directory and return {patient_id: DataFrame}.

    split_dir should be one of:
        data/ohio/2018/train/  data/ohio/2018/test/
        data/ohio/2020/train/  data/ohio/2020/test/
    """
    split_dir = Path(split_dir)
    return {
        xml_file.stem.split("-")[0]: load_patient(xml_file)
        for xml_file in sorted(split_dir.glob("*.xml"))
    }
