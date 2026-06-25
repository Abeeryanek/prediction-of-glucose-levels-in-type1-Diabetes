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
        rows.append({"ts": e.get("ts"), "bolus": float(e.get("dose") or "nan")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], format=_TS_FMT)
    return df


def load_patient(xml_path: Path | str) -> pd.DataFrame:
    """
    Parse one OhioT1DM XML file and return a 5-minute-indexed DataFrame.

    Columns
    -------
    ts        : datetime index
    glucose   : CGM reading [mg/dL]; NaN where the sensor was unavailable
    carbs     : meal carbohydrates [g];  NaN where no meal was logged
    bolus     : insulin bolus dose [U];  NaN where no bolus was delivered
    heartrate : Basis wristband heart rate (NaN when device was off)
    gsr       : Basis wristband galvanic skin response (NaN when device was off)

    Event NaN (carbs, bolus) should be filled with 0 by the pipeline.
    Sensor NaN rows (glucose, heartrate, gsr) should be dropped by the pipeline.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    glucose = _parse_value_events(root, "glucose_level", "value").rename(
        columns={"value": "glucose"}
    )
    meal = _parse_value_events(root, "meal", "carbs")
    bolus = _parse_bolus(root)
    hr = _parse_value_events(root, "basis_heart_rate", "value").rename(
        columns={"value": "heartrate"}
    )
    gsr = _parse_value_events(root, "basis_gsr", "value").rename(
        columns={"value": "gsr"}
    )

    # Build 5-minute grid anchored on CGM timestamps
    glucose = glucose.sort_values("ts").set_index("ts")
    idx = pd.date_range(glucose.index.min(), glucose.index.max(), freq=_RESAMPLE_FREQ)
    df = glucose.reindex(idx)
    df.index.name = "ts"

    def _merge(events: pd.DataFrame, col: str) -> None:
        if events.empty:
            df[col] = float("nan")
            return
        ev = events.sort_values("ts").set_index("ts")
        # Sum events that land in the same 5-minute bin (e.g. split boluses)
        ev = ev.groupby(level=0)[col].sum()
        df[col] = ev.reindex(idx)

    _merge(meal, "carbs")
    _merge(bolus, "bolus")

    for aux, col in [(hr, "heartrate"), (gsr, "gsr")]:
        if not aux.empty:
            ev = aux.sort_values("ts").set_index("ts")[col].reindex(idx)
            df[col] = ev
        else:
            df[col] = float("nan")

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
        xml_file.stem: load_patient(xml_file)
        for xml_file in sorted(split_dir.glob("*.xml"))
    }
