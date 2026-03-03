from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIME_BASE = pd.Timestamp("1899-12-30", tz="UTC")

TEAM_TABLES = [
    "GreenhouseClimate.csv",
    "GrodanSens.csv",
    "Resources.csv",
    "Production.csv",
    "LabAnalysis.csv",
    "CropParameters.csv",
    "TomQuality.csv",
]


def _normalize_columns(columns: list[str]) -> list[str]:
    out = []
    for col in columns:
        c = str(col).replace("\t", " ").replace("%", "pct").strip()
        c = "_".join(c.split())
        out.append(c)
    return out


def _read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep=r"[,\t]+", engine="python", low_memory=False)

    # Fix malformed TomQuality in Reference where Weight and DMC are merged in one header.
    if path.name == "TomQuality.csv" and any("Weight" in c and "DMC" in c for c in df.columns):
        try:
            df = pd.read_csv(path, sep=r"[,\t]+", engine="python", low_memory=False)
        except Exception:
            pass

    df.columns = _normalize_columns(list(df.columns))

    # Drop parser artifacts.
    keep_cols = [c for c in df.columns if not c.lower().startswith("unnamed")]
    df = df[keep_cols]

    return df


def _parse_time(series: pd.Series) -> pd.DatetimeIndex:
    val = pd.to_numeric(series, errors="coerce")
    ts = TIME_BASE + pd.to_timedelta(val, unit="D")
    return pd.DatetimeIndex(ts)


def _coerce_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    exclude = exclude or set()
    for col in out.columns:
        if col in exclude:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _resample_numeric(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    if not numeric_cols:
        return df
    out = (
        df.set_index("timestamp")[numeric_cols]
        .sort_index()
        .resample(rule)
        .mean()
        .reset_index()
    )
    return out


def _iqr_clip(df: pd.DataFrame, k: float) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out[col] = out[col].clip(lo, hi)
    return out


def _robust_z_clip(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        med = out[col].median()
        mad = (out[col] - med).abs().median()
        if pd.isna(mad) or mad == 0:
            continue
        rz = 0.6745 * (out[col] - med) / mad
        out.loc[rz.abs() > thresh, col] = np.nan
    return out


def _ewma(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].ewm(alpha=alpha, adjust=False).mean()
    return out


def _prepare_team_table(path: Path, table_name: str, resample_rule: str) -> pd.DataFrame:
    file_path = path / table_name
    if not file_path.exists():
        return pd.DataFrame(columns=["timestamp"])

    df = _read_csv_robust(file_path)
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if not time_col:
        return pd.DataFrame(columns=["timestamp"])

    df = _coerce_numeric(df)
    df["timestamp"] = _parse_time(df[time_col])
    df = df.drop(columns=[time_col]).dropna(subset=["timestamp"])  # keep valid timeline only
    df = _resample_numeric(df, resample_rule)
    return df


def _build_canonical_features(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    def _pick(*candidates: str) -> pd.Series:
        for c in candidates:
            if c in df:
                return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    df["air_temperature"] = _pick("Tair")
    df["air_humidity"] = _pick("Rhair")
    df["co2_ppm"] = _pick("CO2air")

    par = _pick("Tot_PAR")
    df["light_lux"] = par * 54.0

    df["soil_temperature"] = pd.concat([_pick("t_slab1"), _pick("t_slab2")], axis=1).mean(axis=1)
    df["soil_humidity"] = pd.concat([_pick("WC_slab1"), _pick("WC_slab2")], axis=1).mean(axis=1)
    df["soil_ec"] = pd.concat([_pick("EC_slab1"), _pick("EC_slab2")], axis=1).mean(axis=1)
    df["soil_ph"] = _pick("pH_drain_PC", "drain_PH", "irr_PH")

    df["soil_n"] = _pick("irr_NO3", "drain_NO3")
    df["soil_p"] = _pick("irr_PO4", "drain_PO4")
    df["soil_k"] = _pick("irr_K", "drain_K")

    # Action labels for imitation.
    df["target_temperature_c"] = _pick("t_heat_sp", "t_vent_sp").fillna(df["air_temperature"])
    df["target_humidity_rh_pct"] = (80 - _pick("HumDef") * 2.5).clip(50, 90)
    df["vent_open_pct"] = (_pick("VentLee") * 100).clip(0, 100)
    df["fan_speed_pct"] = (_pick("Ventwind") * 100).clip(0, 100)

    water_sup = _pick("water_sup")
    water_interval_min = _pick("water_sup_intervals_sp_min")
    irr_daily = _pick("Irr")

    df["irrigation_duration_s"] = (water_interval_min * 60).fillna(irr_daily.fillna(0) * 30)
    df["irrigation_flow_lph"] = (water_sup * 60).fillna(100).clip(20, 200)
    df["irrigation_on"] = (df["irrigation_duration_s"].fillna(0) > 0).astype(int)

    # Outcome/reward proxies.
    df["yield_quality_score"] = (
        _pick("ProdA").fillna(0)
        + 0.6 * _pick("ProdB").fillna(0)
        + 0.2 * _pick("Flavour").fillna(0)
        + 0.15 * _pick("TSS").fillna(0)
        - 0.1 * _pick("Acid").fillna(0)
    )

    df["water_use_proxy"] = _pick("Irr", "water_sup").fillna(0)
    df["energy_proxy"] = (
        _pick("Heat_cons").fillna(0)
        + _pick("ElecHigh").fillna(0)
        + _pick("ElecLow").fillna(0)
        + 0.5 * _pick("PipeLow").fillna(0)
    )

    df["crop_day"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / (3600 * 24)

    return df


def _merge_team_frames(team_dir: Path, include_weather: bool, weather_df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
    frames = {}
    for table in TEAM_TABLES:
        frames[table] = _prepare_team_table(team_dir, table, resample_rule)

    merged = None
    for table in TEAM_TABLES:
        f = frames[table]
        if f.empty:
            continue
        if merged is None:
            merged = f
        else:
            merged = merged.merge(f, on="timestamp", how="outer")

    if merged is None:
        return pd.DataFrame()

    if include_weather and not weather_df.empty:
        merged = merged.merge(weather_df, on="timestamp", how="left")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["team"] = team_dir.name
    merged = _build_canonical_features(merged)
    return merged


def load_weather(data_root: Path, resample_rule: str) -> pd.DataFrame:
    path = data_root / "Weather" / "Weather.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    weather = _read_csv_robust(path)
    time_col = next((c for c in weather.columns if "time" in c.lower()), None)
    if not time_col:
        return pd.DataFrame(columns=["timestamp"])
    weather = _coerce_numeric(weather)
    weather["timestamp"] = _parse_time(weather[time_col])
    weather = weather.drop(columns=[time_col]).dropna(subset=["timestamp"])
    weather = _resample_numeric(weather, resample_rule)
    return weather


def load_agc_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    data_root = Path(cfg["paths"]["data_root"])
    teams = cfg["data"]["teams"]
    resample_rule = cfg["data"]["resample_rule"]

    weather_df = load_weather(data_root, resample_rule) if cfg["data"].get("include_weather", True) else pd.DataFrame()

    frames = []
    for team in teams:
        team_dir = data_root / team
        if not team_dir.exists():
            continue
        frame = _merge_team_frames(team_dir, cfg["data"].get("include_weather", True), weather_df, resample_rule)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No team data loaded from {data_root}")

    data = pd.concat(frames, ignore_index=True).sort_values(["team", "timestamp"]).reset_index(drop=True)

    out_cfg = cfg["data"]["outlier"]
    if out_cfg.get("enabled", True):
        if out_cfg.get("method", "iqr") == "iqr":
            data = _iqr_clip(data, float(out_cfg.get("iqr_k", 3.0)))
        else:
            data = _robust_z_clip(data, float(out_cfg.get("robust_z_thresh", 5.0)))

    data = data.sort_values(["team", "timestamp"]).reset_index(drop=True)

    smooth_cfg = cfg["data"]["smoothing"]
    if smooth_cfg.get("enabled", False):
        parts = []
        for team, grp in data.groupby("team", sort=False):
            sm = _ewma(grp, alpha=float(smooth_cfg.get("alpha", 0.35)))
            sm["team"] = team
            sm["timestamp"] = grp["timestamp"].values
            parts.append(sm)
        data = pd.concat(parts, ignore_index=True).sort_values(["team", "timestamp"]).reset_index(drop=True)

    # Final robust filling for downstream training.
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data.groupby("team")[numeric_cols].ffill().bfill()
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    downsample_step = int(cfg["data"].get("downsample_step", 1))
    if downsample_step > 1:
        parts = []
        for _, grp in data.groupby("team", sort=False):
            parts.append(grp.iloc[::downsample_step].copy())
        data = pd.concat(parts, ignore_index=True).sort_values(["team", "timestamp"]).reset_index(drop=True)

    return data


def split_time_blocks(df: pd.DataFrame, val_fraction: float, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chunks = []
    for _, grp in df.groupby("team", sort=False):
        grp = grp.sort_values("timestamp")
        n = len(grp)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        n_train = max(0, n - n_val - n_test)
        chunks.append((grp.iloc[:n_train], grp.iloc[n_train:n_train + n_val], grp.iloc[n_train + n_val:]))

    train = pd.concat([c[0] for c in chunks], ignore_index=True)
    val = pd.concat([c[1] for c in chunks], ignore_index=True)
    test = pd.concat([c[2] for c in chunks], ignore_index=True)
    return train, val, test
