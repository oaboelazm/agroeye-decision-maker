from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LIVE_COLUMNS = [
    "soil_n",
    "soil_p",
    "soil_k",
    "soil_temperature",
    "soil_humidity",
    "soil_ph",
    "soil_ec",
    "air_temperature",
    "air_humidity",
    "light_lux",
    "co2_ppm",
]


@dataclass
class FeatureStore:
    feature_names: list[str]
    medians: dict[str, float]
    dtypes: dict[str, str]
    means: dict[str, float]
    stds: dict[str, float]
    imputer: SimpleImputer
    scaler: StandardScaler

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out)

    @staticmethod
    def load(path: str | Path) -> "FeatureStore":
        return joblib.load(path)


def _safe_slope(values: pd.Series) -> float:
    y = values.dropna().values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return 0.0
    return float(((x * (y - y.mean())).sum()) / denom)


def build_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    feat_cfg = cfg["features"]
    lags = feat_cfg.get("lags_steps", [1, 3, 6, 12, 24])
    rolls = feat_cfg.get("rolling_steps", [3, 6, 12, 24])

    all_frames = []
    for team, grp in df.groupby("team", sort=False):
        g = grp.sort_values("timestamp").copy()

        # Ensure stable live schema columns exist.
        for col in LIVE_COLUMNS:
            if col not in g:
                g[col] = np.nan

        for col in LIVE_COLUMNS + ["water_use_proxy", "energy_proxy", "yield_quality_score"]:
            if col not in g:
                continue
            for lag in lags:
                g[f"{col}_lag_{lag}"] = g[col].shift(lag)
            for win in rolls:
                r = g[col].rolling(win)
                g[f"{col}_roll_mean_{win}"] = r.mean()
                g[f"{col}_roll_min_{win}"] = r.min()
                g[f"{col}_roll_max_{win}"] = r.max()
                if feat_cfg.get("add_slope", True):
                    g[f"{col}_roll_slope_{win}"] = r.apply(_safe_slope, raw=False)

        if feat_cfg.get("add_diurnal", True):
            hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
            g["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            g["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        g["day_of_cycle"] = (g["timestamp"] - g["timestamp"].min()).dt.total_seconds() / (24 * 3600)

        if feat_cfg.get("add_degree_hours", True):
            base_t = 18.0
            g["degree_hours"] = (g["air_temperature"] - base_t).clip(lower=0).cumsum()

        if feat_cfg.get("add_resource_counters", True):
            g["cum_water_proxy"] = g.get("water_use_proxy", 0).fillna(0).cumsum()
            g["cum_energy_proxy"] = g.get("energy_proxy", 0).fillna(0).cumsum()

        all_frames.append(g)

    out = pd.concat(all_frames, ignore_index=True)
    out = out.sort_values(["team", "timestamp"]).reset_index(drop=True)
    return out


def fit_feature_store(train_df: pd.DataFrame, feature_cols: list[str]) -> FeatureStore:
    x = train_df[feature_cols].copy()
    medians = x.median(numeric_only=True).to_dict()
    dtypes = {c: str(x[c].dtype) for c in feature_cols}
    means = x.mean(numeric_only=True).to_dict()
    stds = x.std(numeric_only=True).replace(0, 1.0).to_dict()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(x)
    scaler.fit(x_imp)

    return FeatureStore(
        feature_names=feature_cols,
        medians={k: float(v) for k, v in medians.items()},
        dtypes=dtypes,
        means={k: float(v) for k, v in means.items()},
        stds={k: float(v) for k, v in stds.items()},
        imputer=imputer,
        scaler=scaler,
    )


def transform_with_store(df: pd.DataFrame, store: FeatureStore) -> np.ndarray:
    x = df[store.feature_names].copy()
    x_imp = store.imputer.transform(x)
    return store.scaler.transform(x_imp)
