from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from .common import make_regressor

ACTION_TARGETS = [
    "target_temperature_c",
    "irrigation_duration_s",
    "irrigation_flow_lph",
    "fan_speed_pct",
    "vent_open_pct",
    "target_humidity_rh_pct",
]


@dataclass
class ImitationPolicy:
    model: Any
    action_targets: list[str]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ImitationPolicy":
        return joblib.load(path)

    def predict_action_vector(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


def train_imitation_policy(x_train: np.ndarray, y_train: np.ndarray, cfg: dict[str, Any], seed: int) -> ImitationPolicy:
    model_cfg = cfg["imitation"]
    base = make_regressor(
        model_type=str(model_cfg.get("model_type", "hist_gbrt")),
        max_depth=int(model_cfg.get("max_depth", 8)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        max_iter=int(model_cfg.get("max_iter", 350)),
        random_state=seed,
    )
    model = MultiOutputRegressor(base)
    model.fit(x_train, y_train)
    return ImitationPolicy(model=model, action_targets=ACTION_TARGETS)


def decode_action_modes(pred: np.ndarray, current_air_t: float, current_rh: float) -> tuple[str, str, bool]:
    target_t = float(pred[0])
    target_rh = float(pred[5])

    if target_t > current_air_t + 0.5:
        t_mode = "heat"
    elif target_t < current_air_t - 0.5:
        t_mode = "cool"
    else:
        t_mode = "hold"

    if target_rh > current_rh + 2:
        h_mode = "humidify"
    elif target_rh < current_rh - 2:
        h_mode = "dehumidify"
    else:
        h_mode = "hold"

    irrigation_on = float(pred[1]) > 3.0
    return t_mode, h_mode, irrigation_on
