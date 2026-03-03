from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np

from .common import make_regressor


@dataclass
class PredictorBundle:
    models: dict[str, Any]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PredictorBundle":
        return joblib.load(path)

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {k: v.predict(x) for k, v in self.models.items()}


def train_predictors(x_train: np.ndarray, y_train: dict[str, np.ndarray], cfg: dict[str, Any], seed: int) -> PredictorBundle:
    p_cfg = cfg["predictors"]
    models = {}
    for target, y in y_train.items():
        reg = make_regressor(
            model_type=str(p_cfg.get("model_type", "hist_gbrt")),
            max_depth=int(p_cfg.get("max_depth", 8)),
            learning_rate=float(p_cfg.get("learning_rate", 0.05)),
            max_iter=int(p_cfg.get("max_iter", 300)),
            random_state=seed,
        )
        reg.fit(x_train, y)
        models[target] = reg
    return PredictorBundle(models=models)
