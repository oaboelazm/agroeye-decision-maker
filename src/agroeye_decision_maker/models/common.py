from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


def make_regressor(model_type: str, max_depth: int, learning_rate: float, max_iter: int, random_state: int) -> Any:
    mt = model_type.lower()
    if mt == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if mt == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
            )
        except Exception:
            pass

    if mt == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        except Exception:
            pass

    if mt == "rf":
        return RandomForestRegressor(n_estimators=350, max_depth=max_depth, random_state=random_state, n_jobs=-1)

    return HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )
