from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from agroeye_decision_maker.control import MPCController, SafetyGuard
from agroeye_decision_maker.data import load_agc_dataset, split_time_blocks
from agroeye_decision_maker.features import FeatureStore, build_features, fit_feature_store, transform_with_store
from agroeye_decision_maker.models.imitation import ACTION_TARGETS, ImitationPolicy, decode_action_modes, train_imitation_policy
from agroeye_decision_maker.models.predictors import PredictorBundle, train_predictors
from agroeye_decision_maker.utils.config import save_json


@dataclass
class TrainArtifacts:
    metrics: dict[str, Any]
    report_md: str


def _ensure_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    logs_dir = Path(cfg["paths"]["logs_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir, reports_dir, logs_dir


def _dataset_hash(df: pd.DataFrame) -> str:
    subset = df[["team", "timestamp"]].astype(str).head(5000)
    payload = subset.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _safe_violation_rate(pred: np.ndarray, safety_cfg: dict[str, Any]) -> float:
    lo_t, hi_t = safety_cfg["actions"]["temperature_target_c"]
    lo_rh, hi_rh = safety_cfg["actions"]["humidity_target_rh_pct"]
    lo_fan, hi_fan = safety_cfg["actions"]["fan_speed_pct"]
    lo_vent, hi_vent = safety_cfg["actions"]["vent_open_pct"]
    lo_dur, hi_dur = safety_cfg["actions"]["irrigation_duration_s"]
    lo_flow, hi_flow = safety_cfg["actions"]["irrigation_flow_lph"]

    viol = (
        (pred[:, 0] < lo_t) | (pred[:, 0] > hi_t) |
        (pred[:, 5] < lo_rh) | (pred[:, 5] > hi_rh) |
        (pred[:, 3] < lo_fan) | (pred[:, 3] > hi_fan) |
        (pred[:, 4] < lo_vent) | (pred[:, 4] > hi_vent) |
        (pred[:, 1] < lo_dur) | (pred[:, 1] > hi_dur) |
        (pred[:, 2] < lo_flow) | (pred[:, 2] > hi_flow)
    )
    return float(np.mean(viol))


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    n = min(500, len(y_true))
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(y_true[:n, 0], label="true_temp_target")
    ax[0].plot(y_pred[:n, 0], label="pred_temp_target", alpha=0.8)
    ax[0].legend()
    ax[0].set_title("Imitation: Temperature Target")

    ax[1].plot(y_true[:n, 1], label="true_irr_dur")
    ax[1].plot(y_pred[:n, 1], label="pred_irr_dur", alpha=0.8)
    ax[1].legend()
    ax[1].set_title("Imitation: Irrigation Duration")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_feature_importance_surrogate(x: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    corr = x.corrwith(y).abs().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, ax=ax)
    ax.set_title("Top Feature Correlation (Surrogate Importance)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _to_action_dict(pred_row: np.ndarray, sensor_air_t: float, sensor_rh: float) -> dict[str, Any]:
    t_mode, h_mode, irr_on = decode_action_modes(pred_row, sensor_air_t, sensor_rh)
    return {
        "temperature": {"mode": t_mode, "target_c": float(pred_row[0])},
        "irrigation": {
            "on": bool(irr_on),
            "duration_s": int(max(0, round(pred_row[1]))),
            "flow_lph": float(pred_row[2]),
        },
        "ventilation": {
            "fan_speed_pct": float(pred_row[3]),
            "vent_open_pct": float(pred_row[4]),
        },
        "humidity": {"mode": h_mode, "target_rh_pct": float(pred_row[5])},
    }


def _evaluate_predictors(bundle: PredictorBundle, x: np.ndarray, y: dict[str, np.ndarray]) -> dict[str, Any]:
    preds = bundle.predict(x)
    metrics = {}
    for k, yt in y.items():
        yp = np.asarray(preds[k], dtype=float)
        yt_arr = np.asarray(yt, dtype=float)
        mask = np.isfinite(yt_arr) & np.isfinite(yp)
        if int(mask.sum()) < 2:
            metrics[f"{k}_mae"] = float("nan")
            metrics[f"{k}_r2"] = float("nan")
            continue
        metrics[f"{k}_mae"] = float(mean_absolute_error(yt_arr[mask], yp[mask]))
        metrics[f"{k}_r2"] = float(r2_score(yt_arr[mask], yp[mask]))
    return metrics


def _backtest(
    mode: str,
    val_df: pd.DataFrame,
    x_val: np.ndarray,
    imitation: ImitationPolicy | None,
    mpc: MPCController | None,
    safety_cfg: dict[str, Any],
) -> dict[str, Any]:
    safety = SafetyGuard(safety_cfg)

    cumulative_reward = 0.0
    total_water = 0.0
    total_energy = 0.0
    clamp_count = 0

    t_in_range = 0
    rh_in_range = 0

    for i in range(len(val_df)):
        row = val_df.iloc[i]
        if mode == "imitation" and imitation is not None:
            pred = imitation.predict_action_vector(x_val[i:i + 1])[0]
            action = _to_action_dict(pred, float(row["air_temperature"]), float(row["air_humidity"]))
        else:
            assert mpc is not None
            action = mpc.decide(x_val[i:i + 1], np.array([[row["air_temperature"], row["air_humidity"]]]), seed=42 + i)

        sensors = {
            "soil_humidity": float(row.get("soil_humidity", np.nan)),
        }
        safe_action, clamped, _ = safety.apply(action, sensors)
        clamp_count += int(clamped)

        quality = float(row.get("yield_quality_score", 0.0))
        water = float(safe_action["irrigation"]["duration_s"]) / 60.0
        energy = 0.02 * abs(float(safe_action["temperature"]["target_c"]) - float(row.get("air_temperature", 22.0))) + 0.01 * float(safe_action["ventilation"]["fan_speed_pct"])
        reward = quality - 0.25 * water - 0.3 * energy - (5.0 if clamped else 0.0)

        cumulative_reward += reward
        total_water += water
        total_energy += energy

        if 20 <= float(row.get("air_temperature", 22)) <= 26:
            t_in_range += 1
        if 60 <= float(row.get("air_humidity", 75)) <= 85:
            rh_in_range += 1

    n = max(1, len(val_df))
    return {
        "cumulative_pred_reward": float(cumulative_reward),
        "water_usage_proxy": float(total_water),
        "energy_proxy": float(total_energy),
        "safety_clamp_rate": float(clamp_count / n),
        "time_in_range_temp": float(t_in_range / n),
        "time_in_range_rh": float(rh_in_range / n),
    }


def _export_onnx_if_possible(model: Any, x_sample: np.ndarray, out_path: Path) -> str:
    try:
        from skl2onnx import to_onnx

        onx = to_onnx(model, x_sample.astype(np.float32), target_opset=15)
        out_path.write_bytes(onx.SerializeToString())
        return str(out_path)
    except Exception:
        return ""


def run_training(cfg: dict[str, Any], safety_cfg: dict[str, Any]) -> TrainArtifacts:
    artifacts_dir, reports_dir, logs_dir = _ensure_dirs(cfg)

    data = load_agc_dataset(cfg)
    featured = build_features(data, cfg)

    train_df, val_df, test_df = split_time_blocks(featured, cfg["training"]["val_fraction"], cfg["training"]["test_fraction"])
    max_train_rows = int(cfg["training"].get("max_train_rows", 50000))
    max_eval_rows = int(cfg["training"].get("max_eval_rows", 15000))
    seed = int(cfg["project"]["seed"])

    if len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=seed).sort_values(["team", "timestamp"]).reset_index(drop=True)
    if len(val_df) > max_eval_rows:
        val_df = val_df.sample(n=max_eval_rows, random_state=seed).sort_values(["team", "timestamp"]).reset_index(drop=True)
    if len(test_df) > max_eval_rows:
        test_df = test_df.sample(n=max_eval_rows, random_state=seed).sort_values(["team", "timestamp"]).reset_index(drop=True)

    if len(train_df) < int(cfg["training"].get("min_train_rows", 2000)):
        raise RuntimeError("Not enough rows after preprocessing for robust training.")

    target_cols = ACTION_TARGETS + ["yield_quality_score", "water_use_proxy", "energy_proxy", "air_temperature", "air_humidity", "soil_humidity", "co2_ppm"]

    drop_cols = {"timestamp", "team"} | set(target_cols)
    feature_cols = [c for c in train_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])]

    store = fit_feature_store(train_df, feature_cols)
    x_train = transform_with_store(train_df, store)
    x_val = transform_with_store(val_df, store)
    x_test = transform_with_store(test_df, store)

    y_train_actions_df = train_df[ACTION_TARGETS].copy()
    y_val_actions_df = val_df[ACTION_TARGETS].copy()
    y_test_actions_df = test_df[ACTION_TARGETS].copy()
    action_medians = y_train_actions_df.median(numeric_only=True)
    y_train_actions = y_train_actions_df.fillna(action_medians).values
    y_val_actions = y_val_actions_df.fillna(action_medians).values
    y_test_actions = y_test_actions_df.fillna(action_medians).values

    mode = str(cfg["training"].get("controller_mode", "imitation"))

    imitation: ImitationPolicy | None = None
    predictors: PredictorBundle | None = None
    mpc: MPCController | None = None

    metrics: dict[str, Any] = {
        "mode": mode,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_cols)),
    }

    if mode == "imitation":
        imitation = train_imitation_policy(x_train, y_train_actions, cfg, seed)
        pred_val = imitation.predict_action_vector(x_val)
        pred_test = imitation.predict_action_vector(x_test)

        metrics["imitation_val_mae"] = float(mean_absolute_error(y_val_actions, pred_val))
        metrics["imitation_test_mae"] = float(mean_absolute_error(y_test_actions, pred_test))
        metrics["imitation_val_mape"] = float(mean_absolute_percentage_error(np.maximum(np.abs(y_val_actions), 1e-3), np.abs(pred_val)))

        metrics["safety_violation_rate_pre_clamp"] = _safe_violation_rate(pred_val, safety_cfg)

        guard = SafetyGuard(safety_cfg)
        post_viol = 0
        for i in range(len(pred_val)):
            row = val_df.iloc[i]
            action = _to_action_dict(pred_val[i], float(row["air_temperature"]), float(row["air_humidity"]))
            _, clamped, _ = guard.apply(action, {"soil_humidity": float(row.get("soil_humidity", np.nan))})
            post_viol += int(clamped)
        metrics["safety_violation_rate_post_clamp"] = float(post_viol / max(1, len(pred_val)))

        _plot_predictions(y_val_actions, pred_val, reports_dir / "imitation_predictions.png")
        _plot_feature_importance_surrogate(val_df[feature_cols], val_df["target_temperature_c"], reports_dir / "feature_importance.png")

        backtest = _backtest(mode, val_df, x_val, imitation, None, safety_cfg)
        metrics.update({f"backtest_{k}": v for k, v in backtest.items()})

        imitation.save(str(artifacts_dir / "controller_imitation.joblib"))
        onnx_path = _export_onnx_if_possible(imitation.model.estimators_[0], x_val[:1], artifacts_dir / "controller_imitation.onnx")

    else:
        y_train_pred = {t: train_df[t].values for t in cfg["predictors"]["targets"] if t in train_df.columns}
        y_val_pred = {t: val_df[t].values for t in y_train_pred.keys()}

        predictors = train_predictors(x_train, y_train_pred, cfg, seed)
        pred_metrics = _evaluate_predictors(predictors, x_val, y_val_pred)
        metrics.update({f"predictor_{k}": v for k, v in pred_metrics.items()})

        mpc = MPCController(predictors=predictors, cfg=cfg)
        backtest = _backtest(mode, val_df, x_val, None, mpc, safety_cfg)
        metrics.update({f"backtest_{k}": v for k, v in backtest.items()})

        predictors.save(str(artifacts_dir / "predictors.joblib"))
        onnx_path = ""

    # Composite score.
    acc_term = 1.0 / (1.0 + float(metrics.get("imitation_val_mae", np.mean(list(v for k, v in metrics.items() if k.endswith("_mae"))) if any(k.endswith("_mae") for k in metrics) else 1.0)))
    quality_term = float(metrics.get("backtest_cumulative_pred_reward", 0.0)) / max(1.0, len(val_df))
    cost_term = float(metrics.get("backtest_water_usage_proxy", 0.0) + metrics.get("backtest_energy_proxy", 0.0)) / max(1.0, len(val_df))
    metrics["composite_score"] = float(0.5 * acc_term + 0.35 * quality_term - 0.15 * cost_term)

    store.save(artifacts_dir / "feature_store.joblib")

    model_info = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "controller_mode": mode,
        "data_hash": _dataset_hash(data),
        "feature_count": len(feature_cols),
        "feature_schema": feature_cols,
        "metrics": metrics,
        "onnx_path": onnx_path,
    }
    save_json(artifacts_dir / "model_info.json", model_info)

    metrics_df = pd.DataFrame([{"timestamp_utc": datetime.now(timezone.utc).isoformat(), **metrics}])
    metrics_csv = Path(cfg["logging"]["csv_metrics_path"])
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    if metrics_csv.exists():
        existing = pd.read_csv(metrics_csv)
        all_cols = sorted(set(existing.columns) | set(metrics_df.columns))
        existing = existing.reindex(columns=all_cols)
        metrics_df = metrics_df.reindex(columns=all_cols)
        merged = pd.concat([existing, metrics_df], ignore_index=True)
        merged.to_csv(metrics_csv, index=False)
    else:
        metrics_df.to_csv(metrics_csv, index=False)

    report_md = _build_report_md(train_df, val_df, test_df, metrics, model_info)
    (reports_dir / "report.md").write_text(report_md, encoding="utf-8")

    return TrainArtifacts(metrics=metrics, report_md=report_md)


def _build_report_md(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, metrics: dict[str, Any], model_info: dict[str, Any]) -> str:
    lines = [
        "# AgroEye Decision Maker Report",
        "",
        "## Data Splits",
        f"- Train rows: {len(train_df)}",
        f"- Validation rows: {len(val_df)}",
        f"- Test rows: {len(test_df)}",
        "",
        "## Key Metrics",
    ]
    for k, v in sorted(metrics.items()):
        lines.append(f"- {k}: {v:.6f}" if isinstance(v, (float, int)) else f"- {k}: {v}")

    lines += [
        "",
        "## Artifacts",
        "- Plots: `reports/imitation_predictions.png`, `reports/feature_importance.png` (for imitation mode)",
        "- Model info: `artifacts/model_info.json`",
        "",
        "## Safety Clamp Statistics",
        f"- Clamp rate (backtest): {metrics.get('backtest_safety_clamp_rate', float('nan'))}",
        "",
        "## Notes",
        "- Composite score blends action accuracy, predicted reward quality, and resource proxies.",
        "- Reward is proxy-based due offline dataset constraints and no direct actuator-ground-truth for all live outputs.",
        "",
        "## Model Info",
        f"- model_version: {model_info['model_version']}",
        f"- data_hash: {model_info['data_hash']}",
    ]
    return "\n".join(lines)
