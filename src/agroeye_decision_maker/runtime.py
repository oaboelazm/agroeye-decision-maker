from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from agroeye_decision_maker.control import MPCController, SafetyGuard
from agroeye_decision_maker.features import FeatureStore
from agroeye_decision_maker.models.imitation import ImitationPolicy, decode_action_modes
from agroeye_decision_maker.models.predictors import PredictorBundle


class DecisionRuntime:
    def __init__(self, cfg: dict[str, Any], safety_cfg: dict[str, Any]):
        self.cfg = cfg
        self.safety_guard = SafetyGuard(safety_cfg)
        artifacts = Path(cfg["paths"]["artifacts_dir"])

        self.feature_store: FeatureStore = joblib.load(artifacts / "feature_store.joblib")
        model_info_path = artifacts / "model_info.json"
        self.model_info = {}
        if model_info_path.exists():
            import json

            self.model_info = json.loads(model_info_path.read_text(encoding="utf-8"))

        self.mode = str(self.model_info.get("controller_mode", cfg["training"].get("controller_mode", "imitation")))

        self.imitation = None
        self.predictors = None
        self.mpc = None

        if self.mode == "imitation" and (artifacts / "controller_imitation.joblib").exists():
            self.imitation = ImitationPolicy.load(str(artifacts / "controller_imitation.joblib"))
        if (artifacts / "predictors.joblib").exists():
            self.predictors = PredictorBundle.load(str(artifacts / "predictors.joblib"))
            if self.mode == "mpc":
                self.mpc = MPCController(self.predictors, cfg)

        self.history = deque(maxlen=100)
        self.last_sensor = {}

    def _row_from_sensors(self, sensors: dict[str, Any], timestamp_utc: str) -> pd.DataFrame:
        row = {c: self.feature_store.medians.get(c, 0.0) for c in self.feature_store.feature_names}

        # Direct live schema values.
        for key, val in sensors.items():
            if val is None:
                continue
            if key in row:
                row[key] = float(val)

        # Derived aliases used in training features.
        row["air_temperature"] = float(sensors.get("air_temperature", row.get("air_temperature", 22.0)))
        row["air_humidity"] = float(sensors.get("air_humidity", row.get("air_humidity", 75.0)))
        row["soil_humidity"] = float(sensors.get("soil_humidity", row.get("soil_humidity", 35.0)))
        row["co2_ppm"] = float(sensors.get("co2_ppm", row.get("co2_ppm", 700.0)))
        row["soil_temperature"] = float(sensors.get("soil_temperature", row.get("soil_temperature", 20.0)))
        row["soil_ec"] = float(sensors.get("soil_ec", row.get("soil_ec", 2.0)))
        row["soil_ph"] = float(sensors.get("soil_ph", row.get("soil_ph", 6.0)))

        ts = pd.Timestamp(timestamp_utc)
        row["hour_sin"] = np.sin(2 * np.pi * (ts.hour + ts.minute / 60.0) / 24.0)
        row["hour_cos"] = np.cos(2 * np.pi * (ts.hour + ts.minute / 60.0) / 24.0)

        # Missing handling with carry-forward.
        for k in list(row.keys()):
            if pd.isna(row[k]):
                if k in self.last_sensor:
                    row[k] = self.last_sensor[k]
                else:
                    row[k] = self.feature_store.medians.get(k, 0.0)

        return pd.DataFrame([row])

    def _ood_score(self, row: pd.DataFrame) -> float:
        scores = []
        for col in row.columns:
            mu = self.feature_store.means.get(col)
            sd = self.feature_store.stds.get(col, 1.0)
            if mu is None:
                continue
            sd = sd if sd > 1e-6 else 1.0
            z = abs((float(row.iloc[0][col]) - mu) / sd)
            scores.append(z)
        if not scores:
            return 0.0
        return float(np.nanpercentile(scores, 90))

    def _detect_anomaly(self, sensors: dict[str, Any]) -> bool:
        if len(self.history) < 2:
            return False
        prev = self.history[-1]
        jump_sigma = float(self.cfg["inference"]["anomaly"].get("jump_sigma", 5.0))

        jump_count = 0
        for k in ["air_temperature", "air_humidity", "soil_humidity", "co2_ppm"]:
            if k in sensors and k in prev and sensors[k] is not None and prev[k] is not None:
                if abs(float(sensors[k]) - float(prev[k])) > jump_sigma * 5:
                    jump_count += 1

        flatline_steps = int(self.cfg["inference"]["anomaly"].get("flatline_steps", 6))
        is_flat = False
        if len(self.history) >= flatline_steps:
            same = 0
            for p in list(self.history)[-flatline_steps:]:
                if all(abs(float(p.get(k, 0)) - float(prev.get(k, 0))) < 1e-9 for k in ["air_temperature", "air_humidity", "soil_humidity", "co2_ppm"]):
                    same += 1
            is_flat = same == flatline_steps

        return jump_count >= 2 or is_flat

    def _fallback_reasoned(self, ts: str, sensors: dict[str, Any], reason: str) -> dict[str, Any]:
        action = self.safety_guard.fallback_action()
        dry = sensors.get("soil_humidity") is not None and float(sensors["soil_humidity"]) < float(self.safety_guard.cfg["irrigation"]["dryness_trigger_pct"])
        if dry:
            action["irrigation"] = {
                "on": True,
                "duration_s": int(self.safety_guard.cfg["irrigation"]["emergency_pulse_s"]),
                "flow_lph": float(self.safety_guard.cfg["irrigation"]["emergency_flow_lph"]),
            }
        safe_action, clamped, violations = self.safety_guard.apply(action, sensors)
        return {
            "timestamp_utc": ts,
            "actions": safe_action,
            "rationale": f"Fallback safe hold mode triggered: {reason}." + (" Dry soil pulse applied." if dry else ""),
            "safety": {"clamped": clamped, "violations": violations},
        }

    def decide(self, timestamp_utc: str | None, sensors: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

        # Save history for anomaly detection and carry-forward.
        self.history.append(dict(sensors))

        anomaly = self._detect_anomaly(sensors)
        row = self._row_from_sensors(sensors, ts)
        x = self.feature_store.scaler.transform(self.feature_store.imputer.transform(row[self.feature_store.feature_names]))

        ood_threshold = float(self.cfg["inference"]["confidence"].get("ood_zscore_threshold", 4.0))
        ood = self._ood_score(row) > ood_threshold

        if anomaly or ood:
            reason = "sensor anomaly" if anomaly else "low confidence / out-of-distribution"
            return self._fallback_reasoned(ts, sensors, reason)

        if self.mode == "imitation" and self.imitation is not None:
            pred = self.imitation.predict_action_vector(x)[0]
            t_mode, h_mode, irr_on = decode_action_modes(
                pred,
                float(sensors.get("air_temperature", row.iloc[0].get("air_temperature", 22.0))),
                float(sensors.get("air_humidity", row.iloc[0].get("air_humidity", 75.0))),
            )
            action = {
                "temperature": {"mode": t_mode, "target_c": float(pred[0])},
                "irrigation": {"on": bool(irr_on), "duration_s": int(max(0, round(pred[1]))), "flow_lph": float(pred[2])},
                "ventilation": {"fan_speed_pct": float(pred[3]), "vent_open_pct": float(pred[4])},
                "humidity": {"mode": h_mode, "target_rh_pct": float(pred[5])},
            }
        elif self.mpc is not None:
            raw = np.array([[float(row.iloc[0].get("air_temperature", 22.0)), float(row.iloc[0].get("air_humidity", 75.0))]])
            action = self.mpc.decide(x, raw)
        else:
            return self._fallback_reasoned(ts, sensors, "model artifact missing")

        safe_action, clamped, violations = self.safety_guard.apply(action, sensors)

        rationale_bits = []
        if sensors.get("soil_humidity") is not None:
            rationale_bits.append(f"soil_humidity={float(sensors['soil_humidity']):.1f}")
        if sensors.get("air_temperature") is not None:
            rationale_bits.append(f"air_temperature={float(sensors['air_temperature']):.1f}")
        if sensors.get("air_humidity") is not None:
            rationale_bits.append(f"air_humidity={float(sensors['air_humidity']):.1f}")
        if sensors.get("co2_ppm") is not None:
            rationale_bits.append(f"co2_ppm={float(sensors['co2_ppm']):.0f}")

        rationale = "Decision from {} controller using key readings: {}.".format(self.mode, ", ".join(rationale_bits[:4]))

        # Update last sensor cache after decision path.
        for k, v in row.iloc[0].to_dict().items():
            if isinstance(v, (int, float, np.number)) and not pd.isna(v):
                self.last_sensor[k] = float(v)

        return {
            "timestamp_utc": ts,
            "actions": safe_action,
            "rationale": rationale,
            "safety": {"clamped": bool(clamped), "violations": violations},
        }
