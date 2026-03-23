from __future__ import annotations

from collections import deque
from copy import deepcopy
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

    def _merge_overrides(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        if not overrides:
            return self.cfg

        merged = deepcopy(self.cfg)

        def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _deep_update(dst[k], v)
                else:
                    dst[k] = v

        _deep_update(merged, overrides)
        return merged

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
        if len(self.history) < 1:
            return False
        prev = self.history[-1]
        jump_sigma = float(self.cfg["inference"]["anomaly"].get("jump_sigma", 5.0))

        jump_count = 0
        for k in ["air_temperature", "air_humidity", "soil_humidity", "co2_ppm"]:
            cur_v = sensors.get(k)
            prev_v = prev.get(k)
            if cur_v is not None and prev_v is not None:
                if abs(float(cur_v) - float(prev_v)) > jump_sigma * 5:
                    jump_count += 1

        flatline_steps = int(self.cfg["inference"]["anomaly"].get("flatline_steps", 6))
        is_flat = False
        if len(self.history) >= flatline_steps - 1:
            same = 0
            recent = list(self.history)[-(flatline_steps - 1):]
            for p in recent:
                all_same = True
                for k in ["air_temperature", "air_humidity", "soil_humidity", "co2_ppm"]:
                    cur_v = sensors.get(k)
                    hist_v = p.get(k)
                    if cur_v is None or hist_v is None or abs(float(hist_v) - float(cur_v)) >= 1e-9:
                        all_same = False
                        break
                if all_same:
                    same += 1
            is_flat = same == (flatline_steps - 1)

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
        quality_pct = self._decision_quality_pct(
            clamped=bool(clamped),
            violations=violations,
            used_fallback=True,
            ood=True,
            anomaly=("anomaly" in reason.lower()),
        )
        return {
            "timestamp_utc": ts,
            "actions": safe_action,
            "rationale": f"Fallback safe hold mode triggered: {reason}." + (" Dry soil pulse applied." if dry else ""),
            "safety": {"clamped": clamped, "violations": violations},
            "quality_score_pct": quality_pct,
        }

    def _decision_quality_pct(
        self,
        clamped: bool,
        violations: list[str],
        used_fallback: bool,
        ood: bool,
        anomaly: bool,
    ) -> float:
        # Practical management-facing score (0-100) for runtime decision reliability.
        score = 92.0
        if used_fallback:
            score -= 35.0
        if ood:
            score -= 15.0
        if anomaly:
            score -= 20.0
        if clamped:
            score -= 10.0
        score -= min(20.0, 3.0 * float(len(violations)))
        return float(max(0.0, min(100.0, score)))

    def decide(self, timestamp_utc: str | None, sensors: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
        effective_cfg = self._merge_overrides(overrides)

        anomaly = self._detect_anomaly(sensors)
        row = self._row_from_sensors(sensors, ts)
        x = self.feature_store.scaler.transform(self.feature_store.imputer.transform(row[self.feature_store.feature_names]))

        ood_threshold = float(effective_cfg["inference"]["confidence"].get("ood_zscore_threshold", 4.0))
        ood = self._ood_score(row) > ood_threshold

        if anomaly or ood:
            reason = "sensor anomaly" if anomaly else "low confidence / out-of-distribution"
            self.history.append(dict(sensors))
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
            mpc_runtime = MPCController(self.predictors, effective_cfg)
            action = mpc_runtime.decide(x, raw)
        else:
            self.history.append(dict(sensors))
            return self._fallback_reasoned(ts, sensors, "model artifact missing")

        safe_action, clamped, violations = self.safety_guard.apply(action, sensors, ts)

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
        self.history.append(dict(sensors))

        quality_pct = self._decision_quality_pct(
            clamped=bool(clamped),
            violations=violations,
            used_fallback=False,
            ood=bool(ood),
            anomaly=bool(anomaly),
        )

        return {
            "timestamp_utc": ts,
            "actions": safe_action,
            "rationale": rationale,
            "safety": {"clamped": bool(clamped), "violations": violations},
            "quality_score_pct": quality_pct,
        }
