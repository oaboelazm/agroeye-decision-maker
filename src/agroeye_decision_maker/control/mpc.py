from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agroeye_decision_maker.models.predictors import PredictorBundle


@dataclass
class MPCController:
    predictors: PredictorBundle
    cfg: dict[str, Any]

    def _sample_action(self, rng: np.random.Generator) -> dict[str, Any]:
        bounds = self.cfg["mpc"]["action_bounds"]
        target_c = rng.uniform(*bounds["target_c"])
        duration_s = int(rng.uniform(*bounds["irrigation_duration_s"]))
        flow_lph = float(rng.uniform(*bounds["irrigation_flow_lph"]))
        fan = float(rng.uniform(*bounds["fan_speed_pct"]))
        vent = float(rng.uniform(*bounds["vent_open_pct"]))
        rh = float(rng.uniform(*bounds["humidity_target_rh_pct"]))

        return {
            "temperature": {"mode": "hold", "target_c": target_c},
            "irrigation": {"on": duration_s > 0, "duration_s": duration_s, "flow_lph": flow_lph},
            "ventilation": {"fan_speed_pct": fan, "vent_open_pct": vent},
            "humidity": {"mode": "hold", "target_rh_pct": rh},
        }

    def _action_to_vec(self, action: dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                action["temperature"]["target_c"],
                action["irrigation"]["duration_s"],
                action["irrigation"]["flow_lph"],
                action["ventilation"]["fan_speed_pct"],
                action["ventilation"]["vent_open_pct"],
                action["humidity"]["target_rh_pct"],
            ],
            dtype=float,
        )

    def _reward(self, pred: dict[str, float], action: dict[str, Any]) -> float:
        lam_w = float(self.cfg["mpc"]["lambda_water"])
        lam_e = float(self.cfg["mpc"]["lambda_energy"])
        lam_s = float(self.cfg["mpc"]["lambda_safety"])

        quality = float(pred.get("yield_quality_score", 0.0))
        water = float(pred.get("water_use_proxy", action["irrigation"]["duration_s"] / 60.0))
        energy = float(pred.get("energy_proxy", 0.0))

        safety_penalty = 0.0
        if action["temperature"]["target_c"] < 16 or action["temperature"]["target_c"] > 30:
            safety_penalty += 1
        if action["humidity"]["target_rh_pct"] < 50 or action["humidity"]["target_rh_pct"] > 90:
            safety_penalty += 1

        return quality - lam_w * water - lam_e * energy - lam_s * safety_penalty

    def decide(self, x_now_scaled: np.ndarray, x_now_raw: np.ndarray, seed: int = 42) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        n = int(self.cfg["mpc"]["n_candidates"])
        horizon = int(self.cfg["mpc"]["horizon_steps"])

        best_score = -1e18
        best_action = None

        for _ in range(n):
            first_action = self._sample_action(rng)
            cumulative = 0.0
            state = x_now_scaled.copy()

            for h in range(horizon):
                action = first_action if h == 0 else self._sample_action(rng)
                # Append action features into the first six dimensions as perturbation proxy.
                action_vec = self._action_to_vec(action)
                state_step = state.copy()
                state_step[0, : min(6, state_step.shape[1])] = state_step[0, : min(6, state_step.shape[1])] * 0.7 + action_vec[: min(6, state_step.shape[1])] * 0.3

                preds = {k: float(v[0]) for k, v in self.predictors.predict(state_step).items()}
                cumulative += self._reward(preds, action)

            if cumulative > best_score:
                best_score = cumulative
                best_action = first_action

        if best_action is None:
            best_action = self._sample_action(rng)

        air_t = float(x_now_raw[0, 0]) if x_now_raw.shape[1] > 0 else 22.0
        air_rh = float(x_now_raw[0, 1]) if x_now_raw.shape[1] > 1 else 75.0
        if best_action["temperature"]["target_c"] > air_t + 0.5:
            best_action["temperature"]["mode"] = "heat"
        elif best_action["temperature"]["target_c"] < air_t - 0.5:
            best_action["temperature"]["mode"] = "cool"
        else:
            best_action["temperature"]["mode"] = "hold"

        if best_action["humidity"]["target_rh_pct"] > air_rh + 2:
            best_action["humidity"]["mode"] = "humidify"
        elif best_action["humidity"]["target_rh_pct"] < air_rh - 2:
            best_action["humidity"]["mode"] = "dehumidify"
        else:
            best_action["humidity"]["mode"] = "hold"

        return best_action
