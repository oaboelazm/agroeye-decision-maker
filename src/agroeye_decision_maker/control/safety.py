from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyState:
    last_temp_target: float | None = None
    last_rh_target: float | None = None
    last_fan: float | None = None
    last_vent: float | None = None
    steps_since_irrigation: int = 999
    irrigation_hour_accum: float = 0.0
    irrigation_day_accum: float = 0.0


@dataclass
class SafetyGuard:
    cfg: dict[str, Any]
    state: SafetyState = field(default_factory=SafetyState)

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, v)))

    def fallback_action(self) -> dict[str, Any]:
        fb = self.cfg["fallback"]
        return {
            "temperature": {"mode": "hold", "target_c": float(fb["hold_temp_c"])},
            "irrigation": {"on": False, "duration_s": 0, "flow_lph": float(self.cfg["actions"]["irrigation_flow_lph"][0])},
            "ventilation": {
                "fan_speed_pct": float(fb["min_fan_pct"]),
                "vent_open_pct": float(fb["min_vent_pct"]),
            },
            "humidity": {"mode": "hold", "target_rh_pct": float(fb["hold_rh_pct"])},
        }

    def apply(self, action: dict[str, Any], sensors: dict[str, float]) -> tuple[dict[str, Any], bool, list[str]]:
        out = {k: dict(v) if isinstance(v, dict) else v for k, v in action.items()}
        violations: list[str] = []

        a_cfg = self.cfg["actions"]
        r_cfg = self.cfg["ramp_limits"]
        i_cfg = self.cfg["irrigation"]

        # Clamp targets.
        t = self._clamp(float(out["temperature"]["target_c"]), *a_cfg["temperature_target_c"])
        rh = self._clamp(float(out["humidity"]["target_rh_pct"]), *a_cfg["humidity_target_rh_pct"])
        fan = self._clamp(float(out["ventilation"]["fan_speed_pct"]), *a_cfg["fan_speed_pct"])
        vent = self._clamp(float(out["ventilation"]["vent_open_pct"]), *a_cfg["vent_open_pct"])
        dur = int(self._clamp(float(out["irrigation"]["duration_s"]), *a_cfg["irrigation_duration_s"]))
        flow = self._clamp(float(out["irrigation"]["flow_lph"]), *a_cfg["irrigation_flow_lph"])

        # Ramp constraints.
        if self.state.last_temp_target is not None and abs(t - self.state.last_temp_target) > r_cfg["temperature_target_delta_per_step"]:
            t = self.state.last_temp_target + r_cfg["temperature_target_delta_per_step"] * (1 if t > self.state.last_temp_target else -1)
            violations.append("temperature_target_ramp_limited")

        if self.state.last_rh_target is not None and abs(rh - self.state.last_rh_target) > r_cfg["humidity_target_delta_per_step"]:
            rh = self.state.last_rh_target + r_cfg["humidity_target_delta_per_step"] * (1 if rh > self.state.last_rh_target else -1)
            violations.append("humidity_target_ramp_limited")

        if self.state.last_fan is not None and abs(fan - self.state.last_fan) > r_cfg["fan_delta_per_step"]:
            fan = self.state.last_fan + r_cfg["fan_delta_per_step"] * (1 if fan > self.state.last_fan else -1)
            violations.append("fan_ramp_limited")

        if self.state.last_vent is not None and abs(vent - self.state.last_vent) > r_cfg["vent_delta_per_step"]:
            vent = self.state.last_vent + r_cfg["vent_delta_per_step"] * (1 if vent > self.state.last_vent else -1)
            violations.append("vent_ramp_limited")

        # Irrigation limits.
        irrigation_on = bool(out["irrigation"].get("on", False)) and dur > 0
        if irrigation_on:
            if self.state.steps_since_irrigation < int(i_cfg["min_off_time_steps"]):
                irrigation_on = False
                dur = 0
                violations.append("irrigation_min_off_time")
            if self.state.irrigation_hour_accum + dur > float(i_cfg["max_duration_per_hour_s"]):
                dur = max(0, int(float(i_cfg["max_duration_per_hour_s"]) - self.state.irrigation_hour_accum))
                if dur == 0:
                    irrigation_on = False
                violations.append("irrigation_hour_limit")
            if self.state.irrigation_day_accum + dur > float(i_cfg["max_duration_per_day_s"]):
                dur = max(0, int(float(i_cfg["max_duration_per_day_s"]) - self.state.irrigation_day_accum))
                if dur == 0:
                    irrigation_on = False
                violations.append("irrigation_day_limit")

        # Sensor-based emergency clamp.
        soil_h = sensors.get("soil_humidity")
        if soil_h is not None and soil_h < float(i_cfg["dryness_trigger_pct"]) and not irrigation_on:
            irrigation_on = True
            dur = max(dur, int(i_cfg["emergency_pulse_s"]))
            flow = max(flow, float(i_cfg["emergency_flow_lph"]))
            violations.append("dryness_emergency_pulse")

        out["temperature"]["target_c"] = float(t)
        out["humidity"]["target_rh_pct"] = float(rh)
        out["ventilation"]["fan_speed_pct"] = float(fan)
        out["ventilation"]["vent_open_pct"] = float(vent)
        out["irrigation"]["on"] = bool(irrigation_on)
        out["irrigation"]["duration_s"] = int(dur)
        out["irrigation"]["flow_lph"] = float(flow)

        # Update state counters.
        self.state.last_temp_target = float(t)
        self.state.last_rh_target = float(rh)
        self.state.last_fan = float(fan)
        self.state.last_vent = float(vent)
        if irrigation_on and dur > 0:
            self.state.steps_since_irrigation = 0
            self.state.irrigation_hour_accum += dur
            self.state.irrigation_day_accum += dur
        else:
            self.state.steps_since_irrigation += 1

        return out, len(violations) > 0, violations
