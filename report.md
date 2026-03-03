# AgroEye Decision Maker Report

## 1) Data Splits and Setup
- Dataset: local AGC 2019 folder (`./Dataset`)
- Preprocessing: 10-minute resampling, IQR clipping, EWMA smoothing, optional feature fallback for missing live fields
- Downsampling for runtime practicality: every 3rd sample (`data.downsample_step=3`)

### Imitation Run (A)
- Train/Val/Test rows: 30000 / 8000 / 8000
- Feature count: 383

### MPC Run (B)
- Train/Val/Test rows: 15000 / 1000 / 1000
- Feature count: 383

## 2) Key Metrics

### A) Imitation / Direct Policy
- Validation MAE (action vector): 4089.5222
- Test MAE (action vector): 7544.0332
- Validation MAPE: 2205793.9820
- Safety Violation Rate (pre-clamp): 0.9999
- Safety Violation Rate (post-clamp): 0.5196
- Backtest cumulative predicted reward: -22336.2225
- Backtest water usage proxy: 5.0000
- Backtest energy proxy: 5438.5838
- Backtest clamp rate: 0.5196
- Time-in-range temperature: 0.4711
- Time-in-range RH: 0.3828
- Composite score: -1.0792

### B) Predictors + MPC
- Predictor MAE / R2:
  - air_temperature: 0.0029 / 1.0000
  - air_humidity: 0.0446 / 1.0000
  - soil_humidity: 0.0424 / 0.9999
  - co2_ppm: 0.2196 / 1.0000
  - yield_quality_score: 0.0099 / 0.5151
  - water_use_proxy: 0.4721 / -3.6699
  - energy_proxy: 1.7939 / 0.9557
- Backtest cumulative predicted reward: -5147.3986
- Backtest water usage proxy: 5.0000
- Backtest energy proxy: 507.2378
- Backtest clamp rate: 0.9990
- Time-in-range temperature: 0.5050
- Time-in-range RH: 0.3780
- Composite score: -1.5133

## 3) Interpretation
- Imitation has high raw action error because historical actuator proxies are noisy and only partially aligned with the requested live schema.
- MPC predictor fit is strong for climate channels, moderate for the yield-quality proxy, and weak for the water-use proxy (negative R2).
- SafetyGuard reduces unsafe commands, but clamp rates remain high, indicating a policy/action-space mismatch under strict greenhouse constraints.

## 4) Backtest and Safety
- Safety constraints enforced after every decision:
  - Temperature/humidity target bounds
  - Vent/fan bounds and ramp limits
  - Irrigation max per hour/day + minimum off-time + emergency dryness pulse
- All API responses include `safety.clamped` and `safety.violations`.

## 5) Generated Graphs
- Dataset distributions: `reports/eda_feature_distributions.png`
- Temperature timeline: `reports/eda_temperature_timeline.png`
- Imitation prediction traces: `reports/imitation_predictions.png`
- Surrogate feature importance: `reports/feature_importance.png`

## 6) Artifacts
- `artifacts/feature_store.joblib`
- `artifacts/controller_imitation.joblib`
- `artifacts/predictors.joblib`
- `artifacts/model_info.json`
- `artifacts/logs/metrics.csv`

## 7) Notes and Next Improvements
- Improve action targets by reconstructing actuator semantics from challenge documentation and controller internals.
- Add calibrated confidence model and better OOD gating for MPC proposals.
- Tune reward shaping and constraints to reduce clamp rate while preserving yield/resource balance.
