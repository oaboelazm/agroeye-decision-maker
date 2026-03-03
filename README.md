# AgroEye Decision Maker

Production-ready ML controller for tomato greenhouse decision support using the local AGC 2019 dataset.

## 1. Setup
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

## 2. Configure Dataset Path
Edit `configs/base.yaml`:
```yaml
paths:
  data_root: "./Dataset"
```
Point this to your local AGC folder if different.

## 3. Explore Dataset (Deep Dive)
```bash
python scripts/explore_dataset.py
```
Outputs:
- `reports/dataset_profile.csv`
- `reports/dataset_exploration.md`
- `reports/eda_feature_distributions.png`
- `reports/eda_temperature_timeline.png`

## 4. Train and Evaluate
Select the controller mode in `configs/base.yaml`:
- `training.controller_mode: imitation`
- `training.controller_mode: mpc`

Run training:
```bash
python main.py --config configs/base.yaml --safety configs/safety.yaml train
```
Outputs:
- `artifacts/feature_store.joblib`
- `artifacts/controller_imitation.joblib` (imitation mode)
- `artifacts/predictors.joblib` (MPC mode)
- `artifacts/model_info.json`
- `reports/report.md`
- plots in `reports/`

## 5. Run API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /health`
- `POST /control/decide`

Sample request:
```bash
curl -X POST http://127.0.0.1:8000/control/decide \
  -H "Content-Type: application/json" \
  -d @examples/request.json
```

## 6. Run a One-Off CLI Decision
```bash
python main.py decide --input examples/request.json
```

## 7. Safety Limits
Edit `configs/safety.yaml` to adjust:
- Sensor safe ranges
- Irrigation max duration/hour/day
- Min off-time constraints
- Ramp-rate limits for targets and ventilation

## 8. Reproducibility and Logging
- Random seed set via `project.seed` in `configs/base.yaml`
- Experiment metrics appended to `artifacts/logs/metrics.csv`
- Model metadata and schema stored in `artifacts/model_info.json`

## 9. ONNX Export
- The imitation model attempts ONNX export if `skl2onnx` is installed (`pip install -e .[onnx]`).

## 10. Files
- `schema_map.md`: dataset-to-live I/O mapping and assumptions
- `reports/report.md`: metrics + backtest summary after training
- `examples/request.json`: quick input payload
