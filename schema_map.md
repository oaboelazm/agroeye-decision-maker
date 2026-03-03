# Schema Mapping: AGC 2019 -> AgroEye Live I/O

## Dataset Sources Used
- `GreenhouseClimate.csv` (10-min climate + setpoints)
- `GrodanSens.csv` (substrate/soil analog signals)
- `Resources.csv` (daily resource usage)
- `Production.csv` (yield proxies)
- `TomQuality.csv` (quality proxies)
- `LabAnalysis.csv` (chemistry used for N/P/K proxies)
- `CropParameters.csv` (crop stage support)
- `Weather/Weather.csv` (external environment)

## Time Handling
- AGC `%time` / `%Time` is parsed as Excel serial day and converted to UTC timestamp.
- All tables are resampled to `configs/base.yaml -> data.resample_rule` (default `10min`).
- Sparse tables (e.g., production/lab) are aligned by timestamp and forward/back filled within each team.

## Live Sensor Schema Mapping
| Live Field | Primary AGC Column(s) | Transform / Assumption |
|---|---|---|
| `air_temperature` | `Tair` | direct |
| `air_humidity` | `Rhair` | direct |
| `light_lux` | `Tot_PAR` | `lux ~= PAR * 54` (configurable in code) |
| `co2_ppm` | `CO2air` | direct |
| `soil_temperature` | `t_slab1`, `t_slab2` | mean of slabs |
| `soil_humidity` | `WC_slab1`, `WC_slab2` | mean of slabs |
| `soil_ec` | `EC_slab1`, `EC_slab2` | mean of slabs |
| `soil_ph` | `pH_drain_PC`, fallback `drain_PH`, `irr_PH` | proxy for root-zone pH |
| `soil_n` | `irr_NO3`/`drain_NO3` | proxy (not direct N sensor) |
| `soil_p` | `irr_PO4`/`drain_PO4` | proxy (not direct P sensor) |
| `soil_k` | `irr_K`/`drain_K` | proxy (not direct K sensor) |

## Action Label Mapping (Imitation Target)
| Control Output | AGC Source | Assumption |
|---|---|---|
| `temperature.target_c` | `t_heat_sp` fallback `t_vent_sp` | direct setpoint proxy |
| `temperature.mode` | derived vs current `air_temperature` | `heat/cool/hold` by difference threshold |
| `irrigation.duration_s` | `water_sup_intervals_sp_min * 60` fallback from `Irr` | duration proxy |
| `irrigation.flow_lph` | `water_sup * 60` | flow proxy |
| `irrigation.on` | duration > 0 | boolean |
| `ventilation.fan_speed_pct` | `Ventwind * 100` | percent proxy |
| `ventilation.vent_open_pct` | `VentLee * 100` | percent proxy |
| `humidity.target_rh_pct` | derived from `HumDef` as `clip(80 - 2.5*HumDef, 50, 90)` | no direct RH setpoint in source |
| `humidity.mode` | derived vs current `air_humidity` | `humidify/dehumidify/hold` |

## Ambiguities Logged
- AGC lacks direct soil N/P/K hardware channels; chemistry proxies are used.
- Some teams have malformed extra unnamed columns in `Resources.csv`; dropped automatically.
- `Reference/TomQuality.csv` has malformed tab/comma header; robust parser fallback is implemented.
- Some actuators in the requested live schema do not have exact historical ground truth; model trains on closest available setpoint/actuation proxies and SafetyGuard enforces safe output.
