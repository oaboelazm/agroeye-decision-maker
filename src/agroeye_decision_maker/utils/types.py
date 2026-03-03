from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LIVE_SENSOR_FIELDS = [
    "soil_n",
    "soil_p",
    "soil_k",
    "soil_temperature",
    "soil_humidity",
    "soil_ph",
    "soil_ec",
    "air_temperature",
    "air_humidity",
    "light_lux",
    "co2_ppm",
]


@dataclass
class ModelBundle:
    controller: Any
    feature_store: Any
    predictors: dict[str, Any]
    metadata: dict[str, Any]
