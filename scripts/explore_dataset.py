from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agroeye_decision_maker.data import load_agc_dataset
from agroeye_decision_maker.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/base.yaml")
    out_dir = Path(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_agc_dataset(cfg)

    profile = []
    for team, grp in df.groupby("team"):
        span_days = (grp["timestamp"].max() - grp["timestamp"].min()).total_seconds() / (3600 * 24)
        profile.append(
            {
                "team": team,
                "rows": len(grp),
                "start": grp["timestamp"].min(),
                "end": grp["timestamp"].max(),
                "span_days": span_days,
                "avg_missing": float(grp.isna().mean().mean()),
            }
        )

    prof_df = pd.DataFrame(profile)
    prof_df.to_csv(out_dir / "dataset_profile.csv", index=False)

    key_cols = ["air_temperature", "air_humidity", "soil_humidity", "co2_ppm", "yield_quality_score"]
    long = df[["team"] + key_cols].melt(id_vars=["team"], var_name="feature", value_name="value")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long, x="feature", y="value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "eda_feature_distributions.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    for team, grp in df.groupby("team"):
        sampled = grp.iloc[::20]
        plt.plot(sampled["timestamp"], sampled["air_temperature"], label=team, alpha=0.8)
    plt.legend()
    plt.title("Air Temperature Timeline (sampled)")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_temperature_timeline.png")
    plt.close()

    lines = [
        "# Dataset Exploration",
        "",
        "## Summary by Team",
        prof_df.to_markdown(index=False),
        "",
        "## Key Observations",
        "- Greenhouse and substrate sensor streams are dense (~10-minute cadence).",
        "- Production/quality/lab tables are sparse and aligned by timestamp via resampling and forward-fill.",
        "- N/P/K are not directly available as soil sensors in AGC tables; inferred proxies use irrigation/lab chemistry columns.",
    ]
    (out_dir / "dataset_exploration.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
