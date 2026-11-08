"""
OverWatch AI - synthetic burn-in / ESS dataset generator (Part 1 of the project).

Implements the dataset prompt:
  * 5 batches x 20 units = 100 components, 8 readings each (0..168 h)
  * 90% NORMAL         : datasheet-typical values + Gaussian fixture noise, tiny drift
  * 5%  BATCH_ANOMALY  : 3x-5x away from the batch mean but still below the hard limit
  * 5%  LATENT_DEFECT  : starts normal at hour 0, then accelerates (exponential /
                         accelerating-linear) toward hour 168
  * all numbers rounded to 2 decimals, schema depends on the component category

Usage:
    python data_generator.py --component 5            # electrolytic capacitors
    python data_generator.py --component LED --seed 7
    python data_generator.py --all --out sample_data
"""
from __future__ import annotations

import argparse
import os
from typing import Union

import numpy as np
import pandas as pd

from config import (BATCHES, BATCH_ANOMALY_FRACTION, BURN_IN_HOURS, COMPONENTS,
                    HORIZON_HOUR, LATENT_DEFECT_FRACTION, UNITS_PER_BATCH,
                    ComponentSpec, resolve_component)


def _latent_shape(kind: str, f: float) -> float:
    """Degradation progress 0..1 at normalised time f = hour/168 (accelerating)."""
    if kind == "exponential":
        return float((np.exp(3.0 * f) - 1.0) / (np.exp(3.0) - 1.0))
    return float(0.25 * f + 0.75 * f ** 2)  # accelerating-linear


def generate_dataset(component: Union[int, str, ComponentSpec], seed: int = 42,
                     units_per_batch: int = UNITS_PER_BATCH) -> pd.DataFrame:
    spec = resolve_component(component)
    rng = np.random.default_rng([seed, spec.id])
    n_each_bad = max(1, int(round(BATCH_ANOMALY_FRACTION * units_per_batch)))
    n_each_lat = max(1, int(round(LATENT_DEFECT_FRACTION * units_per_batch)))
    limit = spec.pri_limit
    rows = []

    for b_idx, batch in enumerate(BATCHES):
        letter = chr(ord("A") + b_idx)
        # batch-level process offsets (each batch has its own mean)
        pri_b = spec.pri_mean * (1 + rng.normal(0, 0.05))
        sec_b = spec.sec_mean * (1 + rng.normal(0, 0.02))
        temp_b = spec.temp_c + rng.normal(0, 1.5)

        picks = rng.choice(units_per_batch, n_each_bad + n_each_lat, replace=False)
        roles = {int(u): "BATCH_ANOMALY" for u in picks[:n_each_bad]}
        roles.update({int(u): "LATENT_DEFECT" for u in picks[n_each_bad:]})

        for u in range(units_per_batch):
            role = roles.get(u, "NORMAL")
            comp_id = f"{spec.code}-{letter}{u + 1:02d}"

            pri_u = pri_b * (1 + rng.normal(0, spec.pri_spread))
            sec_u = sec_b * (1 + rng.normal(0, spec.sec_spread))
            temp_u = temp_b + rng.normal(0, 1.0)

            # natural (tiny) drift shared by every unit
            nat_pri = rng.normal(0.004, 0.003)
            nat_sec = rng.normal(0.002, 0.002)

            if role == "BATCH_ANOMALY":
                # 3x-5x away from the batch mean, but kept under the datasheet limit
                pri_u = min(pri_b * rng.uniform(3.0, 5.0), limit * 0.95)
                sec_u = sec_u * (1 + spec.sec_dir * rng.uniform(3, 5) * spec.sec_spread)
                temp_u += rng.uniform(0.5, 2.0)

            if role == "LATENT_DEFECT":
                kind = rng.choice(["exponential", "accelerating_linear"])
                mult = rng.uniform(2.2, 3.5)       # primary growth at hour 168
                sec_chg = rng.uniform(0.12, 0.30)  # secondary change at hour 168
                temp_rise = rng.uniform(4.0, 10.0)

            for h in BURN_IN_HOURS:
                f = h / HORIZON_HOUR
                pri = pri_u * (1 + nat_pri * f)
                sec = sec_u * (1 + spec.sec_dir * nat_sec * f)
                temp = temp_u
                if role == "LATENT_DEFECT":
                    s = _latent_shape(kind, f)
                    pri *= 1 + (mult - 1) * s
                    sec *= 1 + spec.sec_dir * sec_chg * s
                    temp += temp_rise * s
                pri *= 1 + rng.normal(0, spec.pri_noise)
                sec *= 1 + rng.normal(0, spec.sec_noise)
                temp += rng.normal(0, 1.0)
                rows.append((comp_id, batch, h, pri, sec, temp, role))

    df = pd.DataFrame(rows, columns=spec.columns)
    num = df.columns[2:6]
    df[num[1:]] = df[num[1:]].clip(lower=0.01)
    df = df.round(2)
    return df.sort_values(["batch_id", "component_id", "burn_in_hour"]).reset_index(drop=True)


def to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, float_format="%.2f")


def main() -> None:
    ap = argparse.ArgumentParser(description="OverWatch AI synthetic dataset generator")
    ap.add_argument("--component", help="id 1-16, code (ECAP) or name")
    ap.add_argument("--all", action="store_true", help="generate all 16 components")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sample_data", help="output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    targets = COMPONENTS if args.all else [resolve_component(args.component or 5)]
    for spec in targets:
        df = generate_dataset(spec, seed=args.seed)
        path = os.path.join(args.out, f"{spec.id:02d}_{spec.code.lower()}_burnin.csv")
        with open(path, "w", newline="") as fh:
            fh.write(to_csv_text(df))
        counts = df.drop_duplicates("component_id")["status_ground_truth"].value_counts().to_dict()
        print(f"[{spec.id:>2}] {spec.name:<46} -> {path}  rows={len(df)}  units={counts}")


if __name__ == "__main__":
    main()
