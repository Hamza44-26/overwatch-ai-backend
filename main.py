"""
OverWatch AI - command line entry point.

    python main.py demo [--component 5]          generate data -> run ML -> print report
    python main.py generate --component LED      write a synthetic CSV
    python main.py generate --all                write all 16 CSVs into sample_data/
    python main.py analyze --input file.csv [--out report.csv] [--fit-until 96]
    python main.py serve                         start the REST API on :5000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

from config import resolve_component
from data_generator import generate_dataset, to_csv_text
from pipeline import OverWatchPipeline, PipelineConfig, evaluate

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.max_columns", 30)

ICON = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


def print_report(result, show_all: bool = False) -> None:
    s = result.summary
    print("\n=== OverWatch AI screening summary ===")
    print(f"Total {s['total']}   🟢 {s['green']}   🟡 {s['yellow']}   🔴 {s['red']}")
    comp = result.components.copy()
    comp["risk"] = comp["risk_category"].map(ICON) + " " + comp["risk_category"]
    cols = ["component_id", "batch_id", "anomaly_score", "anomaly_flag", "drift_slope_per_hr",
            "projected_168h", "risk_score", "risk"]
    view = comp if show_all else comp[comp["risk_category"] != "GREEN"]
    print("\n--- Flagged components ---" if not show_all else "\n--- All components ---")
    print(view[cols].round(3).to_string(index=False))
    print("\n--- XAI reason codes ---")
    for _, r in view.iterrows():
        print(f"{ICON[r['risk_category']]} {r['component_id']:<10} [{r['reason_codes']}]\n     {r['xai_reason']}")
    if result.meta["has_ground_truth"]:
        ev = evaluate(result.components)
        print("\n--- Validation vs status_ground_truth (synthetic data only) ---")
        for k, v in ev.items():
            if k != "confusion":
                print(f"  {k:<32} {v:.2%}" if v is not None else f"  {k:<32} n/a")
        print(pd.crosstab(result.components["status_ground_truth"], result.components["risk_category"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="OverWatch AI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo")
    d.add_argument("--component", default="5")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--all", action="store_true", help="print every component, not only flagged ones")

    g = sub.add_parser("generate")
    g.add_argument("--component", default="5")
    g.add_argument("--all", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="sample_data")

    a = sub.add_parser("analyze")
    a.add_argument("--input", required=True)
    a.add_argument("--out", default="screening_report.csv")
    a.add_argument("--fit-until", type=float, default=None,
                   help="fit drift only on readings up to this hour, then project to 168 h")
    a.add_argument("--component", default=None, help="optional, adds datasheet limit to XAI text")

    sub.add_parser("serve")
    args = ap.parse_args()

    if args.cmd == "demo":
        spec = resolve_component(args.component)
        print(f"Component: {spec.name}  (category: {spec.category_name})")
        df = generate_dataset(spec, args.seed)
        cfg = PipelineConfig(primary_limit=spec.pri_limit)
        res = OverWatchPipeline(cfg).run(df, component_name=spec.name)
        print_report(res, args.all)

    elif args.cmd == "generate":
        os.makedirs(args.out, exist_ok=True)
        from config import COMPONENTS
        for spec in (COMPONENTS if args.all else [resolve_component(args.component)]):
            path = os.path.join(args.out, f"{spec.id:02d}_{spec.code.lower()}_burnin.csv")
            with open(path, "w", newline="") as fh:
                fh.write(to_csv_text(generate_dataset(spec, args.seed)))
            print("wrote", path)

    elif args.cmd == "analyze":
        cfg = PipelineConfig(fit_until_hour=args.fit_until)
        if args.component:
            cfg.primary_limit = resolve_component(args.component).pri_limit
        pipe = OverWatchPipeline(cfg)
        res = pipe.run(args.input)
        res.components.to_csv(args.out, index=False)
        os.makedirs("models", exist_ok=True)
        pipe.save(os.path.join("models", "overwatch_model.joblib"))
        print_report(res)
        print(f"\nReport written to {args.out}")

    elif args.cmd == "serve":
        from api import app
        app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    sys.exit(main())
