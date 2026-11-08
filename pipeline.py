"""
OverWatch AI - dual-stage ML pipeline (Part 2 of the project).

Stage 1  Pre-processing & batch-aware grouping (batch x hour statistics)
Stage 2  Module A  - batch-relative anomaly detection  (Isolation Forest + robust z / ratio)
Stage 3  Module B  - time-series drift prediction      (Ridge regression, 168 h projection)
Stage 4  Risk engine + Explainable-AI reason codes     (GREEN / YELLOW / RED)

Works on any CSV that follows the OverWatch schema (the primary/secondary column
names are auto-detected from the `primary_parameter*` / `secondary_parameter*` prefix).
"""
from __future__ import annotations

import io
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ID_COLS = ["component_id", "batch_id", "burn_in_hour"]
TEMP_COL = "thermal_stress_temp_c"
LABELS = {
    "GREEN": "LOW RISK (Green)",
    "YELLOW": "WATCH / EXTENDED TESTING (Yellow)",
    "RED": "HIGH RISK / REJECTION (Red)",
}


# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    # Module A
    contamination: Union[float, str] = 0.05
    n_estimators: int = 300
    random_state: int = 42
    ratio_watch: float = 1.30     # persistent deviation vs batch baseline -> yellow starts
    ratio_reject: float = 2.50    # ... -> full-scale (red)
    sec_z_watch: float = 2.5
    sec_z_reject: float = 6.0
    if_frac_watch: float = 0.25   # share of a unit's readings isolated by the forest
    if_frac_reject: float = 0.75
    # Module B
    ridge_alpha: float = 1e-3
    horizon_hour: float = 168.0
    fit_until_hour: Optional[float] = None   # e.g. 96 = early screening, project to 168
    growth_watch: float = 1.20    # projected / starting value
    growth_reject: float = 1.80
    sec_drift_watch: float = 0.06
    sec_drift_reject: float = 0.20
    # Risk engine
    yellow_at: float = 25.0
    red_at: float = 55.0
    # Optional static datasheet limit on the primary parameter (used in XAI text)
    primary_limit: Optional[float] = None


@dataclass
class PipelineResult:
    records: pd.DataFrame        # per reading: anomaly_score, anomaly_flag (1 / -1), z-scores
    batch_stats: pd.DataFrame    # per batch x hour mean / std
    components: pd.DataFrame     # per component: drift, risk, XAI
    summary: dict
    series: dict
    meta: dict

    def to_dict(self, include_records: bool = False, include_series: bool = True) -> dict:
        out = {
            "meta": self.meta,
            "summary": self.summary,
            "results": _records_json(self.components),
            "batch_stats": _records_json(self.batch_stats),
        }
        if include_series:
            out["series"] = self.series
        if include_records:
            out["records"] = _records_json(self.records)
        return out


def _records_json(df: pd.DataFrame) -> list:
    clean = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records"))


def _ramp(x, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear in between (NaN -> 0)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _dev(r: float) -> float:
    """Symmetric multiplicative deviation: 4x above and 0.25x below both give 4."""
    if r is None or not np.isfinite(r) or r <= 0:
        return 1.0
    return float(r if r >= 1 else 1.0 / r)


def _pretty(col: str) -> str:
    for p in ("primary_parameter_", "secondary_parameter_"):
        col = col.replace(p, "")
    return col.replace("_", " ")


# --------------------------------------------------------------------------- #
class OverWatchPipeline:
    FEATURES = ["pri_log_ratio", "pri_z", "sec_z", "temp_z"]

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.pri_col = self.sec_col = None

    # ---------------------------- STAGE 1 --------------------------------- #
    def load(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, str) and "\n" in data:
            df = pd.read_csv(io.StringIO(data))
        else:
            df = pd.read_csv(data)
        df.columns = [c.strip() for c in df.columns]
        return df

    def preprocess(self, df: pd.DataFrame):
        missing = [c for c in ID_COLS + [TEMP_COL] if c not in df.columns]
        pri = [c for c in df.columns if c.startswith("primary_parameter")]
        sec = [c for c in df.columns if c.startswith("secondary_parameter")]
        if missing or not pri or not sec:
            raise ValueError(
                "CSV does not match the OverWatch schema. Missing: "
                f"{missing + ([] if pri else ['primary_parameter_*']) + ([] if sec else ['secondary_parameter_*'])}")
        self.pri_col, self.sec_col = pri[0], sec[0]
        num = [self.pri_col, self.sec_col, TEMP_COL, "burn_in_hour"]
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["component_id", "batch_id", "burn_in_hour"])
        df = (df.sort_values(["component_id", "burn_in_hour"])
                .drop_duplicates(["component_id", "burn_in_hour"], keep="last")
                .reset_index(drop=True))

        # missing values: per-component interpolation -> batch/hour median -> global median
        n_missing = int(df[num[:3]].isna().sum().sum())
        for c in num[:3]:
            df[c] = df.groupby("component_id")[c].transform(
                lambda s: s.interpolate(limit_direction="both"))
            df[c] = df[c].fillna(df.groupby(["batch_id", "burn_in_hour"])[c].transform("median"))
            df[c] = df[c].fillna(df[c].median())
        df.attrs["imputed_cells"] = n_missing

        # batch x hour statistics (mean / std) for the telemetry metrics
        stats = (df.groupby(["batch_id", "burn_in_hour"])[[self.pri_col, self.sec_col, TEMP_COL]]
                   .agg(["mean", "std", "median"]))
        stats.columns = [f"{a}_{b}" for a, b in stats.columns]
        stats = stats.reset_index()

        # per-record, batch-relative features (robust median / MAD so outliers cannot hide themselves)
        g = df.groupby(["batch_id", "burn_in_hour"])
        for col, tag in [(self.pri_col, "pri"), (self.sec_col, "sec"), (TEMP_COL, "temp")]:
            med = g[col].transform("median")
            mad = g[col].transform(lambda x: (x - x.median()).abs().median()) * 1.4826
            sd = np.maximum(mad, np.maximum(np.abs(med) * 0.02, 1e-6))
            df[f"{tag}_batch_mean"] = g[col].transform("mean")
            df[f"{tag}_batch_std"] = g[col].transform("std")
            df[f"{tag}_batch_median"] = med
            df[f"{tag}_z"] = ((df[col] - med) / sd).clip(-50, 50)
        df["pri_ratio"] = df[self.pri_col] / df["pri_batch_median"].replace(0, np.nan)
        df["pri_ratio"] = df["pri_ratio"].fillna(1.0)
        df["pri_log_ratio"] = np.log(df["pri_ratio"].clip(lower=1e-3))
        return df, stats

    # ---------------------------- STAGE 2 --------------------------------- #
    def module_a_records(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.FEATURES].to_numpy(dtype=float)
        if len(df) < 20:
            df["anomaly_score"], df["anomaly_flag"] = 0.5, 1
            return df
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(n_estimators=self.cfg.n_estimators,
                                     contamination=self.cfg.contamination,
                                     random_state=self.cfg.random_state).fit(Xs)
        # anomaly_score in (0,1): ~0.5 typical, -> 1.0 clearly anomalous (higher = worse)
        df["anomaly_score"] = -self.model.score_samples(Xs)
        df["anomaly_flag"] = self.model.predict(Xs)          # 1 normal, -1 outlier
        return df

    def module_a_components(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("component_id").agg(
            batch_id=("batch_id", "first"),
            n_records=("anomaly_flag", "size"),
            outlier_records=("anomaly_flag", lambda s: int((s == -1).sum())),
            anomaly_score=("anomaly_score", "mean"),
            anomaly_score_max=("anomaly_score", "max"),
            persistent_ratio=("pri_ratio", "median"),
            peak_ratio=("pri_ratio", "max"),
            persistent_sec_z=("sec_z", "median"),
            temp_z_max=("temp_z", "max"),
            primary_max=(self.pri_col, "max"),
        ).reset_index()
        c = self.cfg
        out_frac = agg["outlier_records"] / agg["n_records"]
        agg["outlier_fraction"] = out_frac
        agg["module_a_score"] = [
            max(_ramp(_dev(r), c.ratio_watch, c.ratio_reject),
                0.8 * _ramp(abs(z), c.sec_z_watch, c.sec_z_reject),
                _ramp(f, c.if_frac_watch, c.if_frac_reject))
            for r, z, f in zip(agg["persistent_ratio"], agg["persistent_sec_z"], out_frac)]
        agg["anomaly_flag"] = np.where(agg["module_a_score"] >= 0.5, -1, 1)
        return agg

    # ---------------------------- STAGE 3 --------------------------------- #
    def _fit_ridge(self, t, y):
        m = Ridge(alpha=self.cfg.ridge_alpha).fit(t.reshape(-1, 1), y)
        return m

    def module_b(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        rows = []
        for cid, g in df.groupby("component_id"):
            g = g.sort_values("burn_in_hour")
            if c.fit_until_hour is not None:
                g = g[g["burn_in_hour"] <= c.fit_until_hour]
            h = g["burn_in_hour"].to_numpy(float)
            y = g[self.pri_col].to_numpy(float)
            s = g[self.sec_col].to_numpy(float)
            last = df[df["component_id"] == cid].sort_values("burn_in_hour").iloc[-1]
            rec = {"component_id": cid,
                   "primary_value": float(last[self.pri_col]),
                   "secondary_value": float(last[self.sec_col]),
                   "temp_c": float(last[TEMP_COL])}
            if len(h) < 3:
                rec.update(drift_slope_per_hr=np.nan, drift_pct_per_100h=np.nan,
                           primary_baseline=np.nan, projected_168h=np.nan,
                           projected_growth_ratio=np.nan, secondary_change_pct=np.nan,
                           accelerating=False, module_b_score=0.0)
                rows.append(rec)
                continue
            t = h / c.horizon_hour
            base = float(np.median(y[:2]))
            base = base if base > 0 else float(np.mean(y)) or 1.0

            lin = self._fit_ridge(t, y)                      # linear drift velocity
            slope_hr = float(lin.coef_[0]) / c.horizon_hour
            proj_lin = float(lin.predict([[1.0]])[0])
            exp_m = self._fit_ridge(t, np.log(np.clip(y, 1e-9, None)))   # exponential trend
            proj_exp = float(min(np.exp(exp_m.predict([[1.0]])[0]), 20 * base))
            proj = max(proj_lin, proj_exp)                   # conservative projection

            sec_base = float(np.median(s[:2])) or 1.0
            sec_lin = self._fit_ridge(t, s)
            sec_proj = float(sec_lin.predict([[1.0]])[0])
            sec_change = (sec_proj - sec_base) / abs(sec_base)

            accel = False
            if len(h) >= 6:
                half = len(h) // 2
                early = np.polyfit(h[:half], y[:half], 1)[0]
                late = np.polyfit(h[-half:], y[-half:], 1)[0]
                accel = bool(late > 0 and late > 1.5 * max(early, 1e-12))

            growth = proj / base
            b_pri = _ramp(_dev(growth), c.growth_watch, c.growth_reject)
            b_sec = 0.7 * _ramp(abs(sec_change), c.sec_drift_watch, c.sec_drift_reject)
            rec.update(
                drift_slope_per_hr=slope_hr,
                drift_pct_per_100h=slope_hr * 100.0 / base * 100.0,
                primary_baseline=base,
                projected_168h=proj,
                projected_growth_ratio=growth,
                secondary_change_pct=sec_change * 100.0,
                accelerating=accel,
                module_b_score=max(b_pri, b_sec),
            )
            rows.append(rec)
        return pd.DataFrame(rows)

    # ---------------------------- STAGE 4 --------------------------------- #
    def risk_engine(self, comp: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        a, b = comp["module_a_score"], comp["module_b_score"]
        comp["risk_score"] = 100.0 * (0.7 * np.maximum(a, b) + 0.3 * np.minimum(a, b))
        comp["risk_category"] = np.select(
            [comp["risk_score"] >= c.red_at, comp["risk_score"] >= c.yellow_at],
            ["RED", "YELLOW"], default="GREEN")
        comp["risk_label"] = comp["risk_category"].map(LABELS)
        comp["triggered_by"] = [
            "A+B" if x >= 0.3 and y >= 0.3 else "A" if x >= 0.3 else "B" if y >= 0.3 else "-"
            for x, y in zip(a, b)]
        codes, texts = [], []
        pri_name, sec_name = _pretty(self.pri_col), _pretty(self.sec_col)
        for _, r in comp.iterrows():
            cd, tx = self._explain(r, pri_name, sec_name)
            codes.append("|".join(cd))
            prefix = {"RED": "REJECT - ", "YELLOW": "WATCH - ", "GREEN": "OK - "}[r["risk_category"]]
            texts.append(prefix + "; ".join(tx))
        comp["reason_codes"], comp["xai_reason"] = codes, texts
        return comp

    def _explain(self, r, pri, sec):
        c = self.cfg
        codes, txt = [], []
        ratio = float(r["persistent_ratio"])
        drifting = _dev(r["projected_growth_ratio"]) >= c.growth_watch
        # a moderate median ratio on a drifting unit is the drift itself, not a batch outlier
        if _dev(ratio) >= c.ratio_watch and not (drifting and _dev(ratio) < 2.0):
            codes.append("BATCH_OUTLIER_PRIMARY")
            way = "above" if ratio >= 1 else "below"
            t = (f"primary parameter ({pri}) sits {_dev(ratio):.1f}x {way} the {r['batch_id']} "
                 f"batch baseline across the burn-in")
            if c.primary_limit and r["primary_max"] < c.primary_limit:
                t += f" despite passing the static datasheet limit ({c.primary_limit:g})"
            txt.append(t)
        if abs(float(r["persistent_sec_z"])) >= c.sec_z_watch:
            codes.append("BATCH_OUTLIER_SECONDARY")
            txt.append(f"secondary parameter ({sec}) deviates {abs(r['persistent_sec_z']):.1f} robust "
                       f"sigma from its batch peers")
        if r["outlier_fraction"] >= c.if_frac_watch:
            codes.append("ISOLATION_FOREST_OUTLIER")
            txt.append(f"Isolation Forest isolated {int(r['outlier_records'])}/{int(r['n_records'])} "
                       f"readings (mean anomaly score {r['anomaly_score']:.2f})")
        if _dev(r["projected_growth_ratio"]) >= c.growth_watch:
            codes.append("ACCELERATING_DRIFT" if r["accelerating"] else "PROGRESSIVE_DRIFT")
            txt.append(f"{pri} drifting {r['drift_pct_per_100h']:+.1f}%/100h "
                       f"(slope {r['drift_slope_per_hr']:+.4f}/h), projected {r['projected_168h']:.2f} at "
                       f"{c.horizon_hour:g}h = {r['projected_growth_ratio']:.1f}x its starting value"
                       + (" with an accelerating trend" if r["accelerating"] else ""))
        if abs(r["secondary_change_pct"]) >= c.sec_drift_watch * 100:
            codes.append("SECONDARY_DRIFT")
            txt.append(f"{sec} projected to change {r['secondary_change_pct']:+.1f}% by {c.horizon_hour:g}h")
        if r["temp_z_max"] >= 4:
            codes.append("THERMAL_ANOMALY")
            txt.append(f"thermal stress temperature peaks {r['temp_z_max']:.1f} sigma above batch norm")
        if not codes:
            codes.append("STABLE")
            txt.append(f"within batch norms (primary {ratio:.2f}x batch baseline, drift "
                       f"{r['drift_pct_per_100h']:+.1f}%/100h, {int(r['outlier_records'])}/"
                       f"{int(r['n_records'])} readings flagged)")
        txt[0] = txt[0][0].upper() + txt[0][1:]
        return codes, txt

    # ---------------------------- RUN ------------------------------------- #
    def run(self, data: Union[str, pd.DataFrame], component_name: Optional[str] = None) -> PipelineResult:
        df = self.load(data)
        truth = None
        if "status_ground_truth" in df.columns:
            truth = df.drop_duplicates("component_id").set_index("component_id")["status_ground_truth"]
        df, stats = self.preprocess(df)
        records = self.module_a_records(df)
        comp = self.module_a_components(records).merge(self.module_b(records), on="component_id")
        comp = self.risk_engine(comp)
        if truth is not None:
            comp["status_ground_truth"] = comp["component_id"].map(truth)

        cols = ["component_id", "batch_id", "primary_value", "secondary_value", "temp_c",
                "anomaly_score", "anomaly_flag", "outlier_records", "n_records",
                "persistent_ratio", "peak_ratio", "drift_slope_per_hr", "drift_pct_per_100h",
                "primary_baseline", "projected_168h", "projected_growth_ratio",
                "secondary_change_pct", "module_a_score", "module_b_score", "risk_score",
                "risk_category", "risk_label", "triggered_by", "reason_codes", "xai_reason"]
        if truth is not None:
            cols.append("status_ground_truth")
        comp = comp[cols].sort_values("risk_score", ascending=False).reset_index(drop=True)
        num = comp.select_dtypes("number").columns
        comp[num] = comp[num].round(4)

        counts = comp["risk_category"].value_counts()
        by_batch = (comp.groupby(["batch_id", "risk_category"]).size().unstack(fill_value=0)
                    .reindex(columns=["GREEN", "YELLOW", "RED"], fill_value=0))
        summary = {
            "total": int(len(comp)),
            "green": int(counts.get("GREEN", 0)),
            "yellow": int(counts.get("YELLOW", 0)),
            "red": int(counts.get("RED", 0)),
            "by_batch": {b: {k.lower(): int(v) for k, v in row.items()} for b, row in by_batch.iterrows()},
        }
        series = {}
        for cid, g in records.sort_values("burn_in_hour").groupby("component_id"):
            series[cid] = [{"hour": int(h), "primary": float(p), "secondary": float(s), "temp": float(t)}
                           for h, p, s, t in zip(g["burn_in_hour"], g[self.pri_col],
                                                 g[self.sec_col], g[TEMP_COL])]
        meta = {
            "component": component_name,
            "primary_column": self.pri_col, "secondary_column": self.sec_col,
            "n_rows": int(len(records)), "n_batches": int(records["batch_id"].nunique()),
            "imputed_cells": int(df.attrs.get("imputed_cells", 0)),
            "config": asdict(self.cfg),
            "has_ground_truth": truth is not None,
        }
        out_rec = records[["component_id", "batch_id", "burn_in_hour", self.pri_col, self.sec_col,
                           TEMP_COL, "pri_ratio", "pri_z", "sec_z", "temp_z",
                           "anomaly_score", "anomaly_flag"]].copy()
        out_rec[out_rec.select_dtypes("number").columns] = out_rec.select_dtypes("number").round(4)
        return PipelineResult(out_rec, stats.round(4), comp, summary, series, meta)

    # ---------------------------- misc ------------------------------------ #
    def save(self, path: str) -> None:
        joblib.dump({"config": self.cfg, "model": self.model, "scaler": self.scaler,
                     "features": self.FEATURES}, path)

    @staticmethod
    def load_model(path: str) -> dict:
        return joblib.load(path)


def evaluate(components: pd.DataFrame) -> dict:
    """Compare risk categories with `status_ground_truth` (only for synthetic data)."""
    if "status_ground_truth" not in components.columns:
        raise ValueError("No status_ground_truth column to evaluate against")
    gt = components["status_ground_truth"]
    flagged = components["risk_category"] != "GREEN"
    out = {}
    for kind in ["BATCH_ANOMALY", "LATENT_DEFECT"]:
        m = gt == kind
        out[f"{kind.lower()}_recall_flagged"] = float(flagged[m].mean()) if m.any() else None
        out[f"{kind.lower()}_recall_red"] = float((components["risk_category"][m] == "RED").mean()) if m.any() else None
    n = gt == "NORMAL"
    out["normal_false_flag_rate"] = float(flagged[n].mean()) if n.any() else None
    out["normal_false_red_rate"] = float((components["risk_category"][n] == "RED").mean()) if n.any() else None
    out["confusion"] = pd.crosstab(gt, components["risk_category"]).to_dict()
    return out


def analyze_csv(path_or_text: str, config: Optional[PipelineConfig] = None) -> PipelineResult:
    warnings.filterwarnings("ignore")
    return OverWatchPipeline(config).run(path_or_text)
