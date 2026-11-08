"""
OverWatch AI - REST API (connects the ML pipeline to the Bolt-generated UI).

Run:  python api.py          ->  http://localhost:5000

Endpoints
  GET  /api/health
  GET  /api/components                      taxonomy for the Category -> Component dropdowns
  GET  /api/generate?component=5&seed=42    synthetic CSV (add &format=json for rows)
  POST /api/analyze                         multipart 'file' (.csv)  OR  JSON {"csv": "..."}
         optional: component, fit_until_hour, contamination, include_records
         add ?format=csv to download the results table as CSV
  POST /api/analyze/sample                  JSON {"component": 5, "seed": 42} - generate + analyze
"""
from __future__ import annotations

import io
import warnings

import pandas as pd
from flask import Flask, Response, jsonify, request

from config import resolve_component, taxonomy
from data_generator import generate_dataset, to_csv_text
from pipeline import OverWatchPipeline, PipelineConfig, evaluate

warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # 25 MB uploads


# ----------------------------- CORS (for the Bolt dev server) --------------- #
@app.before_request
def _preflight():
    if request.method == "OPTIONS":
        return Response(status=204)


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.errorhandler(ValueError)
def _bad_request(e):
    return jsonify({"error": str(e)}), 400


@app.errorhandler(Exception)
def _server_error(e):
    code = getattr(e, "code", 500)
    return jsonify({"error": str(e)}), code if isinstance(code, int) else 500


# ----------------------------- helpers -------------------------------------- #
def _config_from(payload: dict, spec=None) -> PipelineConfig:
    cfg = PipelineConfig()
    if payload.get("fit_until_hour") not in (None, "", "null"):
        cfg.fit_until_hour = float(payload["fit_until_hour"])
    if payload.get("contamination") not in (None, "", "null"):
        cfg.contamination = float(payload["contamination"])
    if spec is not None:
        cfg.primary_limit = spec.pri_limit
    return cfg


def _run(csv_text: str, payload: dict):
    spec = resolve_component(payload["component"]) if payload.get("component") else None
    cfg = _config_from(payload, spec)
    result = OverWatchPipeline(cfg).run(csv_text, component_name=spec.name if spec else None)
    return result, spec


def _payload_dict() -> dict:
    data = dict(request.form) if request.form else {}
    data.update(request.get_json(silent=True) or {})
    data.update({k: v for k, v in request.args.items()})
    return data


def _shape(result, spec, payload) -> dict:
    out = result.to_dict(include_records=str(payload.get("include_records", "")).lower() in ("1", "true"))
    for row in out["results"]:
        row["reason_codes"] = [c for c in (row.get("reason_codes") or "").split("|") if c]
    if spec is not None:
        out["meta"]["component_info"] = spec.to_public()
    if result.meta["has_ground_truth"]:
        ev = evaluate(result.components)
        ev.pop("confusion", None)
        out["evaluation"] = ev
    return out


# ----------------------------- routes --------------------------------------- #
@app.get("/api/health")
def health():
    return jsonify({"status": "online", "service": "OverWatch AI", "version": "1.0.0"})


@app.get("/api/components")
def components():
    return jsonify({"categories": taxonomy()})


@app.get("/api/generate")
def generate():
    spec = resolve_component(request.args.get("component", 5))
    df = generate_dataset(spec, seed=int(request.args.get("seed", 42)))
    if request.args.get("format") == "json":
        return jsonify({"component": spec.to_public(), "rows": df.to_dict(orient="records")})
    return Response(to_csv_text(df), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename={spec.id:02d}_{spec.code.lower()}_burnin.csv"})


@app.post("/api/analyze")
def analyze():
    payload = _payload_dict()
    if "file" in request.files:
        csv_text = request.files["file"].read().decode("utf-8-sig")
    elif payload.get("csv"):
        csv_text = payload["csv"]
    else:
        raise ValueError("Send a .csv as multipart field 'file' or JSON {'csv': '...'}")
    result, spec = _run(csv_text, payload)
    if request.args.get("format") == "csv":
        buf = io.StringIO()
        result.components.to_csv(buf, index=False)
        return Response(buf.getvalue(), mimetype="text/csv", headers={
            "Content-Disposition": "attachment; filename=overwatch_screening_report.csv"})
    return jsonify(_shape(result, spec, payload))


@app.post("/api/analyze/sample")
def analyze_sample():
    payload = _payload_dict()
    spec = resolve_component(payload.get("component", 5))
    df = generate_dataset(spec, seed=int(payload.get("seed", 42)))
    result = OverWatchPipeline(_config_from(payload, spec)).run(df, component_name=spec.name)
    return jsonify(_shape(result, spec, payload))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
