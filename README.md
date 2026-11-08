# OverWatch AI - AI-Driven Anomaly Detection in Component Burn-In & Screening

Backend + ML for your mini project. Three parts from your prompt file:

| Part | File | What it does |
|---|---|---|
| 1. Synthetic dataset | `data_generator.py`, `config.py` | 16 components x 3 categories, 5 batches x 20 units x 8 readings (0-168 h), 90% normal / 5% batch-relative anomaly / 5% latent drift, 2-decimal CSV |
| 2. ML model | `pipeline.py` | Stage 1 batch-aware preprocessing, Module A (Isolation Forest + robust batch z-score/ratio), Module B (Ridge drift velocity + 168 h projection), Risk engine + XAI reason codes |
| 3. Frontend | *(you generate in Bolt)* | Talks to `api.py` over REST |

## Setup

```bash
pip install -r requirements.txt
python main.py demo --component 5        # generate -> ML -> report in terminal
python -m pytest tests -q                # 24 tests
python main.py serve                     # REST API on http://localhost:5000
```

## CLI

```bash
python main.py generate --component LED          # one CSV (id 1-16, code, or name)
python main.py generate --all                    # all 16 -> sample_data/
python main.py analyze --input sample_data/05_ecap_burnin.csv --out report.csv
python main.py analyze --input x.csv --fit-until 96   # early screening: fit to 96 h, project to 168 h
```

## How the model works

1. **Stage 1** - load CSV, coerce numerics, interpolate / batch-median fill missing values, compute batch x hour mean/std/median, then per-reading batch-relative features (robust z-score using median/MAD, ratio to batch baseline).
2. **Module A** (unsupervised) - Isolation Forest on `[log ratio, primary z, secondary z, temp z]`. Per reading: `anomaly_score` (0-1, higher = more anomalous, ~0.5 typical) and `anomaly_flag` (`1` normal, `-1` outlier). Per component it also scores persistent deviation from the batch (e.g. 4.3x batch baseline) even when the part passes the datasheet limit.
3. **Module B** - Ridge regression per component gives drift velocity (slope/hour, %/100 h) plus an exponential trend; the larger of the two is projected to hour 168 (`projected_168h`).
4. **Risk engine** - `risk_score = 100 x (0.7 x max(A,B) + 0.3 x min(A,B))`; >= 55 RED, >= 25 YELLOW, else GREEN. Every row gets `reason_codes` (e.g. `BATCH_OUTLIER_PRIMARY`, `ACCELERATING_DRIFT`) and a plain-English `xai_reason`. Thresholds live in `PipelineConfig`.

Note: on clean synthetic data nearly everything is either GREEN or RED. YELLOW appears for borderline cases (mild drift, moderate batch deviation) - lower `ramp` thresholds in `PipelineConfig` or use real/noisier data to see more.

## Component -> column mapping

Column names come from the category (as in your prompt); what they mean per part is in `config.py` and returned by `/api/components` (`primary.label`, `primary.unit`, ...), so the UI can show real labels.

## REST API (base URL `http://localhost:5000`, CORS open)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, service, version}` - for the live status badge |
| GET | `/api/components` | `{categories:[{category_id, category, schema, components:[{id, code, name, primary{column,label,unit,nominal,datasheet_limit}, secondary{...}, stress_temp_c}]}]}` - drives the Category -> Component dropdowns |
| GET | `/api/generate?component=5&seed=42` | Download synthetic CSV (`&format=json` for rows) |
| POST | `/api/analyze` | multipart field `file` (.csv) or JSON `{"csv": "..."}`; optional `component`, `fit_until_hour`, `contamination`. `?format=csv` downloads the report |
| POST | `/api/analyze/sample` | JSON `{"component": 5, "seed": 42}` - generates and analyzes in one call (demo button) |

`/api/analyze` response:

```json
{
  "meta": {"component": "...", "primary_column": "...", "n_rows": 800, "config": {...}},
  "summary": {"total": 100, "green": 90, "yellow": 0, "red": 10,
              "by_batch": {"BATCH_A": {"green": 18, "yellow": 0, "red": 2}}},
  "results": [{"component_id": "ECAP-A02", "batch_id": "BATCH_A",
               "primary_value": 113.3, "secondary_value": 360.1, "temp_c": 111.4,
               "anomaly_score": 0.585, "anomaly_flag": -1,
               "drift_slope_per_hr": 0.406, "drift_pct_per_100h": 81.5, "projected_168h": 113.3,
               "risk_score": 85.0, "risk_category": "RED", "risk_label": "HIGH RISK / REJECTION (Red)",
               "reason_codes": ["ACCELERATING_DRIFT"], "xai_reason": "REJECT - ..."}],
  "batch_stats": [...],
  "series": {"ECAP-A02": [{"hour": 0, "primary": 41.2, "secondary": 471.0, "temp": 104.9}]},
  "evaluation": {...}   // only when the CSV has status_ground_truth
}
```

## Connecting the Bolt UI

Paste this into Bolt (it has your original UI requirements covered; this adds the wiring):

> Build the "OverWatch AI" dashboard (React + TypeScript + Tailwind). Dark `#1f2937` header titled "AI-Driven Anomaly Detection in Component Burn-In & Screening" with a live status badge from `GET {API}/api/health`; dark nav with smooth-scroll links (Dashboard, Component Config, Upload Data, Component Analysis, Reports). Category and Component dropdowns populated from `GET {API}/api/components` (component list updates when the category changes). Upload section: drag-and-drop `.csv`, plus a "Generate sample data" button calling `POST {API}/api/analyze/sample`; "Run AI Screening Analysis" posts the file as multipart field `file` (plus `component`) to `POST {API}/api/analyze` with a loading state. Show summary cards (total, green, yellow, red) from `summary`; a table from `results` (component_id, batch_id, primary_value, secondary_value, temp_c, anomaly_score, color-coded risk_category badge, projected_168h) with search/filter by risk and batch; clicking a row opens an XAI card showing `reason_codes` chips, `xai_reason`, and a line chart of `series[component_id]`. Reports section: button to download CSV via `POST {API}/api/analyze?format=csv`. Read `{API}` from `VITE_API_URL` (default `http://localhost:5000`).

Then send me the Bolt files and I'll wire/fix anything that doesn't line up with the API.

## Project layout

```
overwatch_ai/
  config.py          component catalog + schemas
  data_generator.py  Part 1
  pipeline.py        Part 2 (ML)
  api.py             REST API
  main.py            CLI
  requirements.txt
  sample_data/       16 ready CSVs (one per component)
  tests/test_project.py
```
