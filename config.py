"""
OverWatch AI - component catalog.

Defines the 16 industrial components, their 3 architectural categories, the CSV
column schema per category, and the physical baselines used by the synthetic
data generator (and by the API's /api/components endpoint for the UI dropdowns).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union

# --------------------------------------------------------------------------- #
# Global generation parameters (from the dataset prompt)
# --------------------------------------------------------------------------- #
BATCHES: List[str] = ["BATCH_A", "BATCH_B", "BATCH_C", "BATCH_D", "BATCH_E"]
UNITS_PER_BATCH: int = 20
BURN_IN_HOURS: List[int] = [0, 24, 48, 72, 96, 120, 144, 168]
HORIZON_HOUR: int = 168

# Fraction of units per batch assigned to each hidden class (5% of 20 = 1 unit)
BATCH_ANOMALY_FRACTION = 0.05
LATENT_DEFECT_FRACTION = 0.05

# Hard datasheet limit on the primary parameter = LIMIT_FACTOR x nominal mean.
# (Batch-relative anomalies at 3x-5x the batch mean therefore still PASS the limit.)
LIMIT_FACTOR = 6.5

# --------------------------------------------------------------------------- #
# Category -> CSV schema
# --------------------------------------------------------------------------- #
CATEGORIES: Dict[int, str] = {
    1: "Active Semiconductors",
    2: "Passive Components",
    3: "Electro-Mechanical & Power",
}

CATEGORY_COLUMNS: Dict[int, List[str]] = {
    1: [
        "component_id", "batch_id", "burn_in_hour",
        "primary_parameter_leakage_ua",
        "secondary_parameter_rds_or_gain",
        "thermal_stress_temp_c", "status_ground_truth",
    ],
    2: [
        "component_id", "batch_id", "burn_in_hour",
        "primary_parameter_esr_mohm",
        "secondary_parameter_capacitance_or_resistance",
        "thermal_stress_temp_c", "status_ground_truth",
    ],
    3: [
        "component_id", "batch_id", "burn_in_hour",
        "primary_parameter_resistance_or_vibration",
        "secondary_parameter_current_or_voltage_drop",
        "thermal_stress_temp_c", "status_ground_truth",
    ],
}


@dataclass
class ComponentSpec:
    id: int
    code: str                 # prefix used in component_id, e.g. "ECAP-A01"
    name: str
    category: int
    pri_label: str            # what the primary column physically means for this part
    pri_unit: str
    pri_mean: float           # nominal mean of the primary parameter
    sec_label: str
    sec_unit: str
    sec_mean: float
    sec_dir: int              # +1: secondary rises when degrading, -1: it falls
    temp_c: float             # burn-in chamber stress temperature
    pri_noise: float = 0.04   # relative test-fixture noise (primary)
    pri_spread: float = 0.08  # relative unit-to-unit spread inside a batch (primary)
    sec_noise: float = 0.01
    sec_spread: float = 0.03

    @property
    def category_name(self) -> str:
        return CATEGORIES[self.category]

    @property
    def columns(self) -> List[str]:
        return CATEGORY_COLUMNS[self.category]

    @property
    def primary_column(self) -> str:
        return self.columns[3]

    @property
    def secondary_column(self) -> str:
        return self.columns[4]

    @property
    def pri_limit(self) -> float:
        return round(self.pri_mean * LIMIT_FACTOR, 2)

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "category_id": self.category,
            "category": self.category_name,
            "primary": {"column": self.primary_column, "label": self.pri_label,
                        "unit": self.pri_unit, "nominal": self.pri_mean,
                        "datasheet_limit": self.pri_limit},
            "secondary": {"column": self.secondary_column, "label": self.sec_label,
                          "unit": self.sec_unit, "nominal": self.sec_mean},
            "stress_temp_c": self.temp_c,
        }


# id, code, name, cat, pri_label, pri_unit, pri_mean, sec_label, sec_unit, sec_mean, sec_dir, temp
_C = ComponentSpec
COMPONENTS: List[ComponentSpec] = [
    # ---- Category 1: Active Semiconductors -------------------------------- #
    _C(1, "MOS", "Power MOSFETs & IGBTs", 1, "Drain-source leakage (IDSS)", "uA", 2.0,
       "On-resistance Rds(on)", "mOhm", 45.0, +1, 85.0),
    _C(2, "BJT", "Bipolar Junction Transistors (BJTs)", 1, "Collector cut-off leakage (ICBO)", "uA", 1.5,
       "DC current gain hFE", "-", 180.0, -1, 85.0),
    _C(3, "IC", "Integrated Circuits (ICs) & Microcontrollers", 1, "Quiescent leakage (IDDQ)", "uA", 12.0,
       "Propagation delay", "ns", 8.0, +1, 105.0),
    _C(4, "PMIC", "Voltage Regulators / PMICs", 1, "Quiescent / ground-pin leakage", "uA", 20.0,
       "Pass-element Rds(on)", "mOhm", 120.0, +1, 95.0),
    # ---- Category 2: Passive Components ----------------------------------- #
    _C(5, "ECAP", "Electrolytic Capacitors", 2, "Equivalent series resistance (ESR)", "mOhm", 45.0,
       "Capacitance", "uF", 470.0, -1, 105.0),
    _C(6, "CCAP", "Ceramic Capacitors", 2, "Equivalent series resistance (ESR)", "mOhm", 8.0,
       "Capacitance", "uF", 10.0, -1, 125.0),
    _C(7, "PRES", "Power Resistors", 2, "Termination / lead resistance", "mOhm", 25.0,
       "Resistance", "Ohm", 100.0, +1, 125.0),
    _C(8, "IND", "Inductors", 2, "DC winding resistance (DCR)", "mOhm", 35.0,
       "Inductance", "uH", 22.0, -1, 105.0),
    _C(9, "XFMR", "Transformers (Inductive Components)", 2, "Primary winding resistance", "mOhm", 120.0,
       "Inter-winding capacitance", "pF", 150.0, +1, 105.0),
    # ---- Category 3: Electro-Mechanical & Power --------------------------- #
    _C(10, "MOT", "Industrial Motors (DC/BLDC/Induction)", 3, "Vibration velocity", "mm/s", 2.0,
       "Phase current", "A", 3.5, +1, 80.0),
    _C(11, "RLY", "Electromechanical Relays", 3, "Contact resistance", "mOhm", 50.0,
       "Contact voltage drop", "mV", 100.0, +1, 85.0),
    _C(12, "CTR", "Contactors", 3, "Contact resistance", "mOhm", 80.0,
       "Coil current", "mA", 180.0, +1, 85.0),
    _C(13, "LED", "Light-Emitting Diodes (LEDs)", 3, "Dynamic resistance", "Ohm", 8.0,
       "Forward voltage drop", "V", 3.10, +1, 85.0),
    _C(14, "OPT", "Optocouplers", 3, "Input dynamic resistance", "Ohm", 25.0,
       "Output saturation voltage VCE(sat)", "mV", 200.0, +1, 100.0),
    _C(15, "DIO", "Discrete Diodes / Rectifiers", 3, "Series resistance", "mOhm", 30.0,
       "Forward voltage drop", "mV", 700.0, +1, 125.0),
    _C(16, "TVS", "Transient Voltage Suppression (TVS) Diodes", 3, "Dynamic resistance", "mOhm", 150.0,
       "Standby leakage current", "uA", 5.0, +1, 125.0),
]

_BY_ID = {c.id: c for c in COMPONENTS}


def resolve_component(ref: Union[int, str, ComponentSpec]) -> ComponentSpec:
    """Find a component by id (1-16), code ('ECAP'), or (partial) name."""
    if isinstance(ref, ComponentSpec):
        return ref
    if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
        cid = int(ref)
        if cid in _BY_ID:
            return _BY_ID[cid]
        raise ValueError(f"Component id must be 1-16, got {cid}")
    text = str(ref).strip().lower()
    for c in COMPONENTS:
        if text == c.code.lower() or text == c.name.lower():
            return c
    for c in COMPONENTS:
        if text and text in c.name.lower():
            return c
    raise ValueError(f"Unknown component: {ref!r}")


def taxonomy() -> List[dict]:
    """Category -> components structure used by the UI's dependent dropdowns."""
    out = []
    for cat_id, cat_name in CATEGORIES.items():
        out.append({
            "category_id": cat_id,
            "category": cat_name,
            "schema": CATEGORY_COLUMNS[cat_id],
            "components": [c.to_public() for c in COMPONENTS if c.category == cat_id],
        })
    return out


if __name__ == "__main__":
    for cat in taxonomy():
        print(f"Category {cat['category_id']}: {cat['category']}")
        for c in cat["components"]:
            print(f"   {c['id']:>2}. {c['name']}  [{c['code']}]")
