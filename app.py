import os
import re
import io
import json
import math
import uuid
import shutil
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    VIDEO_DEPS = True
except Exception:
    VIDEO_DEPS = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_DEPS = True
except Exception:
    SKLEARN_DEPS = False


APP_TITLE = "Credit Portfolio Early-Warning and Explainability Platform"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
DEMO_DIR = os.path.join(DATA_DIR, "demo")
ERROR_DIR = os.path.join(DATA_DIR, "errors")
DB_PATH = os.path.join(DATA_DIR, "credit_early_warning.sqlite3")

for _d in [DATA_DIR, UPLOAD_DIR, BACKUP_DIR, DEMO_DIR, ERROR_DIR]:
    os.makedirs(_d, exist_ok=True)

SCHEMA_VERSION = "1.0.0"
RULES_VERSION = "ew-rules-1.0.0"
ML_VERSION = "logreg-baseline-1.0.0"
SYNTHETIC_TAG = "SYNTHETIC_DEMO"

ROLE_ACTORS = {
    "administrator": "usr_admin",
    "analyst_operator": "usr_analyst",
    "reviewer": "usr_reviewer",
    "read_only_stakeholder": "usr_readonly",
}

ROLE_LABELS = {
    "administrator": "Administrator",
    "analyst_operator": "Analyst / Operator",
    "reviewer": "Reviewer",
    "read_only_stakeholder": "Read-only Stakeholder",
}

ROLE_PERMISSIONS = {
    "administrator": {
        "view", "import", "run_engine", "submit_review", "approve", "override",
        "archive", "export", "manage_config", "backup_restore", "delete",
        "manage_scenarios",
    },
    "analyst_operator": {
        "view", "import", "run_engine", "submit_review", "export", "annotate",
        "manage_scenarios",
    },
    "reviewer": {
        "view", "run_engine", "approve", "override", "archive", "export",
        "signoff", "manage_scenarios",
    },
    "read_only_stakeholder": {
        "view", "export_summary",
    },
}

PAGES = [
    "Live Monitor",
    "Product Brief",
    "Data Import",
    "Portfolio Records",
    "Early-Warning Engine",
    "Scenarios & Forecast",
    "Review & Approvals",
    "Explainability",
    "Evaluation & Models",
    "Reports & Export",
    "Audit & Governance",
    "Data Dictionary",
    "Self Tests",
]

RULE_SETS = {
    "ew-rules-1.0.0": {
        "dpd_risk": 0.35,
        "arrears_risk": 0.25,
        "utilization_risk": 0.20,
        "payment_risk": 0.10,
        "macro_stress": 0.10,
    },
    "ew-rules-1.0.1": {
        "dpd_risk": 0.30,
        "arrears_risk": 0.30,
        "utilization_risk": 0.15,
        "payment_risk": 0.15,
        "macro_stress": 0.10,
    },
}

ALLOWED_TRANSITIONS = {
    "generated": ["validated", "under_review", "corrected", "needs_validation"],
    "needs_validation": ["corrected", "validated", "under_review"],
    "validated": ["under_review", "approved", "corrected"],
    "under_review": ["approved", "corrected", "archived", "validated"],
    "approved": ["archived", "exported", "corrected"],
    "corrected": ["validated", "under_review"],
    "archived": ["exported"],
    "exported": ["archived"],
}

IMPORT_SCHEMAS = {
    "accounts": {
        "required": [
            "account_ref", "org_name", "product", "limit_amount", "balance",
            "currency", "origination_date", "maturity_date", "status",
        ],
        "numeric_cols": ["limit_amount", "balance"],
        "ranges": {
            "limit_amount": (0.0, 1e15),
            "balance": (0.0, 1e15),
        },
        "date_cols": ["origination_date", "maturity_date"],
        "enums": {
            "status": ["active", "closed", "written_off", "deferred"],
        },
        "unique": ["account_ref"],
    },
    "repayments": {
        "required": [
            "account_ref", "due_date", "payment_date", "amount", "currency", "status",
        ],
        "numeric_cols": ["amount", "days_late"],
        "ranges": {
            "amount": (0.0, 1e15),
            "days_late": (0, 3650),
        },
        "date_cols": ["due_date", "payment_date"],
        "enums": {
            "status": ["on_time", "late", "missed", "partial"],
        },
        "unique": ["account_ref", "due_date", "payment_date", "amount"],
        "ref_account": True,
    },
    "exposures": {
        "required": [
            "account_ref", "observation_date", "drawn", "undrawn",
            "collateral_value", "ead", "currency",
        ],
        "numeric_cols": ["drawn", "undrawn", "collateral_value", "ead"],
        "ranges": {
            "drawn": (0.0, 1e15),
            "undrawn": (0.0, 1e15),
            "collateral_value": (0.0, 1e15),
            "ead": (0.0, 1e15),
        },
        "date_cols": ["observation_date"],
        "unique": ["account_ref", "observation_date"],
        "ref_account": True,
    },
    "arrears": {
        "required": [
            "account_ref", "observation_date", "days_past_due",
            "arrears_balance", "bucket",
        ],
        "numeric_cols": ["days_past_due", "arrears_balance"],
        "ranges": {
            "days_past_due": (0, 3650),
            "arrears_balance": (0.0, 1e15),
        },
        "date_cols": ["observation_date"],
        "enums": {
            "bucket": ["current", "1-30", "31-60", "61-90", "90+"],
        },
        "unique": ["account_ref", "observation_date"],
        "ref_account": True,
    },
    "macro": {
        "required": [
            "observation_date", "indicator", "value", "unit",
            "jurisdiction", "scenario_tag",
        ],
        "numeric_cols": ["value"],
        "date_cols": ["observation_date"],
        "unique": ["observation_date", "indicator", "jurisdiction", "scenario_tag"],
    },
}


class DomainValidationError(ValueError):
    pass


class ImportValidationError(ValueError):
    pass


@dataclass
class ValidationIssue:
    row: int
    column: str
    value: str
    issue: str
    severity: str


@dataclass
class MetricResult:
    account_id: str
    account_ref: str
    ead: float
    dpd: int
    arrears_balance: float
    utilization: float
    on_time_rate: float
    late_frequency: float
    macro_stress: float
    score: float
    severity: str
    factors: Dict[str, Any]
    data_quality: List[str]
    uncertainty: float
    rules_version: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_row_hash(parts: List[Any]) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    return str(obj)


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=json_default)


def safe_json_loads(value: Any) -> Any:
    if value is None or value == "":
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def bucket_for_dpd(dpd: int) -> str:
    dpd = int(dpd)
    if dpd <= 0:
        return "current"
    if dpd <= 30:
        return "1-30"
    if dpd <= 60:
        return "31-60"
    if dpd <= 90:
        return "61-90"
    return "90+"


def as_float(value: Any) -> Optional[float]:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return float(num)


def as_int(value: Any) -> Optional[int]:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return int(num)


def as_date_str(value: Any) -> Optional[str]:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS actors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_systems (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system_type TEXT,
    contact TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS reference_catalogs (
    id TEXT PRIMARY KEY,
    catalog_name TEXT,
    version TEXT,
    effective_from TEXT,
    payload_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    jurisdiction TEXT,
    currency TEXT,
    sector TEXT,
    created_at TEXT,
    updated_at TEXT,
    source_system_id TEXT,
    schema_version TEXT
);

CREATE TABLE IF NOT EXISTS consents (
    id TEXT PRIMARY KEY,
    account_id TEXT,
    consent_type TEXT,
    granted INTEGER,
    effective_date TEXT,
    expiry_date TEXT,
    source_system_id TEXT,
    schema_version TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    org_id TEXT,
    account_ref TEXT NOT NULL UNIQUE,
    product TEXT,
    facility_type TEXT,
    jurisdiction TEXT,
    currency TEXT,
    limit_amount REAL,
    balance REAL,
    origination_date TEXT,
    maturity_date TEXT,
    status TEXT,
    source_system_id TEXT,
    schema_version TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS repayments (
    id TEXT PRIMARY KEY,
    account_id TEXT,
    due_date TEXT,
    payment_date TEXT,
    amount REAL,
    currency TEXT,
    channel TEXT,
    status TEXT,
    days_late INTEGER,
    source_file_id TEXT,
    row_number INTEGER,
    row_hash TEXT UNIQUE,
    schema_version TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS exposures (
    id TEXT PRIMARY KEY,
    account_id TEXT,
    observation_date TEXT,
    drawn REAL,
    undrawn REAL,
    collateral_value REAL,
    ead REAL,
    currency TEXT,
    source_file_id TEXT,
    row_number INTEGER,
    row_hash TEXT UNIQUE,
    schema_version TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS arrears (
    id TEXT PRIMARY KEY,
    account_id TEXT,
    observation_date TEXT,
    days_past_due INTEGER,
    arrears_balance REAL,
    bucket TEXT,
    cure_flag INTEGER,
    source_file_id TEXT,
    row_number INTEGER,
    row_hash TEXT UNIQUE,
    schema_version TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS macro_observations (
    id TEXT PRIMARY KEY,
    observation_date TEXT,
    indicator TEXT,
    value REAL,
    unit TEXT,
    jurisdiction TEXT,
    source_system_id TEXT,
    scenario_tag TEXT,
    row_hash TEXT UNIQUE,
    schema_version TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS scenarios (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    horizon_months INTEGER,
    parameters_json TEXT,
    created_by TEXT,
    created_at TEXT,
    version TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    account_id TEXT,
    scenario_id TEXT,
    generated_at TEXT,
    rules_version TEXT,
    severity TEXT,
    score REAL,
    factors_json TEXT,
    data_quality_json TEXT,
    uncertainty_json TEXT,
    status TEXT,
    reviewer_id TEXT,
    approval_at TEXT,
    archive_at TEXT,
    export_batch_id TEXT,
    schema_version TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    alert_id TEXT,
    actor_id TEXT,
    role TEXT,
    action TEXT,
    reason TEXT,
    override_score REAL,
    override_severity TEXT,
    created_at TEXT,
    audit_event_id TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    ts TEXT,
    actor_id TEXT,
    role TEXT,
    action TEXT,
    entity_type TEXT,
    entity_id TEXT,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    filename TEXT,
    sha256 TEXT,
    size INTEGER,
    uploaded_at TEXT,
    stored_path TEXT,
    import_status TEXT,
    batch_id TEXT,
    rows_total INTEGER,
    rows_valid INTEGER,
    rows_invalid INTEGER,
    error_report_path TEXT,
    entity_type TEXT,
    schema_version TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    file_id TEXT,
    imported_at TEXT,
    actor_id TEXT,
    status TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    id TEXT PRIMARY KEY,
    model_name TEXT,
    version TEXT,
    trained_at TEXT,
    training_data_version TEXT,
    feature_schema_json TEXT,
    metrics_json TEXT,
    limitations TEXT,
    status TEXT,
    artifact_json TEXT
);
"""


def init_db() -> None:
    with db_conn() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("schema_version", SCHEMA_VERSION, now_utc()),
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("synthetic_demo", "true", now_utc()),
        )


def clear_all(conn: sqlite3.Connection) -> None:
    tables = [
        "decisions", "alerts", "audit_events", "files", "batches",
        "model_registry", "macro_observations", "arrears", "exposures",
        "repayments", "consents", "accounts", "organizations",
        "reference_catalogs", "source_systems", "actors", "scenarios", "meta",
    ]
    for t in tables:
        conn.execute(f"DELETE FROM {t}")


def log_audit_conn(
    conn: sqlite3.Connection,
    action: str,
    entity_type: str,
    entity_id: str,
    details: Optional[Dict[str, Any]] = None,
    actor_id: str = "system",
    role: str = "system",
) -> str:
    event_id = new_id("aud")
    conn.execute(
        """
        INSERT INTO audit_events(id, ts, actor_id, role, action, entity_type, entity_id, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, now_utc(), actor_id, role, action, entity_type, entity_id, to_json(details or {})),
    )
    return event_id


def log_audit(
    action: str,
    entity_type: str,
    entity_id: str,
    details: Optional[Dict[str, Any]] = None,
    actor_id: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    actor_id = actor_id or st.session_state.get("actor_id", "system")
    role = role or st.session_state.get("role", "system")
    with db_conn() as conn:
        return log_audit_conn(conn, action, entity_type, entity_id, details, actor_id, role)


def can(permission: str) -> bool:
    role = st.session_state.get("role", "read_only_stakeholder")
    return permission in ROLE_PERMISSIONS.get(role, set())


def get_active_rules_version(conn: Optional[sqlite3.Connection] = None) -> str:
    if conn is not None:
        row = conn.execute("SELECT value FROM meta WHERE key='active_rules_version'").fetchone()
        if row and row["value"]:
            return str(row["value"])
    return st.session_state.get("active_rules_version", RULES_VERSION)


def scenario_stress(params: Dict[str, Any]) -> float:
    return clamp01(
        0.40 * clamp01(float(params.get("unemployment_delta", 0.0)) / 5.0)
        + 0.30 * clamp01(float(params.get("rate_delta", 0.0)) / 5.0)
        + 0.20 * clamp01(float(params.get("fx_depreciation_pct", 0.0)) / 20.0)
        + 0.10 * clamp01(float(params.get("sector_shock", 0.0)))
    )


def compute_weighted_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    return 100.0 * sum(float(components.get(k, 0.0)) * float(weights.get(k, 0.0)) for k in weights)


def compute_rule_score_row(row: pd.Series, weights: Dict[str, float]) -> float:
    components = {
        "dpd_risk": clamp01(float(row.get("days_past_due", 0.0)) / 90.0),
        "arrears_risk": clamp01(float(row.get("arrears_ratio", 0.0))),
        "utilization_risk": clamp01(float(row.get("utilization", 0.0))),
        "payment_risk": clamp01(1.0 - float(row.get("on_time_rate", 0.5))),
        "macro_stress": clamp01(float(row.get("macro_stress", 0.0))),
    }
    return compute_weighted_score(components, weights)


def calculate_account_metrics_conn(
    conn: sqlite3.Connection,
    acc: sqlite3.Row,
    scenario_params: Dict[str, Any],
    rules_version: Optional[str] = None,
) -> MetricResult:
    rules_version = rules_version or get_active_rules_version(conn)
    weights = RULE_SETS.get(rules_version, RULE_SETS[RULES_VERSION])
    data_quality: List[str] = []

    exp = conn.execute(
        "SELECT * FROM exposures WHERE account_id=? ORDER BY observation_date DESC LIMIT 1",
        (acc["id"],),
    ).fetchone()
    arr = conn.execute(
        "SELECT * FROM arrears WHERE account_id=? ORDER BY observation_date DESC LIMIT 1",
        (acc["id"],),
    ).fetchone()

    if exp is None:
        ead = float("nan")
        drawn = 0.0
        data_quality.append("CRITICAL missing exposure observation")
    else:
        ead = float(exp["ead"] or 0.0)
        drawn = float(exp["drawn"] or 0.0)

    if arr is None:
        dpd = 0
        arrears_balance = 0.0
        data_quality.append("WARNING no arrears observation; assumed current")
    else:
        dpd = int(arr["days_past_due"] or 0)
        arrears_balance = float(arr["arrears_balance"] or 0.0)

    limit_amount = float(acc["limit_amount"] or 0.0)
    balance = float(acc["balance"] or 0.0)

    if limit_amount <= 0:
        data_quality.append("WARNING non-positive credit limit")
    if not math.isnan(ead) and ead > 0 and arrears_balance > ead * 1.1:
        data_quality.append("CRITICAL arrears balance exceeds exposure")
    if not math.isnan(ead) and drawn > ead * 1.05:
        data_quality.append("WARNING drawn exceeds EAD materially")

    cutoff = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(days=365)).date().isoformat()
    reps = conn.execute(
        "SELECT status, days_late, amount FROM repayments WHERE account_id=? AND payment_date >= ?",
        (acc["id"], cutoff),
    ).fetchall()

    if not reps:
        on_time_rate = 0.5
        late_frequency = 0.5
        payment_var = 0.5
        data_quality.append("WARNING no repayment history in last 365 days")
    else:
        total = len(reps)
        on_time = sum(1 for r in reps if str(r["status"]).lower() == "on_time")
        on_time_rate = on_time / float(total) if total else 0.5
        late_frequency = 1.0 - on_time_rate
        amounts = [float(r["amount"]) for r in reps if r["amount"] is not None]
        if len(amounts) > 1 and np.mean(amounts) != 0:
            payment_var = float(np.std(amounts) / np.mean(amounts))
        else:
            payment_var = 0.0

    macro_stress = scenario_stress(scenario_params)

    dpd_risk = clamp01(dpd / 90.0)
    arrears_risk = 0.0 if math.isnan(ead) else clamp01(arrears_balance / max(ead, 1.0))
    utilization_risk = clamp01(balance / limit_amount) if limit_amount > 0 else 0.5
    payment_risk = clamp01(late_frequency)
    macro_risk = clamp01(macro_stress)

    components = {
        "dpd_risk": dpd_risk,
        "arrears_risk": arrears_risk,
        "utilization_risk": utilization_risk,
        "payment_risk": payment_risk,
        "macro_stress": macro_risk,
    }
    weighted = {k: components[k] * weights.get(k, 0.0) for k in weights}
    score = compute_weighted_score(components, weights)

    critical = any(q.startswith("CRITICAL") for q in data_quality)
    if critical:
        severity = "data_exception"
    elif score >= 75:
        severity = "high"
    elif score >= 50:
        severity = "medium"
    elif score >= 25:
        severity = "low"
    else:
        severity = "watch"

    warning_count = sum(1 for q in data_quality if q.startswith("WARNING"))
    uncertainty = 0.05 + 0.03 * warning_count + (0.20 if critical else 0.0) + 0.10 * clamp01(payment_var)
    uncertainty = max(0.01, min(0.99, uncertainty))

    return MetricResult(
        account_id=str(acc["id"]),
        account_ref=str(acc["account_ref"]),
        ead=0.0 if math.isnan(ead) else float(ead),
        dpd=int(dpd),
        arrears_balance=float(arrears_balance),
        utilization=float(utilization_risk),
        on_time_rate=float(on_time_rate),
        late_frequency=float(late_frequency),
        macro_stress=float(macro_stress),
        score=float(score),
        severity=severity,
        factors={
            "components": components,
            "weighted_contributions": weighted,
            "weights": weights,
        },
        data_quality=data_quality,
        uncertainty=float(uncertainty),
        rules_version=rules_version,
    )


def run_early_warning_engine(
    scenario_id: str,
    actor_id: Optional[str] = None,
    role: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    actor_id = actor_id or st.session_state.get("actor_id", "system")
    role = role or st.session_state.get("role", "system")
    results: List[MetricResult] = []

    with db_conn() as conn:
        scen = conn.execute("SELECT * FROM scenarios WHERE id=?", (scenario_id,)).fetchone()
        if not scen:
            raise DomainValidationError("Scenario not found")
        params = safe_json_loads(scen["parameters_json"])
        rules_version = get_active_rules_version(conn)
        accounts = conn.execute("SELECT * FROM accounts ORDER BY account_ref").fetchall()

        for acc in accounts:
            metrics = calculate_account_metrics_conn(conn, acc, params, rules_version)
            critical = any(q.startswith("CRITICAL") for q in metrics.data_quality)
            new_status = "needs_validation" if critical else "generated"

            existing = conn.execute(
                """
                SELECT * FROM alerts
                WHERE account_id=? AND scenario_id=? AND status NOT IN ('archived','superseded','exported')
                ORDER BY generated_at DESC LIMIT 1
                """,
                (acc["id"], scenario_id),
            ).fetchone()

            if existing:
                if existing["status"] in ("generated", "validated", "needs_validation") or force:
                    alert_id = existing["id"]
                    conn.execute(
                        """
                        UPDATE alerts
                        SET generated_at=?, rules_version=?, severity=?, score=?, factors_json=?,
                            data_quality_json=?, uncertainty_json=?, status=?
                        WHERE id=?
                        """,
                        (
                            now_utc(), metrics.rules_version, metrics.severity, metrics.score,
                            to_json(metrics.factors), to_json(metrics.data_quality),
                            to_json({"uncertainty": metrics.uncertainty}), new_status, alert_id,
                        ),
                    )
                    log_audit_conn(
                        conn, "alert_recalculated", "alert", alert_id,
                        {"score": metrics.score, "severity": metrics.severity},
                        actor_id, role,
                    )
                else:
                    continue
            else:
                alert_id = new_id("alert")
                conn.execute(
                    """
                    INSERT INTO alerts(
                        id, account_id, scenario_id, generated_at, rules_version, severity, score,
                        factors_json, data_quality_json, uncertainty_json, status, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_id, acc["id"], scenario_id, now_utc(), metrics.rules_version,
                        metrics.severity, metrics.score, to_json(metrics.factors),
                        to_json(metrics.data_quality), to_json({"uncertainty": metrics.uncertainty}),
                        new_status, SCHEMA_VERSION,
                    ),
                )
                log_audit_conn(
                    conn, "alert_created", "alert", alert_id,
                    {"account_ref": metrics.account_ref, "score": metrics.score, "severity": metrics.severity},
                    actor_id, role,
                )

            results.append(metrics)

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("last_engine_run", now_utc(), now_utc()),
        )

    df = pd.DataFrame([asdict(m) for m in results]) if results else pd.DataFrame()
    return df


def get_monthly_delinquency_df_conn(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT substr(e.observation_date,1,7) AS month,
               SUM(a.arrears_balance) AS arrears,
               SUM(e.ead) AS ead
        FROM exposures e
        JOIN arrears a ON a.account_id=e.account_id AND a.observation_date=e.observation_date
        GROUP BY month
        ORDER BY month
        """,
        conn,
    )
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"] + "-01", errors="coerce")
    df["delinquency_rate"] = np.where(df["ead"] > 0, df["arrears"] / df["ead"], 0.0)
    return df


def forecast_delinquency(history: pd.DataFrame, horizon_months: int, params: Dict[str, Any]) -> pd.DataFrame:
    if history is None or history.empty or horizon_months <= 0:
        return pd.DataFrame()

    h = history.tail(12).copy().reset_index(drop=True)
    y = h["delinquency_rate"].astype(float).values
    x = np.arange(len(y), dtype=float)

    if len(y) < 2:
        base_val = float(y[-1]) if len(y) else 0.0
        fut = np.full(horizon_months, base_val)
        std = 0.01
        last_month = h["month"].iloc[-1] if "month" in h and not h.empty else pd.Timestamp.now(tz="UTC").normalize()
        months = [last_month + pd.DateOffset(months=int(i + 1)) for i in range(horizon_months)]
    else:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        resid = y - fitted
        std = float(np.std(resid)) if len(resid) > 1 else 0.01
        fx = np.arange(len(y), len(y) + horizon_months, dtype=float)
        fut = np.clip(intercept + slope * fx, 0.0, 1.0)
        last_month = h["month"].iloc[-1]
        months = [last_month + pd.DateOffset(months=int(i + 1)) for i in range(horizon_months)]

    stress = scenario_stress(params)
    mult = 1.0 + 0.6 * stress
    scen = np.clip(fut * mult, 0.0, 1.0)
    se = std * math.sqrt(1.0 + 1.0 / max(len(y), 1))
    upper = np.clip(scen + 1.96 * se, 0.0, 1.0)
    lower = np.clip(scen - 1.96 * se, 0.0, 1.0)

    return pd.DataFrame({
        "month": months,
        "base": fut,
        "scenario": scen,
        "lower": lower,
        "upper": upper,
        "uncertainty": se,
    })


def seed_demo_data(force: bool = False) -> None:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key='demo_seeded'").fetchone()
        if row and not force:
            return
        if force:
            clear_all(conn)

        now = now_utc()
        rng = np.random.default_rng(20261112)

        actors = [
            ("usr_admin", "Platform Administrator", "administrator", now),
            ("usr_analyst", "Portfolio Analyst", "analyst_operator", now),
            ("usr_reviewer", "Credit Reviewer", "reviewer", now),
            ("usr_readonly", "Treasury Stakeholder", "read_only_stakeholder", now),
        ]
        conn.executemany(
            "INSERT INTO actors(id, name, role, created_at) VALUES (?, ?, ?, ?)",
            actors,
        )

        sources = [
            ("src_core", "Core Loan System", "ledger", "core-ops@example.local", now),
            ("src_payments", "Payments Gateway", "payments", "pay-ops@example.local", now),
            ("src_market", "Market Data Feed", "market", "market-ops@example.local", now),
            ("src_manual", "Manual Upload", "file", "analyst@example.local", now),
        ]
        conn.executemany(
            "INSERT INTO source_systems(id, name, system_type, contact, created_at) VALUES (?, ?, ?, ?, ?)",
            sources,
        )

        conn.execute(
            """
            INSERT INTO reference_catalogs(id, catalog_name, version, effective_from, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("ref"), "product_catalog", "1.0", now,
                to_json(["term_loan", "revolver", "invoice_finance", "trade_receivable"]), now,
            ),
        )

        org_ids = []
        org_rows = []
        sectors = ["Manufacturing", "Retail", "Logistics", "Agribusiness", "Technology", "Healthcare"]
        jurisdictions = ["US", "GB", "EU", "SG", "IN", "BR"]
        for i in range(1, 31):
            oid = f"org_{i:03d}"
            org_ids.append(oid)
            org_rows.append((
                oid,
                f"{SYNTHETIC_TAG} Corp {i:02d}",
                jurisdictions[(i - 1) % len(jurisdictions)],
                "USD",
                sectors[(i - 1) % len(sectors)],
                now,
                now,
                "src_core",
                SCHEMA_VERSION,
            ))
        conn.executemany(
            """
            INSERT INTO organizations(id, name, jurisdiction, currency, sector, created_at, updated_at, source_system_id, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            org_rows,
        )

        products = ["term_loan", "revolver", "invoice_finance", "trade_receivable"]
        facility_types = ["secured", "unsecured", "receivable_backed"]
        statuses = ["active", "active", "active", "deferred", "closed"]

        account_rows = []
        consent_rows = []
        account_info = []

        for i in range(1, 121):
            aid = f"acc_{i:03d}"
            aref = f"DEMO-{i:05d}"
            org = org_ids[(i - 1) % len(org_ids)]
            product = str(rng.choice(products))
            facility = str(rng.choice(facility_types))
            jur = jurisdictions[(i - 1) % len(jurisdictions)]
            limit = round(float(rng.lognormal(11.5, 0.5)), 2)
            balance = round(limit * float(rng.uniform(0.10, 0.95)), 2)
            orig = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=int(rng.integers(6, 36)))).date().isoformat()
            mat = (pd.Timestamp(orig) + pd.DateOffset(months=int(rng.integers(12, 60)))).date().isoformat()
            status = str(rng.choice(statuses))
            profile = str(rng.choice(["healthy", "watch", "stress"], p=[0.62, 0.25, 0.13]))

            account_rows.append((
                aid, org, aref, product, facility, jur, "USD", limit, balance,
                orig, mat, status, "src_core", SCHEMA_VERSION, now, now,
            ))
            consent_rows.append((
                new_id("cns"), aid, "portfolio_monitoring", 1, orig,
                (pd.Timestamp(mat) + pd.DateOffset(days=30)).date().isoformat(),
                "src_core", SCHEMA_VERSION, now,
            ))
            account_info.append((aid, aref, limit, profile))

        conn.executemany(
            """
            INSERT INTO accounts(
                id, org_id, account_ref, product, facility_type, jurisdiction, currency,
                limit_amount, balance, origination_date, maturity_date, status,
                source_system_id, schema_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            account_rows,
        )
        conn.executemany(
            """
            INSERT INTO consents(
                id, account_id, consent_type, granted, effective_date, expiry_date,
                source_system_id, schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            consent_rows,
        )

        base_ts = pd.Timestamp.now(tz="UTC").normalize()
        months = [(base_ts - pd.DateOffset(months=m)).normalize().date().isoformat() for m in range(18, -1, -1)]

        exposure_rows = []
        arrears_rows = []
        repayment_rows = []

        for aid, aref, limit, profile in account_info:
            for mi, obs_date in enumerate(months):
                season = 1.0 + 0.08 * math.sin(mi / 3.0)

                if profile == "healthy":
                    dpd = max(0, int(rng.normal(2, 3)))
                    arrears_ratio = max(0.0, float(rng.normal(0.01, 0.01)))
                elif profile == "watch":
                    dpd = max(0, int(rng.normal(22, 12)))
                    arrears_ratio = max(0.0, float(rng.normal(0.08, 0.04)))
                else:
                    dpd = max(0, int(rng.normal(65, 25)))
                    arrears_ratio = max(0.0, float(rng.normal(0.22, 0.08)))

                if profile != "healthy" and mi > 12:
                    dpd += int((mi - 12) * rng.integers(1, 4))

                ead = round(float(limit * rng.uniform(0.45, 0.95) * season), 2)
                drawn = round(ead * float(rng.uniform(0.60, 0.95)), 2)
                undrawn = round(max(0.0, limit - drawn), 2)
                collateral = round(ead * float(rng.uniform(0.0, 0.35)), 2)
                arrears_balance = round(min(ead, ead * arrears_ratio), 2)
                bucket = bucket_for_dpd(dpd)

                exposure_rows.append((
                    new_id("exp"), aid, obs_date, drawn, undrawn, collateral, ead, "USD",
                    None, None, make_row_hash(["seed-exp", aid, obs_date, ead]), SCHEMA_VERSION, now,
                ))
                arrears_rows.append((
                    new_id("arr"), aid, obs_date, dpd, arrears_balance, bucket,
                    1 if dpd == 0 else 0, None, None,
                    make_row_hash(["seed-arr", aid, obs_date, dpd, arrears_balance]),
                    SCHEMA_VERSION, now,
                ))

                due_date = obs_date
                if dpd <= 5:
                    status = "on_time"
                    days_late = 0
                    pay_date = due_date
                elif dpd <= 30:
                    status = "late"
                    days_late = int(rng.integers(1, 31))
                    pay_date = (pd.Timestamp(due_date) + pd.Timedelta(days=days_late)).date().isoformat()
                elif rng.random() < 0.70:
                    status = "late"
                    days_late = min(dpd, 90)
                    pay_date = (pd.Timestamp(due_date) + pd.Timedelta(days=days_late)).date().isoformat()
                else:
                    status = "missed"
                    days_late = dpd
                    pay_date = due_date

                amount = 0.0 if status == "missed" else round(float(limit * rng.uniform(0.01, 0.05)), 2)
                channel = str(rng.choice(["ach", "wire", "card", "direct_debit"]))

                repayment_rows.append((
                    new_id("rep"), aid, due_date, pay_date, amount, "USD", channel, status,
                    days_late, None, None,
                    make_row_hash(["seed-rep", aid, due_date, pay_date, amount]),
                    SCHEMA_VERSION, now,
                ))

        conn.executemany(
            """
            INSERT INTO exposures(
                id, account_id, observation_date, drawn, undrawn, collateral_value, ead,
                currency, source_file_id, row_number, row_hash, schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            exposure_rows,
        )
        conn.executemany(
            """
            INSERT INTO arrears(
                id, account_id, observation_date, days_past_due, arrears_balance, bucket,
                cure_flag, source_file_id, row_number, row_hash, schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            arrears_rows,
        )
        conn.executemany(
            """
            INSERT INTO repayments(
                id, account_id, due_date, payment_date, amount, currency, channel, status,
                days_late, source_file_id, row_number, row_hash, schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            repayment_rows,
        )

        indicators = [
            ("unemployment_rate", "percent", 4.2),
            ("policy_rate", "percent", 3.5),
            ("credit_spread_bps", "bps", 180.0),
            ("fx_stress_index", "index", 0.35),
        ]
        macro_rows = []
        for obs_date in months:
            for tag in ["base", "downside", "upside"]:
                shock = {"base": 0.0, "downside": 1.0, "upside": -1.0}[tag]
                for ind, unit, base_val in indicators:
                    if ind == "fx_stress_index":
                        value = round(float(np.clip(base_val + shock * rng.uniform(0.02, 0.08), 0.0, 1.0)), 4)
                    elif ind == "credit_spread_bps":
                        value = round(base_val + shock * rng.uniform(20.0, 80.0), 2)
                    else:
                        value = round(base_val + shock * rng.uniform(0.3, 1.2), 4)
                    macro_rows.append((
                        new_id("mac"), obs_date, ind, value, unit, "US", "src_market", tag,
                        make_row_hash(["seed-macro", obs_date, ind, "US", tag, value]),
                        SCHEMA_VERSION, now,
                    ))
        conn.executemany(
            """
            INSERT INTO macro_observations(
                id, observation_date, indicator, value, unit, jurisdiction, source_system_id,
                scenario_tag, row_hash, schema_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            macro_rows,
        )

        scenarios = [
            (
                "scen_base", "Base case", "Current macro conditions", 6,
                to_json({
                    "unemployment_delta": 0.0, "rate_delta": 0.0,
                    "fx_depreciation_pct": 0.0, "sector_shock": 0.0,
                    "horizon_months": 6,
                }),
                "usr_admin", now, "1.0", "active",
            ),
            (
                "scen_downside", "Downside recession", "Unemployment +3pp, rates +150bp", 6,
                to_json({
                    "unemployment_delta": 3.0, "rate_delta": 1.5,
                    "fx_depreciation_pct": 8.0, "sector_shock": 0.45,
                    "horizon_months": 6,
                }),
                "usr_admin", now, "1.0", "active",
            ),
            (
                "scen_sector", "Sector stress", "Receivables concentration shock", 3,
                to_json({
                    "unemployment_delta": 1.2, "rate_delta": 0.4,
                    "fx_depreciation_pct": 3.0, "sector_shock": 0.80,
                    "horizon_months": 3,
                }),
                "usr_analyst", now, "1.0", "active",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO scenarios(id, name, description, horizon_months, parameters_json, created_by, created_at, version, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            scenarios,
        )

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("demo_seeded", now, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("active_rules_version", RULES_VERSION, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("retention_days", "1825", now),
        )

    try:
        run_early_warning_engine("scen_base", "system", "system", force=True)
    except Exception:
        pass


def validate_import_dataframe(
    df: pd.DataFrame,
    entity_type: str,
    conn: sqlite3.Connection,
    options: Dict[str, Any],
) -> Tuple[List[ValidationIssue], Set[int], pd.DataFrame]:
    issues: List[ValidationIssue] = []
    skip: Set[int] = set()

    schema = IMPORT_SCHEMAS.get(entity_type)
    if not schema:
        issues.append(ValidationIssue(0, "entity", entity_type, "Unknown entity type", "critical"))
        return issues, set(df.index), df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in schema["required"] if c not in df.columns]
    if missing_cols:
        issues.append(ValidationIssue(0, "columns", ",".join(missing_cols), "Missing required columns", "critical"))
        return issues, set(df.index), df

    account_refs: Set[str] = set()
    if schema.get("ref_account"):
        account_refs = {str(r["account_ref"]) for r in conn.execute("SELECT account_ref FROM accounts")}

    org_names: Set[str] = set()
    if entity_type == "accounts":
        org_names = {str(r["name"]) for r in conn.execute("SELECT name FROM organizations")}

    unique_cols = schema.get("unique", [])
    if unique_cols:
        dup_mask = df.duplicated(subset=unique_cols, keep="first")
        for idx in df.index[dup_mask]:
            val = str(df.loc[idx, unique_cols[0]]) if unique_cols else ""
            issues.append(ValidationIssue(idx + 2, ",".join(unique_cols), val, "Duplicate key within file", "critical"))
            skip.add(idx)

    today = pd.Timestamp.now(tz="UTC").normalize()

    for idx, row in df.iterrows():
        if idx in skip:
            continue

        null_found = False
        for col in schema["required"]:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                issues.append(ValidationIssue(idx + 2, col, str(val), "Required value missing", "critical"))
                skip.add(idx)
                null_found = True
                break
        if null_found:
            continue

        parsed_dates: Dict[str, pd.Timestamp] = {}
        for col in schema.get("date_cols", []):
            ts = pd.to_datetime(row.get(col), errors="coerce", utc=True)
            if pd.isna(ts):
                issues.append(ValidationIssue(idx + 2, col, str(row.get(col)), "Invalid or unparsable date", "critical"))
                skip.add(idx)
            else:
                parsed_dates[col] = ts
                if ts > today + pd.Timedelta(days=1):
                    issues.append(ValidationIssue(idx + 2, col, str(row.get(col)), "Future-dated record outside allowed range", "critical"))
                    skip.add(idx)

        if idx in skip:
            continue

        if entity_type == "accounts" and "origination_date" in parsed_dates and "maturity_date" in parsed_dates:
            if parsed_dates["maturity_date"] < parsed_dates["origination_date"]:
                issues.append(ValidationIssue(idx + 2, "maturity_date", str(row.get("maturity_date")), "Maturity before origination", "critical"))
                skip.add(idx)

        for col in schema.get("numeric_cols", []):
            num = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(num):
                issues.append(ValidationIssue(idx + 2, col, str(row.get(col)), "Non-numeric value where number required", "critical"))
                skip.add(idx)
            else:
                rng = schema.get("ranges", {}).get(col)
                if rng is not None and (float(num) < rng[0] or float(num) > rng[1]):
                    issues.append(ValidationIssue(idx + 2, col, str(row.get(col)), f"Out of range [{rng[0]}, {rng[1]}]", "critical"))
                    skip.add(idx)

        if idx in skip:
            continue

        for col, allowed in schema.get("enums", {}).items():
            val = str(row.get(col, "")).strip().lower()
            if val and val not in allowed:
                issues.append(ValidationIssue(idx + 2, col, val, f"Value not in allowed set: {allowed}", "critical"))
                skip.add(idx)

        if idx in skip:
            continue

        if "currency" in df.columns:
            cur = str(row.get("currency", "")).strip().upper()
            if cur and not re.fullmatch(r"[A-Z]{3}", cur):
                issues.append(ValidationIssue(idx + 2, "currency", cur, "Currency must be 3-letter ISO-like code", "critical"))
                skip.add(idx)

        if idx in skip:
            continue

        if schema.get("ref_account"):
            ref = str(row.get("account_ref", "")).strip()
            if ref not in account_refs:
                issues.append(ValidationIssue(idx + 2, "account_ref", ref, "Referential integrity failure: account_ref not found", "critical"))
                skip.add(idx)

        if entity_type == "accounts":
            org = str(row.get("org_name", "")).strip()
            ref = str(row.get("account_ref", "")).strip()
            if org not in org_names:
                if options.get("allow_create_orgs"):
                    issues.append(ValidationIssue(idx + 2, "org_name", org, "Organization will be created during import", "warning"))
                else:
                    issues.append(ValidationIssue(idx + 2, "org_name", org, "Organization not found and auto-create disabled", "critical"))
                    skip.add(idx)
            existing = conn.execute("SELECT id FROM accounts WHERE account_ref=?", (ref,)).fetchone()
            if existing:
                issues.append(ValidationIssue(idx + 2, "account_ref", ref, "Account already exists; row skipped to preserve validated source data", "warning"))
                skip.add(idx)

        if entity_type == "arrears" and idx not in skip:
            dpd = pd.to_numeric(row.get("days_past_due"), errors="coerce")
            bucket = str(row.get("bucket", "")).strip().lower()
            if not pd.isna(dpd):
                expected = bucket_for_dpd(int(dpd))
                if bucket and bucket != expected:
                    issues.append(ValidationIssue(idx + 2, "bucket", bucket, f"Bucket contradicts DPD; expected {expected}", "critical"))
                    skip.add(idx)

        if entity_type == "exposures" and idx not in skip:
            drawn = pd.to_numeric(row.get("drawn"), errors="coerce")
            ead = pd.to_numeric(row.get("ead"), errors="coerce")
            if not pd.isna(drawn) and not pd.isna(ead) and float(ead) < float(drawn):
                issues.append(ValidationIssue(idx + 2, "ead", str(row.get("ead")), "EAD lower than drawn amount", "critical"))
                skip.add(idx)

        if entity_type == "repayments" and idx not in skip:
            status = str(row.get("status", "")).strip().lower()
            amount = pd.to_numeric(row.get("amount"), errors="coerce")
            if status == "missed" and not pd.isna(amount) and float(amount) > 0:
                issues.append(ValidationIssue(idx + 2, "amount", str(row.get("amount")), "Missed status with positive amount; treated as warning", "warning"))

    return issues, skip, df


def issues_to_df(issues: List[ValidationIssue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["row", "column", "value", "issue", "severity"])
    return pd.DataFrame([asdict(i) for i in issues])


def read_uploaded_to_df(uploaded_file) -> Tuple[pd.DataFrame, bytes]:
    raw = uploaded_file.getvalue()
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    else:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df, raw


def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename)
    return "".join(ch for ch in base if ch.isalnum() or ch in "._-") or "upload.csv"


def stage_file(
    raw: bytes,
    filename: str,
    entity_type: str,
    file_id: str,
    batch_id: str,
    actor_id: str,
) -> str:
    sha = sha256_bytes(raw)
    safe = sanitize_filename(filename)
    path = os.path.join(UPLOAD_DIR, f"{sha[:16]}_{safe}")
    with open(path, "wb") as f:
        f.write(raw)

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO files(id, filename, sha256, size, uploaded_at, stored_path, import_status, batch_id, entity_type, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, 'staging', ?, ?, ?)
            """,
            (file_id, safe, sha, len(raw), now_utc(), path, batch_id, entity_type, SCHEMA_VERSION),
        )
        conn.execute(
            """
            INSERT INTO batches(id, file_id, imported_at, actor_id, status, notes)
            VALUES (?, ?, ?, ?, 'importing', ?)
            """,
            (batch_id, file_id, now_utc(), actor_id, "Staged import batch"),
        )
    return path


def finalize_file(
    file_id: str,
    batch_id: str,
    status: str,
    rows_total: int,
    rows_valid: int,
    rows_invalid: int,
    error_df: Optional[pd.DataFrame] = None,
) -> None:
    error_path = None
    if error_df is not None and not error_df.empty:
        error_path = os.path.join(ERROR_DIR, f"{file_id}_errors.csv")
        error_df.to_csv(error_path, index=False)

    with db_conn() as conn:
        conn.execute(
            """
            UPDATE files
            SET import_status=?, rows_total=?, rows_valid=?, rows_invalid=?, error_report_path=?
            WHERE id=?
            """,
            (status, rows_total, rows_valid, rows_invalid, error_path, file_id),
        )
        conn.execute(
            "UPDATE batches SET status=?, notes=? WHERE id=?",
            (status, f"Rows total={rows_total}, valid={rows_valid}, invalid={rows_invalid}", batch_id),
        )


def import_dataframe(
    df: pd.DataFrame,
    entity_type: str,
    file_id: str,
    batch_id: str,
    skip_rows: Set[int],
    actor_id: str,
    role: str,
) -> Tuple[int, int]:
    inserted = 0
    rejected = len(skip_rows)

    with db_conn() as conn:
        for idx, row in df.iterrows():
            if idx in skip_rows:
                continue
            try:
                if entity_type == "accounts":
                    org_name = str(row.get("org_name", "")).strip()
                    org = conn.execute("SELECT id FROM organizations WHERE name=?", (org_name,)).fetchone()
                    if not org:
                        org_id = new_id("org")
                        conn.execute(
                            """
                            INSERT INTO organizations(id, name, jurisdiction, currency, sector, created_at, updated_at, source_system_id, schema_version)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                org_id, org_name, str(row.get("jurisdiction", "US")), str(row.get("currency", "USD")),
                                "Imported", now_utc(), now_utc(), "src_manual", SCHEMA_VERSION,
                            ),
                        )
                    else:
                        org_id = str(org["id"])

                    account_ref = str(row.get("account_ref", "")).strip()
                    existing = conn.execute("SELECT id FROM accounts WHERE account_ref=?", (account_ref,)).fetchone()
                    if existing:
                        rejected += 1
                        continue

                    aid = new_id("acc")
                    conn.execute(
                        """
                        INSERT INTO accounts(
                            id, org_id, account_ref, product, facility_type, jurisdiction, currency,
                            limit_amount, balance, origination_date, maturity_date, status,
                            source_system_id, schema_version, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            aid, org_id, account_ref, str(row.get("product", "")).strip(),
                            str(row.get("facility_type", "unsecured")).strip(), str(row.get("jurisdiction", "US")).strip(),
                            str(row.get("currency", "USD")).strip().upper(), as_float(row.get("limit_amount")),
                            as_float(row.get("balance")), as_date_str(row.get("origination_date")),
                            as_date_str(row.get("maturity_date")), str(row.get("status", "active")).strip().lower(),
                            "src_manual", SCHEMA_VERSION, now_utc(), now_utc(),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO consents(id, account_id, consent_type, granted, effective_date, expiry_date, source_system_id, schema_version, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("cns"), aid, "portfolio_monitoring", 1,
                            as_date_str(row.get("origination_date")), as_date_str(row.get("maturity_date")),
                            "src_manual", SCHEMA_VERSION, now_utc(),
                        ),
                    )
                    inserted += 1

                elif entity_type == "repayments":
                    account_ref = str(row.get("account_ref", "")).strip()
                    acc = conn.execute("SELECT id FROM accounts WHERE account_ref=?", (account_ref,)).fetchone()
                    if not acc:
                        rejected += 1
                        continue
                    account_id = str(acc["id"])
                    due_date = as_date_str(row.get("due_date"))
                    payment_date = as_date_str(row.get("payment_date"))
                    amount = as_float(row.get("amount"))
                    row_hash = make_row_hash(["import-rep", account_id, due_date, payment_date, amount, file_id])
                    if conn.execute("SELECT id FROM repayments WHERE row_hash=?", (row_hash,)).fetchone():
                        rejected += 1
                        continue
                    conn.execute(
                        """
                        INSERT INTO repayments(
                            id, account_id, due_date, payment_date, amount, currency, channel, status,
                            days_late, source_file_id, row_number, row_hash, schema_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("rep"), account_id, due_date, payment_date, amount,
                            str(row.get("currency", "USD")).strip().upper(), str(row.get("channel", "unknown")).strip(),
                            str(row.get("status", "on_time")).strip().lower(), as_int(row.get("days_late")) or 0,
                            file_id, int(idx) + 2, row_hash, SCHEMA_VERSION, now_utc(),
                        ),
                    )
                    inserted += 1

                elif entity_type == "exposures":
                    account_ref = str(row.get("account_ref", "")).strip()
                    acc = conn.execute("SELECT id FROM accounts WHERE account_ref=?", (account_ref,)).fetchone()
                    if not acc:
                        rejected += 1
                        continue
                    account_id = str(acc["id"])
                    obs_date = as_date_str(row.get("observation_date"))
                    ead = as_float(row.get("ead"))
                    row_hash = make_row_hash(["import-exp", account_id, obs_date, ead, file_id])
                    if conn.execute("SELECT id FROM exposures WHERE row_hash=?", (row_hash,)).fetchone():
                        rejected += 1
                        continue
                    conn.execute(
                        """
                        INSERT INTO exposures(
                            id, account_id, observation_date, drawn, undrawn, collateral_value, ead,
                            currency, source_file_id, row_number, row_hash, schema_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("exp"), account_id, obs_date, as_float(row.get("drawn")),
                            as_float(row.get("undrawn")), as_float(row.get("collateral_value")), ead,
                            str(row.get("currency", "USD")).strip().upper(), file_id, int(idx) + 2,
                            row_hash, SCHEMA_VERSION, now_utc(),
                        ),
                    )
                    inserted += 1

                elif entity_type == "arrears":
                    account_ref = str(row.get("account_ref", "")).strip()
                    acc = conn.execute("SELECT id FROM accounts WHERE account_ref=?", (account_ref,)).fetchone()
                    if not acc:
                        rejected += 1
                        continue
                    account_id = str(acc["id"])
                    obs_date = as_date_str(row.get("observation_date"))
                    dpd = as_int(row.get("days_past_due")) or 0
                    arrears_balance = as_float(row.get("arrears_balance")) or 0.0
                    row_hash = make_row_hash(["import-arr", account_id, obs_date, dpd, arrears_balance, file_id])
                    if conn.execute("SELECT id FROM arrears WHERE row_hash=?", (row_hash,)).fetchone():
                        rejected += 1
                        continue
                    conn.execute(
                        """
                        INSERT INTO arrears(
                            id, account_id, observation_date, days_past_due, arrears_balance, bucket,
                            cure_flag, source_file_id, row_number, row_hash, schema_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("arr"), account_id, obs_date, dpd, arrears_balance,
                            str(row.get("bucket", bucket_for_dpd(dpd))).strip().lower(),
                            1 if dpd == 0 else 0, file_id, int(idx) + 2, row_hash, SCHEMA_VERSION, now_utc(),
                        ),
                    )
                    inserted += 1

                elif entity_type == "macro":
                    obs_date = as_date_str(row.get("observation_date"))
                    indicator = str(row.get("indicator", "")).strip()
                    jurisdiction = str(row.get("jurisdiction", "US")).strip()
                    scenario_tag = str(row.get("scenario_tag", "base")).strip()
                    value = as_float(row.get("value"))
                    row_hash = make_row_hash(["import-macro", obs_date, indicator, jurisdiction, scenario_tag, value, file_id])
                    if conn.execute("SELECT id FROM macro_observations WHERE row_hash=?", (row_hash,)).fetchone():
                        rejected += 1
                        continue
                    conn.execute(
                        """
                        INSERT INTO macro_observations(
                            id, observation_date, indicator, value, unit, jurisdiction, source_system_id,
                            scenario_tag, row_hash, schema_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("mac"), obs_date, indicator, value, str(row.get("unit", "")).strip(),
                            jurisdiction, "src_manual", scenario_tag, row_hash, SCHEMA_VERSION, now_utc(),
                        ),
                    )
                    inserted += 1

                else:
                    rejected += 1

            except Exception:
                rejected += 1

    return inserted, rejected


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", date_format="iso").encode("utf-8")


def df_to_html_bytes(df: pd.DataFrame) -> bytes:
    return df.to_html(index=False, border=1).encode("utf-8")


def generate_sample_accounts_df() -> pd.DataFrame:
    with db_conn() as conn:
        org = conn.execute("SELECT name FROM organizations LIMIT 1").fetchone()
    org_name = str(org["name"]) if org else f"{SYNTHETIC_TAG} Corp 01"
    today = pd.Timestamp.now(tz="UTC").normalize()
    rows = []
    for i in range(1, 4):
        rows.append({
            "account_ref": f"SAMPLE_ACC_{i:03d}",
            "org_name": org_name,
            "product": "invoice_finance",
            "facility_type": "receivable_backed",
            "jurisdiction": "US",
            "currency": "USD",
            "limit_amount": 250000.0 * i,
            "balance": 120000.0 * i,
            "origination_date": (today - pd.DateOffset(months=12)).date().isoformat(),
            "maturity_date": (today + pd.DateOffset(months=24)).date().isoformat(),
            "status": "active",
        })
    return pd.DataFrame(rows)


def generate_malformed_sample_df() -> pd.DataFrame:
    with db_conn() as conn:
        acc = conn.execute("SELECT account_ref FROM accounts LIMIT 1").fetchone()
    ref = str(acc["account_ref"]) if acc else "DEMO-00001"
    return pd.DataFrame([
        {
            "account_ref": ref,
            "due_date": "2026-01-01",
            "payment_date": "2026-01-02",
            "amount": 1000.0,
            "currency": "USD",
            "status": "on_time",
            "days_late": 0,
        },
        {
            "account_ref": ref,
            "due_date": "2026-01-01",
            "payment_date": "2026-01-02",
            "amount": -50.0,
            "currency": "USD",
            "status": "on_time",
            "days_late": 0,
        },
        {
            "account_ref": "NO_SUCH_ACCOUNT",
            "due_date": "2026-01-01",
            "payment_date": "2026-01-02",
            "amount": 200.0,
            "currency": "USD",
            "status": "late",
            "days_late": 5,
        },
        {
            "account_ref": ref,
            "due_date": "not-a-date",
            "payment_date": "2026-01-02",
            "amount": 300.0,
            "currency": "USD",
            "status": "on_time",
            "days_late": 0,
        },
    ])


def load_reference_caches(force: bool = False) -> None:
    if st.session_state.get("ref_loaded") and not force:
        return

    with db_conn() as conn:
        accounts = pd.read_sql_query(
            """
            SELECT a.*, o.name AS org_name
            FROM accounts a
            LEFT JOIN organizations o ON a.org_id=o.id
            ORDER BY a.account_ref
            """,
            conn,
        )
        scenarios = pd.read_sql_query("SELECT * FROM scenarios ORDER BY name", conn)
        alerts = pd.read_sql_query(
            """
            SELECT al.*, a.account_ref, o.name AS org_name
            FROM alerts al
            LEFT JOIN accounts a ON al.account_id=a.id
            LEFT JOIN organizations o ON a.org_id=o.id
            ORDER BY al.generated_at DESC
            """,
            conn,
        )
        monthly = get_monthly_delinquency_df_conn(conn)
        ead_row = conn.execute(
            """
            SELECT COALESCE(SUM(ead), 0) AS total
            FROM exposures
            WHERE observation_date = (SELECT MAX(observation_date) FROM exposures)
            """
        ).fetchone()

    st.session_state["demo_accounts"] = accounts
    st.session_state["scenarios"] = scenarios
    st.session_state["alerts_df"] = alerts
    st.session_state["monthly_history"] = monthly
    st.session_state["portfolio_ead"] = float(ead_row["total"] if ead_row else 0.0)

    if not alerts.empty:
        active = alerts[~alerts["status"].isin(["archived", "superseded", "exported"])]
        counts = active["severity"].value_counts().to_dict()
    else:
        counts = {}

    st.session_state["live_alert_counts"] = {
        k: int(counts.get(k, 0)) for k in ["high", "medium", "low", "watch", "data_exception"]
    }

    if not st.session_state.get("sim_series"):
        if not monthly.empty:
            last = monthly.iloc[-1]
            st.session_state["sim_series"] = [{
                "timestamp": pd.Timestamp(last["month"]),
                "delinquency_rate": float(last["delinquency_rate"]),
                "ead": st.session_state["portfolio_ead"],
                "high_alerts": st.session_state["live_alert_counts"]["high"],
            }]
        else:
            st.session_state["sim_series"] = [{
                "timestamp": datetime.now(timezone.utc),
                "delinquency_rate": 0.05,
                "ead": st.session_state["portfolio_ead"],
                "high_alerts": 0,
            }]

    st.session_state["ref_loaded"] = True


def get_scenario_params(scenario_id: Optional[str] = None) -> Dict[str, Any]:
    scenario_id = scenario_id or st.session_state.get("active_scenario_id", "scen_base")
    df = st.session_state.get("scenarios", pd.DataFrame())
    if df is None or df.empty:
        return {"horizon_months": 6}
    row = df[df["id"] == scenario_id]
    if row.empty:
        return {"horizon_months": 6}
    params = safe_json_loads(row.iloc[0]["parameters_json"])
    if "horizon_months" not in params:
        params["horizon_months"] = int(row.iloc[0]["horizon_months"] or 6)
    return params


def advance_simulation() -> None:
    clock = st.session_state.get("sim_clock", datetime.now(timezone.utc))
    speed = float(st.session_state.get("sim_speed", 1.0))
    clock = clock + timedelta(minutes=int(15 * speed))
    st.session_state["sim_clock"] = clock

    events = generate_sim_events(clock)
    st.session_state["sim_events"] = (events + st.session_state.get("sim_events", []))[:300]
    update_simulation_state(events, clock)


def generate_sim_events(clock: datetime) -> List[Dict[str, Any]]:
    accounts = st.session_state.get("demo_accounts", pd.DataFrame())
    if accounts.empty:
        return []

    seed = int(clock.timestamp() * 1000) % (2**32)
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    sample = accounts.sample(min(n, len(accounts)), random_state=seed % 100000)
    events: List[Dict[str, Any]] = []

    for _, acct in sample.iterrows():
        kind = str(rng.choice(["payment", "arrears", "exposure", "alert"], p=[0.45, 0.25, 0.15, 0.15]))
        limit = float(acct.get("limit_amount", 100000.0) or 100000.0)

        if kind == "payment":
            amount = round(float(rng.uniform(0.002, 0.03)) * limit, 2)
            events.append({
                "ts": clock.isoformat(), "type": "payment", "account_ref": acct["account_ref"],
                "detail": "On-time payment posted", "amount": amount, "delta": -amount,
            })
        elif kind == "arrears":
            dpd_delta = int(rng.integers(-2, 6))
            amount = round(float(rng.uniform(0.0, 0.01)) * limit, 2)
            events.append({
                "ts": clock.isoformat(), "type": "arrears", "account_ref": acct["account_ref"],
                "detail": f"DPD drift {dpd_delta:+d}", "dpd_delta": dpd_delta, "amount": amount, "delta": amount,
            })
        elif kind == "exposure":
            change = round(float(rng.normal(0.0, 0.005)) * limit, 2)
            events.append({
                "ts": clock.isoformat(), "type": "exposure", "account_ref": acct["account_ref"],
                "detail": "Exposure re-measured", "amount": change, "delta": change,
            })
        else:
            sev = str(rng.choice(["high", "medium", "low", "watch"], p=[0.15, 0.30, 0.30, 0.25]))
            events.append({
                "ts": clock.isoformat(), "type": "alert", "account_ref": acct["account_ref"],
                "detail": f"Score band change: {sev}", "severity": sev, "delta": 1,
            })

    stress = scenario_stress(get_scenario_params())
    macro_value = round(float(np.clip(rng.normal(0.25 + 0.35 * stress, 0.08), 0.0, 1.0)), 4)
    events.append({
        "ts": clock.isoformat(), "type": "macro", "account_ref": "PORTFOLIO",
        "detail": "Synthetic stress index tick", "value": macro_value, "delta": 0,
    })
    return events


def update_simulation_state(events: List[Dict[str, Any]], clock: datetime) -> None:
    series = st.session_state.get("sim_series", [])
    hist = st.session_state.get("monthly_history", pd.DataFrame())

    if not series:
        last_rate = float(hist["delinquency_rate"].iloc[-1]) if not hist.empty else 0.05
        series = [{
            "timestamp": clock - timedelta(minutes=15),
            "delinquency_rate": last_rate,
            "ead": st.session_state.get("portfolio_ead", 0.0),
            "high_alerts": st.session_state.get("live_alert_counts", {}).get("high", 0),
        }]

    last = series[-1]
    rng = np.random.default_rng(int(clock.timestamp() * 1000) % (2**32))

    arrears_delta = sum(float(e.get("amount", 0.0)) for e in events if e.get("type") == "arrears")
    payment_delta = sum(float(e.get("amount", 0.0)) for e in events if e.get("type") == "payment")
    exposure_delta = sum(float(e.get("amount", 0.0)) for e in events if e.get("type") == "exposure")

    stress = scenario_stress(get_scenario_params())
    base_ead = max(float(st.session_state.get("portfolio_ead", 1e6) or 1e6), 1e6)
    rate_shock = (arrears_delta / base_ead) * 0.8 + (stress - 0.2) * 0.0015 + float(rng.normal(0.0, 0.0008))
    new_rate = float(np.clip(float(last.get("delinquency_rate", 0.05)) + rate_shock, 0.0, 0.65))
    new_ead = float(max(0.0, float(last.get("ead", base_ead)) + exposure_delta - payment_delta * 0.2))

    counts = dict(st.session_state.get("live_alert_counts", {}))
    for k in ["high", "medium", "low", "watch", "data_exception"]:
        counts.setdefault(k, 0)

    for e in events:
        if e.get("type") == "alert":
            sev = e.get("severity", "watch")
            counts[sev] = counts.get(sev, 0) + 1
        if e.get("type") == "payment":
            counts["watch"] = max(0, counts.get("watch", 0) - 1)

    st.session_state["live_alert_counts"] = {k: int(max(0, v)) for k, v in counts.items()}
    series.append({
        "timestamp": clock,
        "delinquency_rate": new_rate,
        "ead": new_ead,
        "high_alerts": int(counts.get("high", 0)),
    })
    st.session_state["sim_series"] = series[-240:]
    st.session_state["portfolio_ead"] = new_ead


def load_font(size: int):
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def generate_demo_video(path: str, seconds: int = 5, fps: int = 12, width: int = 640, height: int = 360) -> None:
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    font_large = load_font(22)
    font = load_font(14)
    small = load_font(11)
    total = int(seconds * fps)

    for i in range(total):
        t = i / float(fps)
        img = Image.new("RGB", (width, height), (11, 17, 32))
        d = ImageDraw.Draw(img)

        d.rectangle([0, 0, width, 50], fill=(18, 28, 52))
        d.text((18, 14), "Credit Portfolio Early-Warning Demo", font=font_large, fill=(235, 240, 255))
        d.text((18, 58), f"Synthetic data | Decision support | t={t:.1f}s", font=font, fill=(155, 175, 205))

        cards = [
            ("EAD", f"${120 + 10 * math.sin(t):.1f}M"),
            ("30+ DPD", f"{5 + 2 * math.sin(t * 1.3):.1f}%"),
            ("High alerts", f"{int(4 + 3 * abs(math.sin(t * 0.8)))}"),
        ]
        x = 20
        for label, val in cards:
            d.rectangle([x, 90, x + 180, 150], fill=(24, 36, 62), outline=(70, 95, 140))
            d.text((x + 12, 100), label, font=small, fill=(150, 170, 200))
            d.text((x + 12, 118), val, font=font, fill=(240, 245, 255))
            x += 200

        d.rectangle([20, 170, 420, 330], outline=(60, 80, 115), fill=(16, 24, 42))
        d.text((28, 176), "Delinquency trend + scenario forecast", font=small, fill=(170, 190, 220))

        pts = []
        for j in range(60):
            tt = j / 59.0
            xx = 30 + tt * 380
            yy = 310 - (80 * (0.35 + 0.18 * math.sin(tt * 6.0 + t * 0.8) + 0.08 * tt))
            pts.append((xx, yy))
        d.line(pts, fill=(86, 180, 230), width=2)

        pts2 = []
        for j in range(60):
            tt = j / 59.0
            xx = 30 + tt * 380
            yy = 310 - (80 * (0.45 + 0.20 * math.sin(tt * 6.0 + t * 0.8) + 0.12 * tt))
            pts2.append((xx, yy))
        for k in range(0, len(pts2) - 1, 2):
            d.line([pts2[k], pts2[k + 1]], fill=(240, 170, 70), width=2)

        d.rectangle([440, 170, 620, 330], outline=(60, 80, 115), fill=(16, 24, 42))
        d.text((448, 176), "Alert severity", font=small, fill=(170, 190, 220))
        vals = [
            0.55 + 0.15 * math.sin(t * 1.1),
            0.25 + 0.10 * math.cos(t * 1.4),
            0.12 + 0.05 * math.sin(t * 2.0),
        ]
        colors = [(230, 90, 90), (240, 180, 76), (90, 190, 140)]
        bx = 460
        for v, c in zip(vals, colors):
            bh = int(max(0.0, min(1.0, v)) * 120)
            d.rectangle([bx, 310 - bh, bx + 40, 310], fill=c)
            bx += 55

        d.text((20, 338), "Not automated credit denial. Human review required.", font=small, fill=(130, 150, 180))
        writer.append_data(np.array(img))

    writer.close()


@st.cache_resource(show_spinner=False)
def get_demo_video_path() -> Tuple[str, str]:
    path = os.path.join(DEMO_DIR, "credit_early_warning_demo.mp4")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return ("ok", path)
    if not VIDEO_DEPS:
        return ("unavailable", "Video dependencies unavailable. Install requirements.txt; the live dashboard still works.")
    try:
        generate_demo_video(path)
        return ("ok", path)
    except Exception as exc:
        return ("error", f"Video generation failed: {exc}. The live dashboard still works.")


def transition_allowed(current: str, new: str) -> bool:
    return new in ALLOWED_TRANSITIONS.get(current, [])


def transition_alert(
    alert_id: str,
    new_status: str,
    action: str,
    reason: str,
    override_score: Optional[float] = None,
    override_severity: Optional[str] = None,
) -> str:
    permission_map = {
        "submit_review": "submit_review",
        "approve": "approve",
        "override": "override",
        "archive": "archive",
        "correct": "import",
        "export": "export",
    }
    perm = permission_map.get(action, "view")
    if not can(perm):
        raise PermissionError(f"Role {st.session_state.get('role')} lacks permission {perm}")

    actor_id = st.session_state.get("actor_id", "system")
    role = st.session_state.get("role", "system")

    with db_conn() as conn:
        alert = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        if not alert:
            raise DomainValidationError("Alert not found")

        current = str(alert["status"])
        if not transition_allowed(current, new_status):
            raise DomainValidationError(f"Transition {current} -> {new_status} is not allowed")

        data_quality = safe_json_loads(alert["data_quality_json"])
        if not isinstance(data_quality, list):
            data_quality = []
        critical = any(str(q).startswith("CRITICAL") for q in data_quality)

        if new_status == "approved" and critical and action != "override":
            raise DomainValidationError("Cannot approve alert with CRITICAL data exception without reviewer override")

        factors = safe_json_loads(alert["factors_json"])
        if not isinstance(factors, dict):
            factors = {}

        score = float(alert["score"] or 0.0)
        severity = str(alert["severity"] or "watch")

        if override_score is not None:
            factors["original_score"] = score
            factors["override_score"] = float(override_score)
            score = float(override_score)
        if override_severity:
            factors["override_severity"] = override_severity
            severity = override_severity
        factors["override_reason"] = reason

        approval_at = alert["approval_at"]
        archive_at = alert["archive_at"]
        export_batch_id = alert["export_batch_id"]
        reviewer_id = alert["reviewer_id"]

        if new_status == "approved":
            approval_at = now_utc()
            reviewer_id = actor_id
        elif new_status == "archived":
            archive_at = now_utc()
        elif new_status == "exported":
            export_batch_id = new_id("expbatch")

        conn.execute(
            """
            UPDATE alerts
            SET status=?, severity=?, score=?, factors_json=?, reviewer_id=?, approval_at=?, archive_at=?, export_batch_id=?
            WHERE id=?
            """,
            (new_status, severity, score, to_json(factors), reviewer_id, approval_at, archive_at, export_batch_id, alert_id),
        )

        audit_event_id = log_audit_conn(
            conn, f"alert_{action}", "alert", alert_id,
            {"from": current, "to": new_status, "reason": reason, "override_score": override_score, "override_severity": override_severity},
            actor_id, role,
        )
        decision_id = new_id("dec")
        conn.execute(
            """
            INSERT INTO decisions(id, alert_id, actor_id, role, action, reason, override_score, override_severity, created_at, audit_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (decision_id, alert_id, actor_id, role, action, reason, override_score, override_severity, now_utc(), audit_event_id),
        )
        return decision_id


def render_sidebar() -> None:
    st.sidebar.title(APP_TITLE)
    current_page = st.session_state.get("active_page", "Live Monitor")
    if current_page not in PAGES:
        current_page = "Live Monitor"
    page = st.sidebar.selectbox("Workspace", PAGES, index=PAGES.index(current_page))
    st.session_state["active_page"] = page

    st.sidebar.markdown("---")
    role_key = st.sidebar.selectbox(
        "Local role (demo authorization)",
        list(ROLE_LABELS.keys()),
        index=list(ROLE_LABELS.keys()).index(st.session_state.get("role", "analyst_operator")),
        format_func=lambda x: ROLE_LABELS[x],
    )
    st.session_state["role"] = role_key
    st.session_state["actor_id"] = ROLE_ACTORS[role_key]
    st.sidebar.markdown(f"**Permissions:** `{', '.join(sorted(ROLE_PERMISSIONS[role_key]))}`")

    st.sidebar.markdown("---")
    st.sidebar.header("Live simulation")
    st.session_state["sim_running"] = st.sidebar.checkbox(
        "Auto-run simulation",
        value=bool(st.session_state.get("sim_running", True)),
    )
    st.session_state["sim_speed"] = st.sidebar.slider(
        "Speed multiplier",
        0.5,
        5.0,
        float(st.session_state.get("sim_speed", 1.0)),
        0.5,
    )

    scen_df = st.session_state.get("scenarios", pd.DataFrame())
    if scen_df is not None and not scen_df.empty:
        options = dict(zip(scen_df["id"], scen_df["name"]))
        active_id = st.session_state.get("active_scenario_id", "scen_base")
        if active_id not in options:
            active_id = list(options.keys())[0]
        st.session_state["active_scenario_id"] = st.sidebar.selectbox(
            "Active scenario",
            list(options.keys()),
            index=list(options.keys()).index(active_id),
            format_func=lambda x: options[x],
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Synthetic data only. No paid APIs. Decision support, not automated denial.")


def render_kpi_row() -> None:
    counts = st.session_state.get("live_alert_counts", {})
    ead = float(st.session_state.get("portfolio_ead", 0.0))
    series = st.session_state.get("sim_series", [])
    current_rate = float(series[-1]["delinquency_rate"]) if series else 0.0
    data_exceptions = int(counts.get("data_exception", 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio EAD", f"${ead:,.0f}")
    c2.metric("Live 30+ proxy rate", f"{current_rate:.2%}")
    c3.metric("High alerts", int(counts.get("high", 0)))
    c4.metric("Medium alerts", int(counts.get("medium", 0)))
    c5.metric("Data exceptions", data_exceptions)


def render_live_charts() -> None:
    hist = st.session_state.get("monthly_history", pd.DataFrame())
    series = st.session_state.get("sim_series", [])
    params = get_scenario_params()
    fc = forecast_delinquency(hist, int(params.get("horizon_months", 6)), params)

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["month"], y=hist["delinquency_rate"],
            name="Historical monthly", mode="lines+markers",
        ))
    if series:
        sim_df = pd.DataFrame(series)
        fig.add_trace(go.Scatter(
            x=sim_df["timestamp"], y=sim_df["delinquency_rate"],
            name="Live simulation", mode="lines",
        ))
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["month"], y=fc["upper"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["month"], y=fc["lower"], fill="tonexty",
            mode="lines", line=dict(width=0), name="Forecast uncertainty",
            fillcolor="rgba(240,170,70,0.16)", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["month"], y=fc["scenario"], mode="lines",
            name="Scenario forecast", line=dict(dash="dash", color="rgb(240,170,70)"),
        ))

    fig.update_layout(
        title="Portfolio delinquency: history, live simulation, scenario forecast",
        xaxis_title="Time",
        yaxis_title="Arrears / EAD",
        yaxis_tickformat=".1%",
        height=360,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    counts = st.session_state.get("live_alert_counts", {})
    sev_df = pd.DataFrame({"Severity": list(counts.keys()), "Count": list(counts.values())})
    fig2 = px.bar(sev_df, x="Severity", y="Count", color="Severity", title="Active alert severity mix")
    fig2.update_layout(height=280, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig2, use_container_width=True)


def render_live_table() -> None:
    alerts = st.session_state.get("alerts_df", pd.DataFrame())
    if alerts.empty:
        st.info("No alerts loaded yet. Run the early-warning engine from the sidebar workflow.")
        return

    active = alerts[~alerts["status"].isin(["archived", "superseded", "exported"])].copy()
    active = active.sort_values("score", ascending=False)
    cols = ["account_ref", "org_name", "severity", "score", "status", "generated_at"]
    st.subheader("Top active alerts")
    st.dataframe(active[cols].head(12), use_container_width=True, hide_index=True)

    events = st.session_state.get("sim_events", [])
    if events:
        ev_df = pd.DataFrame(events).head(20)
        st.subheader("Live event feed")
        st.dataframe(ev_df[["ts", "type", "account_ref", "detail"]], use_container_width=True, hide_index=True)


def render_live_body() -> None:
    if st.session_state.get("sim_running", False):
        advance_simulation()
    render_kpi_row()
    render_live_charts()
    render_live_table()


def render_demo_video_section() -> None:
    with st.expander("Generated live demo video (MP4)", expanded=True):
        status, payload = get_demo_video_path()
        if status == "ok":
            try:
                with open(payload, "rb") as f:
                    data = f.read()
                st.video(data, format="video/mp4")
                st.download_button(
                    "Download demo MP4",
                    data=data,
                    file_name="credit_early_warning_demo.mp4",
                    mime="video/mp4",
                )
            except Exception as exc:
                st.warning(f"Could not display video: {exc}")
        else:
            st.warning(payload)


def render_live_monitor() -> None:
    st.title("Live Portfolio Monitor")
    st.caption(
        "This screen starts automatically with synthetic demo data and a running simulation. "
        "It is an early-warning workbench for monitoring, not an automated credit denial engine."
    )

    if hasattr(st, "fragment") and st.session_state.get("sim_running", False):
        st.fragment(render_live_body, run_every=1.0)()
    else:
        render_live_body()
        if st.button("Refresh live view"):
            st.rerun()

    render_demo_video_section()


def render_product_brief() -> None:
    st.title("Product Definition, Users, and Acceptance Criteria")
    st.markdown("""
### One-page product brief
**Problem.** Credit, treasury, and finance operations teams monitor repayment history, exposure, arrears, and macro conditions across fragmented spreadsheets, core-system extracts, and email reviews. The current workflow is slow, inconsistent, and difficult to audit. Early warning is often reactive: arrears are noticed after they mature into losses, and explanations are tribal knowledge.

**Product.** A local-first early-warning workbench that combines repayment history, exposure, arrears, and macroeconomic scenarios into explainable portfolio monitoring alerts. It supports human review, correction, approval, and export. It does not make automated credit denial decisions.

**Initial narrow workflow.** Import account, repayment, exposure, arrears, and macro data; validate rows; calculate deterministic early-warning scores; simulate scenarios; review and approve alerts; export an auditable report.

**Measurable target improvements for pilot validation.**
- Reduce manual monitoring preparation time by 30% or more, measured against a documented baseline.
- Achieve 100% traceable explanation coverage for every generated alert.
- Reject malformed rows with actionable row-level errors.
- Produce reproducible calculations for the same input and rules version.
- Pass automated unit, integration, regression, and API-less workflow tests.

**Do not claim customer impact until tested with real users.**
""")

    st.subheader("User personas and permissions")
    st.markdown("""
| Persona | Primary need | Permissions |
|---|---|---|
| Administrator | Configure sources, roles, retention, backups, and releases | view, import, run_engine, approve, override, archive, export, manage_config, backup_restore, delete, manage_scenarios |
| Analyst / Operator | Load data, correct exceptions, run monitoring, prepare reviews | view, import, run_engine, submit_review, export, annotate, manage_scenarios |
| Reviewer | Validate alerts, approve, override with reason, sign off | view, run_engine, approve, override, archive, export, signoff, manage_scenarios |
| Read-only Stakeholder | Monitor portfolio status and consume reports | view, export_summary |
""")

    st.subheader("Record lifecycle")
    st.markdown("""
1. **Creation:** record enters via CSV/XLSX import, manual correction, or simulation feed.
2. **Validation:** schema, type, range, date, duplicate, referential, and contradiction checks.
3. **Exception path:** missing, contradictory, late, or out-of-range data is routed to correction queue.
4. **Review:** analyst submits, reviewer inspects explanations and data quality.
5. **Approval / override:** reviewer approves or overrides with reason; audit event created.
6. **Correction:** corrected observations are appended; original source data is preserved.
7. **Archival / export:** approved or archived records can be exported with lineage.
""")

    st.subheader("Workflow diagram")
    st.graphviz_chart("""
digraph Workflow {
  rankdir=LR;
  node [shape=box, style=filled, fillcolor="#eef4ff"];
  Source [label="Source systems\\nCSV / XLSX / manual"];
  Ingest [label="Ingestion & preview"];
  Validate [label="Validation engine"];
  Exception [label="Exception queue\\nmissing / late / contradictory"];
  Analyst [label="Analyst correction"];
  Engine [label="Deterministic early-warning engine"];
  Scenario [label="Macro scenario simulation"];
  Review [label="Reviewer queue"];
  Approve [label="Approved monitoring action"];
  Archive [label="Archive / export"];
  Audit [label="Audit trail"];

  Source -> Ingest -> Validate;
  Validate -> Engine [label="valid"];
  Validate -> Exception [label="errors"];
  Exception -> Analyst -> Validate;
  Engine -> Scenario -> Review;
  Review -> Approve [label="approve / override"];
  Review -> Exception [label="send back"];
  Approve -> Archive;
  Archive -> Audit;
  Engine -> Audit;
  Review -> Audit;
}
""")

    st.subheader("Prioritized backlog")
    st.markdown("""
| Tier | Capability | Why |
|---|---|---|
| MVP | SQLite local storage, CSV/XLSX import, validation, deterministic scoring, scenario forecast, review queue, explanations, exports, audit, self-tests | Solve one narrow workflow end-to-end on Windows |
| Post-MVP | PostgreSQL migration, concurrent multi-user roles, real authentication, scheduled imports, API adapters, advanced model registry | Team deployment and integration |
| Enterprise | SSO/directory integration, deployment controls, SLA-backed support, regulated audit exports, data loss prevention, qualified domain review | Production governance and scale |
""")

    st.subheader("Measurable acceptance checklist")
    st.markdown("""
| Acceptance item | Pass condition | Fail condition |
|---|---|---|
| Sample import | Documented sample CSV imports with row count and audit event | Import crashes or silently drops rows |
| Malformed rejection | Bad rows produce row-level actionable errors and are skipped | Invalid values enter database silently |
| Reproducible calculation | Same input + same rules version yields same score | Results vary without version change |
| Traceable explanation | Every alert shows inputs, weights, data quality, uncertainty, reviewer status | Alert has no explanation |
| Export | Report downloads as CSV/JSON/HTML and audit records export | Export missing or untraceable |
| Automated tests | Self-test page reports PASS for core suites | Any FAIL or ERROR in required tests |
| Human approval | Reviewer can approve/override only with reason and audit | Approval without audit or reason |
""")

    st.subheader("Risk register")
    st.markdown("""
| Risk | Domain | Mitigation |
|---|---|---|
| Rule weights are not clinically/economically valid | Domain validity | Qualified practitioner review, versioning, baseline comparison, no real-world claims on synthetic data |
| Poor source data quality | Data quality | Row-level validation, exception queue, lineage, preservation of originals |
| Sensitive account data exposure | Security | Local synthetic demo, no paid APIs, audit, retention, production auth/encryption required before real data |
| Users distrust black-box alerts | Adoption | Explainability panel, manual override, human approval, no automated denial |
| Operational failure during import | Operations | Transactions, staged files, error reports, backup/restore, self-tests |
""")

    st.subheader("Commercialization outline")
    st.markdown("""
**Initial buyer persona:** Treasury or credit risk operations lead at a lender or fintech managing a mid-sized receivables/loan portfolio.

**Expensive workflow improved:** manual aggregation of repayment, exposure, arrears, and macro indicators into weekly monitoring decks.

**Revenue hypotheses:** per-seat subscription for team edition, private on-premises license for regulated lenders, usage-based processing for payment companies, annual support and integration services.

**Release tiers:**
1. Local MVP for one operator.
2. Team edition with shared PostgreSQL database, roles, approvals.
3. Enterprise edition with SSO, audit exports, deployment controls, APIs, SLAs.

**Pilot success metric:** measured reduction in monitoring preparation time and 100% traceable alert explanations, validated with real users before claiming impact.
""")


def render_data_import() -> None:
    st.title("Data Import, Validation, and Governance")
    st.caption("CSV and XLSX first. Original files are preserved. Validated source data is not silently overwritten.")

    tab_upload, tab_samples, tab_docs = st.tabs(["Upload", "Sample files", "Import rules"])

    with tab_upload:
        uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
        entity = st.selectbox("Entity type", list(IMPORT_SCHEMAS.keys()))
        action = st.radio("Action", ["Preview and validate only", "Import valid rows"], horizontal=True)
        allow_create_orgs = st.checkbox("Create missing organizations for accounts import", value=False)

        if uploaded is not None and st.button("Validate / Import"):
            try:
                df, raw = read_uploaded_to_df(uploaded)
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
                return

            if df.empty:
                st.error("File is empty.")
                return

            with db_conn() as conn:
                issues, skip_rows, cleaned = validate_import_dataframe(
                    df, entity, conn, {"allow_create_orgs": allow_create_orgs}
                )

            issues_df = issues_to_df(issues)
            st.subheader("Validation preview")
            st.dataframe(df.head(20), use_container_width=True)
            st.subheader("Row-level issues")
            st.dataframe(issues_df, use_container_width=True, hide_index=True)

            structural = any(i.row == 0 and i.column == "columns" for i in issues)
            critical_count = sum(1 for i in issues if i.severity == "critical")

            if action == "Import valid rows":
                if structural:
                    file_id = new_id("file")
                    batch_id = new_id("batch")
                    stage_file(raw, uploaded.name, entity, file_id, batch_id, st.session_state.get("actor_id", "system"))
                    finalize_file(file_id, batch_id, "failed", len(df), 0, len(df), issues_df)
                    log_audit("import_failed_structural", "file", file_id, {"entity": entity})
                    st.error("Structural column failure. File stored for audit; no rows imported.")
                    st.download_button("Download error report", df_to_csv_bytes(issues_df), f"{file_id}_errors.csv", "text/csv")
                    return

                file_id = new_id("file")
                batch_id = new_id("batch")
                actor_id = st.session_state.get("actor_id", "system")
                role = st.session_state.get("role", "system")
                stage_file(raw, uploaded.name, entity, file_id, batch_id, actor_id)
                inserted, rejected = import_dataframe(cleaned, entity, file_id, batch_id, skip_rows, actor_id, role)
                status = "success" if critical_count == 0 else "partial"
                finalize_file(file_id, batch_id, status, len(df), inserted, rejected, issues_df)
                log_audit("import_completed", "file", file_id, {
                    "entity": entity, "inserted": inserted, "rejected": rejected, "status": status
                }, actor_id, role)
                load_reference_caches(force=True)
                st.success(f"Import finished: {inserted} rows inserted, {rejected} rows rejected/skipped.")
                st.download_button("Download error report", df_to_csv_bytes(issues_df), f"{file_id}_errors.csv", "text/csv")

    with tab_samples:
        st.subheader("Documented sample files")
        sample_acc = generate_sample_accounts_df()
        st.dataframe(sample_acc, use_container_width=True)
        st.download_button("Download sample_accounts.csv", df_to_csv_bytes(sample_acc), "sample_accounts.csv", "text/csv")

        malformed = generate_malformed_sample_df()
        st.subheader("Malformed sample for rejection testing")
        st.dataframe(malformed, use_container_width=True)
        st.download_button("Download malformed_repayments.csv", df_to_csv_bytes(malformed), "malformed_repayments.csv", "text/csv")

    with tab_docs:
        st.markdown("""
### Import governance rules
- Every file receives a SHA-256 checksum and is stored outside relational rows.
- Database stores metadata, checksum, row counts, status, and error report path.
- Re-import of the same checksum is blocked by default to preserve validated source data.
- Corrections should be submitted as new files or manual correction records.
- Critical row errors skip the row; warnings may be imported with audit trace.
- Referential integrity is enforced for account-linked observations.
- Future-dated, negative, duplicate, unparsable, and contradictory records are rejected or flagged.
""")


def render_portfolio_records() -> None:
    st.title("Portfolio Records")
    accounts = st.session_state.get("demo_accounts", pd.DataFrame())
    if accounts.empty:
        st.info("No accounts loaded.")
        return

    org_filter = st.multiselect("Organization", sorted(accounts["org_name"].dropna().unique().tolist()))
    product_filter = st.multiselect("Product", sorted(accounts["product"].dropna().unique().tolist()))
    status_filter = st.multiselect("Status", sorted(accounts["status"].dropna().unique().tolist()))

    df = accounts.copy()
    if org_filter:
        df = df[df["org_name"].isin(org_filter)]
    if product_filter:
        df = df[df["product"].isin(product_filter)]
    if status_filter:
        df = df[df["status"].isin(status_filter)]

    st.dataframe(
        df[[
            "account_ref", "org_name", "product", "jurisdiction", "currency",
            "limit_amount", "balance", "status", "origination_date", "maturity_date"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Record detail and manual correction")
    ref = st.selectbox("Select account", df["account_ref"].tolist())
    acct_row = df[df["account_ref"] == ref].iloc[0]

    with db_conn() as conn:
        latest_exp = conn.execute(
            "SELECT * FROM exposures WHERE account_id=? ORDER BY observation_date DESC LIMIT 1",
            (acct_row["id"],),
        ).fetchone()
        latest_arr = conn.execute(
            "SELECT * FROM arrears WHERE account_id=? ORDER BY observation_date DESC LIMIT 1",
            (acct_row["id"],),
        ).fetchone()

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Latest exposure**")
        if latest_exp:
            st.json({k: latest_exp[k] for k in latest_exp.keys()})
        else:
            st.warning("No exposure observation.")
    with c2:
        st.write("**Latest arrears**")
        if latest_arr:
            st.json({k: latest_arr[k] for k in latest_arr.keys()})
        else:
            st.warning("No arrears observation.")

    if can("import"):
        with st.form("manual_correction_form"):
            st.subheader("Append manual correction observation")
            new_ead = st.number_input("New EAD", min_value=0.0, value=float(latest_exp["ead"]) if latest_exp else 0.0, step=1000.0)
            new_dpd = st.number_input("New days past due", min_value=0, max_value=3650, value=int(latest_arr["days_past_due"]) if latest_arr else 0, step=1)
            new_arrears = st.number_input("New arrears balance", min_value=0.0, value=float(latest_arr["arrears_balance"]) if latest_arr else 0.0, step=100.0)
            reason = st.text_area("Correction reason", "Manual review correction")
            submitted = st.form_submit_button("Submit correction")

            if submitted:
                if new_arrears > new_ead * 1.1 and new_ead > 0:
                    st.error("Contradictory correction: arrears balance materially exceeds EAD.")
                else:
                    obs_date = pd.Timestamp.now(tz="UTC").normalize().date().isoformat()
                    with db_conn() as conn:
                        conn.execute(
                            """
                            INSERT INTO exposures(
                                id, account_id, observation_date, drawn, undrawn, collateral_value, ead,
                                currency, source_file_id, row_number, row_hash, schema_version, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_id("exp"), acct_row["id"], obs_date, new_ead, 0.0, 0.0, new_ead,
                                acct_row["currency"] or "USD", None, None, make_row_hash(["manual-exp", acct_row["id"], obs_date, new_ead, uuid.uuid4().hex]),
                                SCHEMA_VERSION, now_utc(),
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO arrears(
                                id, account_id, observation_date, days_past_due, arrears_balance, bucket,
                                cure_flag, source_file_id, row_number, row_hash, schema_version, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_id("arr"), acct_row["id"], obs_date, int(new_dpd), float(new_arrears),
                                bucket_for_dpd(int(new_dpd)), 1 if int(new_dpd) == 0 else 0, None, None,
                                make_row_hash(["manual-arr", acct_row["id"], obs_date, int(new_dpd), float(new_arrears), uuid.uuid4().hex]),
                                SCHEMA_VERSION, now_utc(),
                            ),
                        )
                        conn.execute(
                            """
                            UPDATE alerts
                            SET status='corrected'
                            WHERE account_id=? AND status NOT IN ('archived','exported','superseded')
                            """,
                            (acct_row["id"],),
                        )
                    log_audit("manual_correction", "account", acct_row["id"], {
                        "reason": reason, "new_ead": new_ead, "new_dpd": new_dpd, "new_arrears": new_arrears
                    })
                    load_reference_caches(force=True)
                    st.success("Correction appended. Original source data preserved.")


def render_engine_page() -> None:
    st.title("Deterministic Early-Warning Engine")
    st.caption("Business rules and formulas run before machine learning. Outputs are explainable and versioned.")

    params = get_scenario_params()
    rules_version = get_active_rules_version()
    weights = RULE_SETS.get(rules_version, RULE_SETS[RULES_VERSION])

    st.subheader("Active formula")
    st.markdown(f"""
**Rules version:** `{rules_version}`

Score = 100 × (
w₁ × DPD risk +
w₂ × arrears ratio +
w₃ × utilization risk +
w₄ × payment risk +
w₅ × macro stress
)

Where:
- DPD risk = min(max(DPD, 0) / 90, 1)
- Arrears ratio = min(max(arrears balance, 0) / max(EAD, 1), 1)
- Utilization risk = min(max(balance, 0) / max(limit, 1), 1)
- Payment risk = 1 − on-time payment rate over trailing 365 days
- Macro stress = normalized scenario stress in [0, 1]

Weights: `{to_json(weights)}`
""")

    st.subheader("Hand-checkable example")
    example_components = {
        "dpd_risk": 45 / 90.0,
        "arrears_risk": 10000.0 / 90000.0,
        "utilization_risk": 80000.0 / 100000.0,
        "payment_risk": 1.0 - 0.80,
        "macro_stress": 0.30,
    }
    example_score = compute_weighted_score(example_components, weights)
    st.metric("Example score", f"{example_score:.2f}")
    st.json({
        "inputs": {
            "limit": 100000,
            "balance": 80000,
            "ead": 90000,
            "dpd": 45,
            "arrears_balance": 10000,
            "on_time_rate": 0.80,
            "macro_stress": 0.30,
        },
        "components": example_components,
        "weighted_contributions": {k: example_components[k] * weights[k] for k in weights},
        "score": example_score,
    })

    if can("run_engine"):
        if st.button("Run engine for active scenario"):
            try:
                df = run_early_warning_engine(st.session_state.get("active_scenario_id", "scen_base"))
                load_reference_caches(force=True)
                st.success(f"Engine run complete. {len(df)} account metrics calculated.")
                st.dataframe(df[[
                    "account_ref", "score", "severity", "ead", "dpd", "arrears_balance",
                    "on_time_rate", "macro_stress", "uncertainty"
                ]].head(50), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Engine failed: {exc}")
    else:
        st.info("Current role cannot run the engine.")

    alerts = st.session_state.get("alerts_df", pd.DataFrame())
    if not alerts.empty:
        st.subheader("Generated alerts")
        st.dataframe(alerts[[
            "account_ref", "org_name", "severity", "score", "status", "rules_version", "generated_at"
        ]].head(50), use_container_width=True, hide_index=True)


def render_scenarios_page() -> None:
    st.title("Macroeconomic Scenarios and Forecast Simulation")
    params = get_scenario_params()
    stress = scenario_stress(params)
    st.metric("Scenario stress index", f"{stress:.3f}")

    hist = st.session_state.get("monthly_history", pd.DataFrame())
    fc = forecast_delinquency(hist, int(params.get("horizon_months", 6)), params)

    if fc.empty:
        st.info("Insufficient history for forecast.")
    else:
        fig = go.Figure()
        if not hist.empty:
            fig.add_trace(go.Scatter(x=hist["month"], y=hist["delinquency_rate"], name="Historical", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=fc["month"], y=fc["base"], name="Base forecast", mode="lines"))
        fig.add_trace(go.Scatter(x=fc["month"], y=fc["scenario"], name="Scenario forecast", mode="lines", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=fc["month"], y=fc["upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["month"], y=fc["lower"], fill="tonexty", mode="lines", line=dict(width=0), name="Uncertainty", fillcolor="rgba(240,170,70,0.16)"))
        fig.update_layout(title="Delinquency forecast with uncertainty band", yaxis_tickformat=".1%", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fc, use_container_width=True, hide_index=True)

    if can("manage_scenarios"):
        with st.form("create_scenario_form"):
            st.subheader("Create scenario")
            name = st.text_input("Scenario name", "Custom stress")
            description = st.text_area("Description", "User-defined scenario")
            horizon = st.number_input("Horizon months", 1, 24, 6)
            unemployment_delta = st.number_input("Unemployment delta (pp)", 0.0, 10.0, 1.5, 0.1)
            rate_delta = st.number_input("Rate delta (%)", 0.0, 10.0, 0.75, 0.05)
            fx_depreciation = st.number_input("FX depreciation (%)", 0.0, 50.0, 5.0, 0.5)
            sector_shock = st.slider("Sector shock", 0.0, 1.0, 0.35, 0.05)
            submitted = st.form_submit_button("Create scenario")

            if submitted:
                scen_id = new_id("scen")
                payload = {
                    "unemployment_delta": float(unemployment_delta),
                    "rate_delta": float(rate_delta),
                    "fx_depreciation_pct": float(fx_depreciation),
                    "sector_shock": float(sector_shock),
                    "horizon_months": int(horizon),
                }
                with db_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO scenarios(id, name, description, horizon_months, parameters_json, created_by, created_at, version, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (scen_id, name, description, int(horizon), to_json(payload), st.session_state.get("actor_id", "system"), now_utc(), "1.0", "active"),
                    )
                log_audit("scenario_created", "scenario", scen_id, payload)
                load_reference_caches(force=True)
                st.success("Scenario created.")


def render_review_page() -> None:
    st.title("Review, Approval, Override, and Exception Handling")
    alerts = st.session_state.get("alerts_df", pd.DataFrame())
    if alerts.empty:
        st.info("No alerts available.")
        return

    status_options = sorted(alerts["status"].dropna().unique().tolist())
    chosen_statuses = st.multiselect("Status filter", status_options, default=["generated", "under_review", "needs_validation"])
    queue = alerts[alerts["status"].isin(chosen_statuses)].copy()
    queue = queue.sort_values("score", ascending=False)

    st.dataframe(queue[[
        "account_ref", "org_name", "severity", "score", "status", "generated_at"
    ]].head(50), use_container_width=True, hide_index=True)

    if queue.empty:
        return

    alert_id = st.selectbox("Select alert", queue["id"].tolist(), format_func=lambda x: x)
    alert = queue[queue["id"] == alert_id].iloc[0]

    st.subheader("Alert explanation")
    factors = safe_json_loads(alert["factors_json"])
    data_quality = safe_json_loads(alert["data_quality_json"])
    uncertainty = safe_json_loads(alert["uncertainty_json"])

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Score", f"{float(alert['score']):.2f}")
        st.metric("Severity", alert["severity"])
        st.metric("Status", alert["status"])
    with c2:
        st.write("**Data quality**")
        if isinstance(data_quality, list) and data_quality:
            for q in data_quality:
                st.warning(str(q))
        else:
            st.success("No data quality flags.")
        st.write("**Uncertainty**")
        st.json(uncertainty)

    if isinstance(factors, dict) and "weighted_contributions" in factors:
        wc = factors["weighted_contributions"]
        wc_df = pd.DataFrame({"Factor": list(wc.keys()), "Contribution": list(wc.values())})
        fig = px.bar(wc_df, x="Factor", y="Contribution", title="Weighted contribution to score")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Actions")
    current_status = alert["status"]

    if can("submit_review") and transition_allowed(current_status, "under_review"):
        if st.button("Submit for review"):
            try:
                transition_alert(alert_id, "under_review", "submit_review", "Analyst submitted for review")
                load_reference_caches(force=True)
                st.success("Submitted for review.")
            except Exception as exc:
                st.error(str(exc))

    if can("approve") and transition_allowed(current_status, "approved"):
        reason = st.text_input("Approval reason", "Reviewed and approved for monitoring action")
        if st.button("Approve alert"):
            try:
                transition_alert(alert_id, "approved", "approve", reason)
                load_reference_caches(force=True)
                st.success("Alert approved.")
            except Exception as exc:
                st.error(str(exc))

    if can("override"):
        with st.form("override_form"):
            st.subheader("Reviewer override")
            override_score = st.number_input("Override score", 0.0, 100.0, float(alert["score"]), 0.1)
            override_severity = st.selectbox("Override severity", ["watch", "low", "medium", "high", "data_exception"], index=["watch", "low", "medium", "high", "data_exception"].index(alert["severity"]) if alert["severity"] in ["watch", "low", "medium", "high", "data_exception"] else 0)
            override_reason = st.text_area("Override reason", "Business context not captured by rules")
            submitted = st.form_submit_button("Override and approve")
            if submitted:
                try:
                    transition_alert(
                        alert_id, "approved", "override", override_reason,
                        override_score=float(override_score), override_severity=override_severity,
                    )
                    load_reference_caches(force=True)
                    st.success("Override recorded with audit trail.")
                except Exception as exc:
                    st.error(str(exc))

    if can("archive") and transition_allowed(current_status, "archived"):
        if st.button("Archive alert"):
            try:
                transition_alert(alert_id, "archived", "archive", "Monitoring action completed")
                load_reference_caches(force=True)
                st.success("Archived.")
            except Exception as exc:
                st.error(str(exc))


def render_explainability_page() -> None:
    st.title("Explainability and Human-Review Controls")
    alerts = st.session_state.get("alerts_df", pd.DataFrame())
    if alerts.empty:
        st.info("No alerts to explain.")
        return

    alert_id = st.selectbox("Alert", alerts["id"].tolist(), format_func=lambda x: x)
    alert = alerts[alerts["id"] == alert_id].iloc[0]
    factors = safe_json_loads(alert["factors_json"])
    data_quality = safe_json_loads(alert["data_quality_json"])
    uncertainty = safe_json_loads(alert["uncertainty_json"])

    st.markdown("""
This panel is decision support. It does not imply causality and does not automate credit denial.
""")

    st.subheader("Input values")
    st.json({
        "account_ref": alert.get("account_ref"),
        "organization": alert.get("org_name"),
        "score": alert.get("score"),
        "severity": alert.get("severity"),
        "rules_version": alert.get("rules_version"),
        "generated_at": alert.get("generated_at"),
    })

    st.subheader("Contributing factors")
    if isinstance(factors, dict):
        components = factors.get("components", {})
        weighted = factors.get("weighted_contributions", {})
        df = pd.DataFrame({
            "Factor": list(components.keys()),
            "Component value": list(components.values()),
            "Weighted contribution": [weighted.get(k, 0.0) for k in components.keys()],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = px.bar(df, x="Factor", y="Weighted contribution", title="Weighted contribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Uncertainty and limitations")
    st.json(uncertainty)
    st.caption("Uncertainty reflects data completeness, volatility, and rule/model version. It is not a probability of default.")

    st.subheader("Data quality warnings")
    if isinstance(data_quality, list) and data_quality:
        for q in data_quality:
            st.warning(str(q))
    else:
        st.success("No data quality warnings.")

    st.subheader("Reviewer status")
    st.json({
        "status": alert.get("status"),
        "reviewer_id": alert.get("reviewer_id"),
        "approval_at": alert.get("approval_at"),
        "archive_at": alert.get("archive_at"),
        "export_batch_id": alert.get("export_batch_id"),
    })


def build_evaluation_dataset_conn(conn: sqlite3.Connection) -> pd.DataFrame:
    exp = pd.read_sql_query("SELECT account_id, observation_date, drawn, ead FROM exposures", conn)
    arr = pd.read_sql_query("SELECT account_id, observation_date, days_past_due, arrears_balance FROM arrears", conn)
    acc = pd.read_sql_query("SELECT id AS account_id, product, limit_amount FROM accounts", conn)
    rep = pd.read_sql_query("SELECT account_id, payment_date, status, days_late FROM repayments", conn)

    if exp.empty or arr.empty:
        return pd.DataFrame()

    panel = exp.merge(arr, on=["account_id", "observation_date"], how="inner").merge(acc, on="account_id", how="left")
    panel["month"] = pd.to_datetime(panel["observation_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    panel = panel.dropna(subset=["month", "ead", "days_past_due", "arrears_balance"])

    panel["arrears_ratio"] = np.where(panel["ead"] > 0, panel["arrears_balance"] / panel["ead"], 0.0)
    panel["utilization"] = np.where(panel["limit_amount"] > 0, panel["drawn"] / panel["limit_amount"], 0.0)

    if not rep.empty:
        rep["month"] = pd.to_datetime(rep["payment_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        rep["status_lower"] = rep["status"].astype(str).str.lower()
        rep_stats = rep.groupby(["account_id", "month"]).agg(
            on_time_rate=("status_lower", lambda s: float((s == "on_time").mean())),
            avg_days_late=("days_late", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        ).reset_index()
        panel = panel.merge(rep_stats, on=["account_id", "month"], how="left")

    for col, default in [("on_time_rate", 0.5), ("avg_days_late", 0.0)]:
        if col not in panel.columns:
            panel[col] = default
        else:
            panel[col] = panel[col].fillna(default)

    panel["macro_stress"] = 0.2
    panel = panel.sort_values(["account_id", "month"])
    panel["next_dpd"] = panel.groupby("account_id")["days_past_due"].shift(-1)
    panel["target"] = np.where(panel["next_dpd"].notna(), (panel["next_dpd"] >= 30).astype(float), np.nan)
    panel = panel.dropna(subset=["target"])
    panel["target"] = panel["target"].astype(int)
    return panel


def basic_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / float(len(y_true)) if len(y_true) else 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_evaluation() -> Dict[str, Any]:
    with db_conn() as conn:
        panel = build_evaluation_dataset_conn(conn)
        if panel.empty:
            return {"error": "Insufficient data to build evaluation dataset."}

        months = sorted(panel["month"].unique())
        cutoff_idx = max(1, int(len(months) * 0.75))
        cutoff = months[cutoff_idx]
        train = panel[panel["month"] <= cutoff].copy()
        test = panel[panel["month"] > cutoff].copy()

        if train.empty or test.empty:
            return {"error": "Insufficient time-based split."}

        rules_version = get_active_rules_version(conn)
        weights = RULE_SETS.get(rules_version, RULE_SETS[RULES_VERSION])
        for df in (train, test):
            df["rule_score"] = [compute_rule_score_row(row, weights) for _, row in df.iterrows()]

        y_true = test["target"].values
        y_rule_pred = (test["rule_score"] >= 50).astype(int).values
        y_rule_prob = test["rule_score"].values / 100.0

        metrics: Dict[str, Any] = {"rules_version": rules_version, "cutoff": str(cutoff)}
        metrics["rule_baseline"] = basic_classification_metrics(y_true, y_rule_pred)
        if SKLEARN_DEPS and len(np.unique(y_true)) == 2:
            try:
                metrics["rule_baseline"]["roc_auc"] = float(roc_auc_score(y_true, y_rule_prob))
                metrics["rule_baseline"]["pr_auc"] = float(average_precision_score(y_true, y_rule_prob))
                metrics["rule_baseline"]["brier"] = float(brier_score_loss(y_true, y_rule_prob))
            except Exception:
                pass

        features = ["days_past_due", "arrears_ratio", "utilization", "on_time_rate", "avg_days_late", "macro_stress"]
        ml_metrics = {}
        artifact = {}
        calibration = []

        if SKLEARN_DEPS:
            X_train = train[features]
            y_train = train["target"]
            X_test = test[features]
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            ml_metrics = basic_classification_metrics(y_true, y_pred)
            if len(np.unique(y_true)) == 2:
                try:
                    ml_metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                    ml_metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
                    ml_metrics["brier"] = float(brier_score_loss(y_true, y_prob))
                    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
                    calibration = [{"fraction_positive": float(a), "mean_predicted": float(b)} for a, b in zip(frac_pos, mean_pred)]
                except Exception:
                    pass
            try:
                coef = pipe.named_steps["clf"].coef_[0].tolist()
                intercept = float(pipe.named_steps["clf"].intercept_[0])
                artifact = {"features": features, "coef": coef, "intercept": intercept}
            except Exception:
                artifact = {"features": features}

        subgroups = []
        if "product" in test.columns:
            for product, g in test.groupby("product"):
                if g["target"].nunique() == 2 and SKLEARN_DEPS:
                    try:
                        prob = pipe.predict_proba(g[features])[:, 1]
                        auc = float(roc_auc_score(g["target"], prob))
                        subgroups.append({"product": product, "rows": int(len(g)), "roc_auc": auc})
                    except Exception:
                        pass

        limitations = (
            "Synthetic demo data only. Do not interpret as real-world predictive performance. "
            "Requires representative, permissioned, independently evaluated data and qualified domain review before production claims."
        )

        model_id = new_id("model")
        version = f"{ML_VERSION}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        training_data_version = f"synthetic-demo-rows-{len(panel)}-cutoff-{cutoff}"
        conn.execute(
            """
            INSERT INTO model_registry(id, model_name, version, trained_at, training_data_version, feature_schema_json, metrics_json, limitations, status, artifact_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id, "logistic_regression_baseline", version, now_utc(), training_data_version,
                to_json(features), to_json({"rule_baseline": metrics["rule_baseline"], "ml": ml_metrics, "subgroups": subgroups}),
                limitations, "candidate", to_json(artifact),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
            ("active_model_version", version, now_utc()),
        )

        return {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "cutoff": str(cutoff),
            "metrics": metrics,
            "ml_metrics": ml_metrics,
            "subgroups": subgroups,
            "calibration": calibration,
            "limitations": limitations,
            "features": features,
        }


def render_evaluation_page() -> None:
    st.title("Reproducible Evaluation, Baselines, and Model Registry")
    st.caption("Deterministic rules are the baseline. ML is compared only when justified and evaluated on time-split data.")

    if can("run_engine"):
        if st.button("Run reproducible evaluation"):
            with st.spinner("Building dataset, splitting by time, evaluating baseline and optional ML..."):
                report = run_evaluation()
            st.session_state["evaluation_report"] = report
    else:
        st.info("Current role cannot run evaluation.")

    report = st.session_state.get("evaluation_report")
    if report:
        if "error" in report:
            st.error(report["error"])
        else:
            st.success("Evaluation complete.")
            st.json({k: v for k, v in report.items() if k not in ["calibration", "subgroups"]})

            metric_rows = []
            for name, vals in [("Rule baseline", report.get("metrics", {}).get("rule_baseline", {})), ("ML baseline", report.get("ml_metrics", {}))]:
                for metric, value in vals.items():
                    metric_rows.append({"Model": name, "Metric": metric, "Value": value})
            if metric_rows:
                st.subheader("Metrics")
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

            if report.get("calibration"):
                cal_df = pd.DataFrame(report["calibration"])
                fig = px.line(cal_df, x="mean_predicted", y="fraction_positive", title="Calibration curve (demo)")
                fig.add_hline(y=0, line_dash="dot")
                st.plotly_chart(fig, use_container_width=True)

            if report.get("subgroups"):
                st.subheader("Subgroup performance")
                st.dataframe(pd.DataFrame(report["subgroups"]), use_container_width=True, hide_index=True)

            st.subheader("Limitations")
            st.warning(report.get("limitations", ""))

    with db_conn() as conn:
        registry = pd.read_sql_query("SELECT * FROM model_registry ORDER BY trained_at DESC", conn)
    if not registry.empty:
        st.subheader("Model registry")
        st.dataframe(registry[["model_name", "version", "trained_at", "training_data_version", "status"]], use_container_width=True, hide_index=True)

    st.subheader("Rules version rollback")
    version = st.selectbox("Rules version", list(RULE_SETS.keys()))
    if can("manage_config"):
        if st.button("Activate rules version"):
            with db_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
                    ("active_rules_version", version, now_utc()),
                )
            st.session_state["active_rules_version"] = version
            log_audit("rules_version_activated", "config", version, {"weights": RULE_SETS[version]})
            st.success(f"Activated {version}.")
    else:
        st.info("Only administrators can activate rules versions.")


def render_reports_page() -> None:
    st.title("Reports and Safe Export")
    report_type = st.selectbox("Report", [
        "Portfolio summary", "Active alerts", "Data exceptions", "Audit trail", "Model registry"
    ])

    if report_type == "Portfolio summary":
        alerts = st.session_state.get("alerts_df", pd.DataFrame())
        if alerts.empty:
            df = pd.DataFrame()
        else:
            df = alerts.groupby(["org_name", "product"], dropna=False).agg(
                accounts=("account_id", "nunique"),
                avg_score=("score", "mean"),
                high_alerts=("severity", lambda s: int((s == "high").sum())),
                medium_alerts=("severity", lambda s: int((s == "medium").sum())),
            ).reset_index()
    elif report_type == "Active alerts":
        alerts = st.session_state.get("alerts_df", pd.DataFrame())
        df = alerts[~alerts["status"].isin(["archived", "exported", "superseded"])].copy() if not alerts.empty else pd.DataFrame()
        if not df.empty:
            df["data_quality_flags"] = df["data_quality_json"].apply(lambda x: "; ".join(safe_json_loads(x) or []) if safe_json_loads(x) else "")
            df = df[[
                "account_ref", "org_name", "product", "severity", "score", "status",
                "rules_version", "generated_at", "data_quality_flags"
            ]]
    elif report_type == "Data exceptions":
        alerts = st.session_state.get("alerts_df", pd.DataFrame())
        if alerts.empty:
            df = pd.DataFrame()
        else:
            mask = alerts["data_quality_json"].astype(str).str.contains("CRITICAL", na=False)
            df = alerts[mask].copy()
            if not df.empty:
                df["data_quality_flags"] = df["data_quality_json"].apply(lambda x: "; ".join(safe_json_loads(x) or []))
                df = df[["account_ref", "org_name", "severity", "score", "status", "data_quality_flags"]]
    elif report_type == "Audit trail":
        with db_conn() as conn:
            df = pd.read_sql_query("SELECT ts, actor_id, role, action, entity_type, entity_id, details_json FROM audit_events ORDER BY ts DESC LIMIT 1000", conn)
    else:
        with db_conn() as conn:
            df = pd.read_sql_query("SELECT model_name, version, trained_at, training_data_version, status, limitations FROM model_registry ORDER BY trained_at DESC", conn)

    if df.empty:
        st.info("No rows for selected report.")
        return

    st.dataframe(df.head(200), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", df_to_csv_bytes(df), f"{report_type.replace(' ', '_').lower()}.csv", "text/csv")
    c2.download_button("Download JSON", df_to_json_bytes(df), f"{report_type.replace(' ', '_').lower()}.json", "application/json")
    c3.download_button("Download HTML", df_to_html_bytes(df), f"{report_type.replace(' ', '_').lower()}.html", "text/html")

    if report_type == "Active alerts" and can("export"):
        mark_exported = st.checkbox("Mark approved alerts as exported")
        if st.button("Export and record audit") and mark_exported:
            alerts = st.session_state.get("alerts_df", pd.DataFrame())
            ids = alerts[(alerts["status"] == "approved")]["id"].tolist() if not alerts.empty else []
            batch_id = new_id("expbatch")
            with db_conn() as conn:
                for aid in ids:
                    conn.execute(
                        "UPDATE alerts SET status='exported', export_batch_id=? WHERE id=?",
                        (batch_id, aid),
                    )
                log_audit_conn(conn, "alert_report_exported", "batch", batch_id, {"alert_count": len(ids)})
            load_reference_caches(force=True)
            st.success(f"Exported {len(ids)} approved alerts under batch {batch_id}.")


def render_governance_page() -> None:
    st.title("Audit, Security, Backup, Restore, and Retention")

    with db_conn() as conn:
        audit = pd.read_sql_query("SELECT * FROM audit_events ORDER BY ts DESC LIMIT 500", conn)
    st.subheader("Audit events")
    st.dataframe(audit[["ts", "actor_id", "role", "action", "entity_type", "entity_id"]], use_container_width=True, hide_index=True)

    st.subheader("Access-control matrix")
    perm_df = pd.DataFrame([
        {"Role": ROLE_LABELS[r], **{p: ("Yes" if p in perms else "No") for p in sorted({p for perms in ROLE_PERMISSIONS.values() for p in perms})}}
        for r, perms in ROLE_PERMISSIONS.items()
    ])
    st.dataframe(perm_df, use_container_width=True, hide_index=True)

    st.subheader("Sensitive data handling")
    st.markdown("""
- This MVP uses synthetic demo data only.
- No paid API keys are used.
- Secrets belong in environment variables or a secrets manager, not source control.
- For real customer data, require authentication, server-side authorization, encryption at rest/in transit, de-identification, consent tracking, and retention controls.
- If authentication is implemented in production, use a recognized password-hashing library such as Argon2 or bcrypt.
- File uploads are stored outside relational rows with checksums and metadata.
- SQL access uses parameterized statements.
- Audit events record material changes, approvals, overrides, imports, exports, backups, restores, and deletions.
""")

    st.subheader("Backup and restore")
    if can("backup_restore"):
        if st.button("Create backup"):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(BACKUP_DIR, f"credit_early_warning_{stamp}.sqlite3")
            shutil.copy2(DB_PATH, dest)
            log_audit("backup_created", "database", dest, {"path": dest})
            st.success(f"Backup created: {dest}")

        backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith(".sqlite3")], reverse=True)
        if backups:
            chosen = st.selectbox("Restore backup", backups)
            confirm = st.checkbox("I understand restore overwrites the local database")
            if st.button("Restore selected backup") and confirm:
                src = os.path.join(BACKUP_DIR, chosen)
                shutil.copy2(src, DB_PATH)
                log_audit("backup_restored", "database", DB_PATH, {"source": src})
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state.clear()
                st.success("Database restored. Reloading...")
                st.rerun()
    else:
        st.info("Only administrators can backup or restore.")

    st.subheader("Retention and deletion")
    with db_conn() as conn:
        ret = conn.execute("SELECT value FROM meta WHERE key='retention_days'").fetchone()
    retention_days = int(ret["value"]) if ret and str(ret["value"]).isdigit() else 1825
    new_retention = st.number_input("Retention days", min_value=1, max_value=36500, value=retention_days)
    if can("manage_config") and st.button("Save retention policy"):
        with db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value, updated_at) VALUES (?, ?, ?)",
                ("retention_days", str(int(new_retention)), now_utc()),
            )
        log_audit("retention_updated", "config", "retention_days", {"days": int(new_retention)})
        st.success("Retention policy saved.")

    if can("delete"):
        accounts = st.session_state.get("demo_accounts", pd.DataFrame())
        if not accounts.empty:
            ref = st.selectbox("Delete synthetic account data", accounts["account_ref"].tolist())
            reason = st.text_input("Deletion reason", "Synthetic demo cleanup")
            confirm = st.checkbox("I confirm deletion of this synthetic account and related records")
            if st.button("Delete account data") and confirm:
                acct = accounts[accounts["account_ref"] == ref].iloc[0]
                with db_conn() as conn:
                    alert_ids = [r["id"] for r in conn.execute("SELECT id FROM alerts WHERE account_id=?", (acct["id"],))]
                    for aid in alert_ids:
                        conn.execute("DELETE FROM decisions WHERE alert_id=?", (aid,))
                    for table in ["alerts", "repayments", "exposures", "arrears", "consents"]:
                        conn.execute(f"DELETE FROM {table} WHERE account_id=?", (acct["id"],))
                    conn.execute("DELETE FROM accounts WHERE id=?", (acct["id"],))
                log_audit("account_deleted", "account", acct["id"], {"reason": reason, "account_ref": ref})
                load_reference_caches(force=True)
                st.success("Synthetic account data deleted.")

    st.subheader("Reseed demo data")
    if can("manage_config"):
        if st.button("Force reseed synthetic demo data"):
            seed_demo_data(force=True)
            load_reference_caches(force=True)
            st.session_state["sim_events"] = []
            st.session_state["sim_series"] = []
            st.session_state["sim_clock"] = datetime.now(timezone.utc)
            st.cache_resource.clear()
            st.success("Demo data reseeded.")
            st.rerun()


DATA_DICTIONARY_ROWS = [
    ("accounts", "id", "TEXT", "stable identifier", "UUID-like string", "system", "internal", "primary key"),
    ("accounts", "account_ref", "TEXT", "business account reference", "unique", "core system", "confidential", "unique, required"),
    ("accounts", "limit_amount", "REAL", "credit limit", "currency units", "core system", "confidential", ">=0"),
    ("accounts", "balance", "REAL", "outstanding balance", "currency units", "core system", "confidential", ">=0"),
    ("repayments", "amount", "REAL", "payment amount", "currency units", "payments gateway", "confidential", ">=0"),
    ("repayments", "status", "TEXT", "payment status", "on_time, late, missed, partial", "payments gateway", "internal", "enum"),
    ("exposures", "ead", "REAL", "exposure at default", "currency units", "risk engine", "confidential", ">=drawn"),
    ("arrears", "days_past_due", "INTEGER", "DPD", "days", "core system", "confidential", "0-3650"),
    ("arrears", "arrears_balance", "REAL", "past-due balance", "currency units", "core system", "confidential", ">=0"),
    ("macro_observations", "indicator", "TEXT", "macro indicator name", "unemployment_rate, policy_rate, etc.", "market feed", "public/internal", "required"),
    ("macro_observations", "value", "REAL", "indicator value", "unit-specific", "market feed", "internal", "numeric"),
    ("alerts", "score", "REAL", "early-warning score", "0-100", "domain engine", "internal", "0-100"),
    ("alerts", "severity", "TEXT", "alert band", "watch, low, medium, high, data_exception", "domain engine", "internal", "enum"),
    ("audit_events", "details_json", "TEXT", "audit payload", "JSON", "system", "internal", "valid JSON"),
]


def render_data_dictionary() -> None:
    st.title("Data Dictionary and Lineage Design")
    df = pd.DataFrame(DATA_DICTIONARY_ROWS, columns=[
        "Entity", "Field", "Type", "Meaning", "Allowed / Unit", "Source", "Sensitivity", "Validation"
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download data dictionary CSV", df_to_csv_bytes(df), "data_dictionary.csv", "text/csv")

    st.markdown("""
### Lineage design
Every derived result can be traced to:
- source system,
- source file checksum,
- row number,
- manual actor and reason,
- transformation/rules version,
- scenario version,
- model version where applicable,
- audit event.

Large files are stored on disk; database rows store metadata and checksums. SQLite is used for the local MVP.
The migration path to PostgreSQL preserves the same normalized entities and adds multi-user concurrency,
server-side authorization, and managed backups.
""")


def test_domain_hand() -> None:
    weights = RULE_SETS[RULES_VERSION]
    components = {
        "dpd_risk": 45 / 90.0,
        "arrears_risk": 10000.0 / 90000.0,
        "utilization_risk": 80000.0 / 100000.0,
        "payment_risk": 0.20,
        "macro_stress": 0.30,
    }
    score = compute_weighted_score(components, weights)
    assert abs(score - 41.2777) < 0.02, f"Expected ~41.28, got {score}"


def test_import_validation() -> None:
    df = generate_malformed_sample_df()
    with db_conn() as conn:
        issues, skip, _ = validate_import_dataframe(df, "repayments", conn, {})
    critical = [i for i in issues if i.severity == "critical"]
    assert len(critical) >= 3, f"Expected at least 3 critical issues, got {len(critical)}"
    assert len(skip) >= 3, f"Expected at least 3 skipped rows, got {len(skip)}"


def test_reproducible_metrics() -> None:
    with db_conn() as conn:
        acc = conn.execute("SELECT * FROM accounts LIMIT 1").fetchone()
        if not acc:
            raise AssertionError("No accounts found for reproducibility test")
        params = {"unemployment_delta": 1.0, "rate_delta": 0.5, "fx_depreciation_pct": 2.0, "sector_shock": 0.2}
        m1 = calculate_account_metrics_conn(conn, acc, params)
        m2 = calculate_account_metrics_conn(conn, acc, params)
        assert abs(m1.score - m2.score) < 1e-9, "Scores differ"
        assert m1.severity == m2.severity, "Severity differs"


def test_export_bytes() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    data = df_to_csv_bytes(df)
    assert b"a" in data and b"x" in data, "CSV export missing expected content"


def test_schema_tables() -> None:
    with db_conn() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r["name"] for r in rows}
    required = {"accounts", "repayments", "exposures", "arrears", "alerts", "audit_events", "files"}
    assert required.issubset(names), f"Missing tables: {required - names}"


def test_transition_rules() -> None:
    assert transition_allowed("generated", "under_review")
    assert not transition_allowed("approved", "generated")
    assert transition_allowed("approved", "exported")


def run_self_tests() -> pd.DataFrame:
    tests = [
        ("domain_hand_calculation", test_domain_hand),
        ("import_validation_rejects_malformed", test_import_validation),
        ("reproducible_calculation", test_reproducible_metrics),
        ("export_csv_bytes", test_export_bytes),
        ("database_schema_present", test_schema_tables),
        ("workflow_transition_rules", test_transition_rules),
    ]
    rows = []
    for name, fn in tests:
        try:
            fn()
            rows.append({"Test": name, "Result": "PASS", "Details": ""})
        except AssertionError as exc:
            rows.append({"Test": name, "Result": "FAIL", "Details": str(exc)})
        except Exception as exc:
            rows.append({"Test": name, "Result": "ERROR", "Details": repr(exc)})
    return pd.DataFrame(rows)


def render_self_tests() -> None:
    st.title("Automated Self Tests")
    st.caption("These tests validate software correctness. Domain validity still requires qualified human review and real permissioned data.")
    if st.button("Run self tests"):
        df = run_self_tests()
        st.session_state["self_test_results"] = df

    df = st.session_state.get("self_test_results")
    if df is not None:
        st.dataframe(df, use_container_width=True, hide_index=True)
        if (df["Result"] == "PASS").all():
            st.success("All self tests passed.")
        else:
            st.error("One or more self tests failed.")
    else:
        st.info("No tests run yet.")


def ensure_session() -> None:
    st.session_state.setdefault("active_page", "Live Monitor")
    st.session_state.setdefault("role", "analyst_operator")
    st.session_state.setdefault("actor_id", ROLE_ACTORS["analyst_operator"])
    st.session_state.setdefault("sim_running", True)
    st.session_state.setdefault("sim_speed", 1.0)
    st.session_state.setdefault("sim_clock", datetime.now(timezone.utc))
    st.session_state.setdefault("sim_events", [])
    st.session_state.setdefault("sim_series", [])
    st.session_state.setdefault("active_scenario_id", "scen_base")
    st.session_state.setdefault("active_rules_version", RULES_VERSION)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    init_db()
    ensure_session()
    seed_demo_data()
    load_reference_caches(force=True)
    render_sidebar()

    page = st.session_state.get("active_page", "Live Monitor")
    if page == "Live Monitor":
        render_live_monitor()
    elif page == "Product Brief":
        render_product_brief()
    elif page == "Data Import":
        render_data_import()
    elif page == "Portfolio Records":
        render_portfolio_records()
    elif page == "Early-Warning Engine":
        render_engine_page()
    elif page == "Scenarios & Forecast":
        render_scenarios_page()
    elif page == "Review & Approvals":
        render_review_page()
    elif page == "Explainability":
        render_explainability_page()
    elif page == "Evaluation & Models":
        render_evaluation_page()
    elif page == "Reports & Export":
        render_reports_page()
    elif page == "Audit & Governance":
        render_governance_page()
    elif page == "Data Dictionary":
        render_data_dictionary()
    elif page == "Self Tests":
        render_self_tests()


if __name__ == "__main__":
    main()