"""
OCR Optimization Story Demo (Dash) - Multi-Field Adaptive Thresholds.

Run with:
    python3 dash_base_app.py
"""

from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"

THRESHOLD_DEFAULT = 0.78
THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.95
LAMBDA_REVIEW = 0.20
ADAPTIVE_THRESHOLD_DEFAULT = 0.72

LEARNING_RATE = 0.08
TYPE1_WEIGHT = 2.0
TYPE2_WEIGHT = 0.50
MIN_FEEDBACK_N = 30
DISPUTE_RATE = 0.15
BATCH_SIZE_PER_CAMERA = 80
RELAXATION_DAMPING = 0.35

FIELDS = ("lpn", "lpj", "lpt")
FIELD_LABELS = {"lpn": "LPN", "lpj": "LPJ", "lpt": "LPT"}
FIELD_OFFSET = {"lpn": 0.0, "lpj": -0.04, "lpt": -0.06}

LANES = [
    {"lane_id": 1, "gantry": "Gantry A", "camera": "CAM-1"},
    {"lane_id": 2, "gantry": "Gantry B", "camera": "CAM-2"},
    {"lane_id": 3, "gantry": "Gantry C", "camera": "CAM-3"},
]

# Deliberate camera heterogeneity in synthetic data.
LANE_CONF_SHIFT = {
    1: 0.08,   # stronger camera environment
    2: 0.00,   # baseline
    3: -0.10,  # weaker camera environment
}

LPJ_VALUES = ["TX", "CA", "FL", "NM", "AZ", "CO", "NV", "OK"]
LPT_VALUES = ["PASSENGER", "COMMERCIAL", "GOV", "TEMP", "TRAILER"]
DEFAULT_DATASET_CSV = Path("data/tolling_ocr_dataset.csv")

# Shared chart palette
PAL_BLUE = "#6366f1"
PAL_ORANGE = "#ef5a3c"
PAL_GREEN = "#10b981"
PAL_SLATE = "#64748b"
PAL_BG = "rgba(255,255,255,0.65)"
PAL_GRID = "#dbe4ec"


def pct(value: float) -> str:
    return f"{value:.0%}"


def lane_key(lane_id: int) -> str:
    return str(lane_id)


def clip_threshold(value: float) -> float:
    return float(np.clip(value, THRESHOLD_MIN, THRESHOLD_MAX))


def random_plate_number(rng: np.random.Generator) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    return "".join(rng.choice(list(letters), size=3)) + "".join(rng.choice(list(digits), size=4))


def random_ocr_miss(value: str, field: str, rng: np.random.Generator) -> str:
    if field == "lpj":
        choices = [v for v in LPJ_VALUES if v != value]
        return str(rng.choice(choices))
    if field == "lpt":
        choices = [v for v in LPT_VALUES if v != value]
        return str(rng.choice(choices))

    chars = list(value)
    i = int(rng.integers(0, len(chars)))
    replacement_pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    replacement = rng.choice(list(replacement_pool))
    chars[i] = replacement if replacement != chars[i] else ("9" if chars[i] != "9" else "A")
    return "".join(chars)


def build_event_truth(rng: np.random.Generator) -> dict[str, str]:
    return {
        "lpn": random_plate_number(rng),
        "lpj": str(rng.choice(LPJ_VALUES)),
        "lpt": str(rng.choice(LPT_VALUES)),
    }


def initial_threshold_store_3field() -> dict:
    current = {
        # Start from the same shared 80 baseline used in the exported CSV.
        lane_key(l["lane_id"]): {field: 0.80 for field in FIELDS}
        for l in LANES
    }
    history = {
        lane_key(l["lane_id"]): {field: [current[lane_key(l["lane_id"])][field]] for field in FIELDS}
        for l in LANES
    }
    last_delta = {
        lane_key(l["lane_id"]): {field: 0.0 for field in FIELDS}
        for l in LANES
    }
    return {"current": current, "history": history, "last_delta": last_delta}


def initial_feedback_store_3field() -> dict:
    per_camera = {
        lane_key(l["lane_id"]): {
            "type1": 0.0,
            "type2": 0.0,
            "correct": 0.0,
            "review": 0.0,
            "dispute": 0.0,
            "auto_clear": 0.0,
            "routed_review": 0.0,
        }
        for l in LANES
    }
    per_camera_field = {
        lane_key(l["lane_id"]): {
            field: {"labeled": 0.0, "type1": 0.0, "type2": 0.0}
            for field in FIELDS
        }
        for l in LANES
    }
    return {
        "per_camera": per_camera,
        "per_camera_field": per_camera_field,
        "last_batch_events": [],
        "last_batch_counts": {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0},
    }


def generate_tolling_events_3field(
    step: int,
    n_per_camera: int = BATCH_SIZE_PER_CAMERA,
) -> list[dict]:
    events: list[dict] = []
    for lane in LANES:
        lane_id = lane["lane_id"]
        rng = np.random.default_rng(1000 + lane_id * 211 + step * 97)
        # Simulate slight camera degradation drift over time for weaker cameras.
        drift = -0.015 * max(0, step - 4) if lane_id == 3 else 0.0

        for i in range(n_per_camera):
            truth = build_event_truth(rng)
            conf_map: dict[str, float] = {}
            ocr_map: dict[str, str] = {}
            for field in FIELDS:
                base = rng.beta(5, 2)
                conf = float(np.clip(base + LANE_CONF_SHIFT[lane_id] + FIELD_OFFSET[field] + drift + rng.normal(0, 0.02), 0.02, 0.99))
                conf_map[field] = conf
                is_correct = bool(rng.random() < conf)
                ocr_map[field] = truth[field] if is_correct else random_ocr_miss(truth[field], field, rng)

            events.append(
                {
                    "event_id": f"s{step}-c{lane_id}-{i}",
                    "camera_id": lane["camera"],
                    "lane_id": lane_id,
                    "LPN": truth["lpn"],
                    "LPJ": truth["lpj"],
                    "LPT": truth["lpt"],
                    "LPN_OCRval": ocr_map["lpn"],
                    "LPJ_OCRval": ocr_map["lpj"],
                    "LPT_OCRval": ocr_map["lpt"],
                    "LPN_conf": conf_map["lpn"],
                    "LPJ_conf": conf_map["lpj"],
                    "LPT_conf": conf_map["lpt"],
                }
            )
    return events


def route_event_with_3field_thresholds(event: dict, thresholds_camera: dict[str, float]) -> tuple[str, list[str]]:
    gate_fields = []
    for field in FIELDS:
        if event[f"{FIELD_LABELS[field]}_conf"] < thresholds_camera[field]:
            gate_fields.append(field)
    decision = "review" if gate_fields else "auto_clear"
    return decision, gate_fields


def label_event_error_type(event: dict, decision: str, gate_fields: list[str], rng: np.random.Generator) -> tuple[str | None, str, bool]:
    field_correct = {
        "lpn": event["LPN"] == event["LPN_OCRval"],
        "lpj": event["LPJ"] == event["LPJ_OCRval"],
        "lpt": event["LPT"] == event["LPT_OCRval"],
    }
    overall_correct = all(field_correct.values())
    feedback_source = "none"
    error_label: str | None = None

    if decision == "review":
        feedback_source = "review"
        if overall_correct:
            error_label = "type2"  # false negative: sent to review but correct
        else:
            error_label = "correct"
    else:
        if not overall_correct and rng.random() < DISPUTE_RATE:
            feedback_source = "dispute"
            error_label = "type1"  # false positive: auto-cleared but wrong

    event["decision"] = decision
    event["gate_fields"] = gate_fields
    event["feedback_source"] = feedback_source
    event["error_label"] = error_label
    event["field_correct"] = field_correct
    event["overall_correct"] = overall_correct
    return error_label, feedback_source, overall_correct


def aggregate_feedback_3field(events: list[dict]) -> tuple[dict, dict, dict]:
    per_camera = {
        lane_key(l["lane_id"]): {
            "type1": 0.0,
            "type2": 0.0,
            "correct": 0.0,
            "review": 0.0,
            "dispute": 0.0,
            "auto_clear": 0.0,
            "routed_review": 0.0,
        }
        for l in LANES
    }
    per_camera_field = {
        lane_key(l["lane_id"]): {
            field: {"labeled": 0.0, "type1": 0.0, "type2": 0.0}
            for field in FIELDS
        }
        for l in LANES
    }

    batch_counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}

    for event in events:
        key = lane_key(event["lane_id"])
        cam = per_camera[key]
        if event["decision"] == "review":
            cam["routed_review"] += 1
            batch_counts["review"] += 1
        else:
            cam["auto_clear"] += 1
            batch_counts["auto_clear"] += 1

        source = event["feedback_source"]
        if source == "review":
            cam["review"] += 1
        elif source == "dispute":
            cam["dispute"] += 1

        label = event["error_label"]
        if label is None:
            continue
        if label == "type1":
            cam["type1"] += 1
            batch_counts["type1"] += 1
        elif label == "type2":
            cam["type2"] += 1
            batch_counts["type2"] += 1
        else:
            cam["correct"] += 1

        # Field-level pressure for updates.
        # Type I pressure: auto-cleared and field incorrect.
        if label == "type1":
            for field in FIELDS:
                if not event["field_correct"][field]:
                    rec = per_camera_field[key][field]
                    rec["labeled"] += 1.0
                    rec["type1"] += 1.0
        # Type II pressure: reviewed, overall correct, and field was a gate trigger.
        elif label == "type2":
            for field in event["gate_fields"]:
                rec = per_camera_field[key][field]
                rec["labeled"] += 1.0
                rec["type2"] += 1.0
        else:
            for field in event["gate_fields"]:
                rec = per_camera_field[key][field]
                rec["labeled"] += 1.0

    return per_camera, per_camera_field, batch_counts


def simulate_batch_counts_for_policy(
    events: list[dict],
    thresholds: dict[str, dict[str, float]],
    seed: int = 0,
) -> dict[str, int]:
    """
    Run the routing + feedback labeling logic on a batch under a given threshold policy.
    Returns policy-level counts for quick KPI comparison.
    """
    rng = np.random.default_rng(991 + seed)
    counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}
    for event in events:
        event_copy = dict(event)
        cam_key = lane_key(event_copy["lane_id"])
        decision, gate_fields = route_event_with_3field_thresholds(event_copy, thresholds[cam_key])
        label, _source, _correct = label_event_error_type(event_copy, decision, gate_fields, rng)
        if decision == "review":
            counts["review"] += 1
        else:
            counts["auto_clear"] += 1
        if label == "type1":
            counts["type1"] += 1
        elif label == "type2":
            counts["type2"] += 1
    return counts


def compute_outcome_counts(events: list[dict], thresholds: dict[str, dict[str, float]]) -> dict[str, int]:
    """
    Policy outcome counts from ground truth (not sampled feedback labels).
    """
    counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}
    for event in events:
        cam_key = lane_key(event["lane_id"])
        decision, _ = route_event_with_3field_thresholds(event, thresholds[cam_key])
        is_correct = (event["LPN"] == event["LPN_OCRval"]) and (event["LPJ"] == event["LPJ_OCRval"]) and (event["LPT"] == event["LPT_OCRval"])
        if decision == "auto_clear":
            counts["auto_clear"] += 1
            if not is_correct:
                counts["type1"] += 1
        else:
            counts["review"] += 1
            if is_correct:
                counts["type2"] += 1
    return counts


def update_thresholds_3field(
    thresholds: dict[str, dict[str, float]],
    per_camera_field: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    updated = deepcopy(thresholds)
    deltas = {cam: {field: 0.0 for field in FIELDS} for cam in thresholds}
    for cam, fields in per_camera_field.items():
        for field, rec in fields.items():
            labeled = rec["labeled"]
            if labeled <= 0:
                continue
            type1_rate = rec["type1"] / labeled
            type2_rate = rec["type2"] / labeled
            eta = LEARNING_RATE if labeled >= MIN_FEEDBACK_N else LEARNING_RATE * 0.25
            delta = eta * (TYPE1_WEIGHT * type1_rate - TYPE2_WEIGHT * type2_rate)
            # Safety bias for demo stability:
            # allow tightening quickly, but relax (lower tau) more slowly.
            if delta < 0:
                delta *= RELAXATION_DAMPING
            updated[cam][field] = clip_threshold(updated[cam][field] + delta)
            deltas[cam][field] = float(delta)
    return updated, deltas


def evaluate_policy(events: list[dict], thresholds: dict[str, dict[str, float]]) -> dict:
    tp = fp = fn = tn = 0
    per_camera = {lane_key(l["lane_id"]): {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for l in LANES}

    for event in events:
        key = lane_key(event["lane_id"])
        decision, _ = route_event_with_3field_thresholds(event, thresholds[key])
        is_correct = (event["LPN"] == event["LPN_OCRval"]) and (event["LPJ"] == event["LPJ_OCRval"]) and (event["LPT"] == event["LPT_OCRval"])

        target = per_camera[key]
        if decision == "auto_clear" and is_correct:
            tp += 1
            target["tp"] += 1
        elif decision == "auto_clear" and (not is_correct):
            fp += 1
            target["fp"] += 1
        elif decision == "review" and is_correct:
            fn += 1
            target["fn"] += 1
        else:
            tn += 1
            target["tn"] += 1

    total = tp + fp + fn + tn
    auto_rate = (tp + fp) / total if total else 0.0
    review_rate = (fn + tn) / total if total else 0.0
    bad_auto_rate = fp / total if total else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    utility = accuracy - LAMBDA_REVIEW * review_rate

    by_camera = {}
    for key, rec in per_camera.items():
        ct = rec["tp"] + rec["fp"] + rec["fn"] + rec["tn"]
        by_camera[key] = {
            "auto_rate": (rec["tp"] + rec["fp"]) / ct if ct else 0.0,
            "review_rate": (rec["fn"] + rec["tn"]) / ct if ct else 0.0,
            "bad_auto_rate": rec["fp"] / ct if ct else 0.0,
        }

    return {
        "overall": {"auto_rate": auto_rate, "review_rate": review_rate, "bad_auto_rate": bad_auto_rate, "utility": utility},
        "by_camera": by_camera,
    }


def export_dataset_csv(path: Path = DEFAULT_DATASET_CSV, n_per_camera: int = 1200, step: int = 0) -> Path:
    """
    Export a deterministic synthetic multi-field OCR dataset to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Main CSV mirrors current real-world-style baseline: fixed shared 80 cutoff.
    thresholds = thresholds_from_shared(0.80)
    events = generate_tolling_events_3field(step=step, n_per_camera=n_per_camera)
    rng = np.random.default_rng(4040 + step)

    fieldnames = [
        "event_id",
        "camera_id",
        "lane_id",
        "LPN",
        "LPJ",
        "LPT",
        "LPN_OCRval",
        "LPJ_OCRval",
        "LPT_OCRval",
        "LPN_conf",
        "LPJ_conf",
        "LPT_conf",
        "LPN_thresh",
        "LPJ_thresh",
        "LPT_thresh",
        "decision",
        "feedback_source",
        "error_label",
        "overall_correct",
    ]

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            cam_key = lane_key(event["lane_id"])
            decision, gate_fields = route_event_with_3field_thresholds(event, thresholds[cam_key])
            label_event_error_type(event, decision, gate_fields, rng)
            writer.writerow(
                {
                    "event_id": event["event_id"],
                    "camera_id": event["camera_id"],
                    "lane_id": event["lane_id"],
                    "LPN": event["LPN"],
                    "LPJ": event["LPJ"],
                    "LPT": event["LPT"],
                    "LPN_OCRval": event["LPN_OCRval"],
                    "LPJ_OCRval": event["LPJ_OCRval"],
                    "LPT_OCRval": event["LPT_OCRval"],
                    "LPN_conf": f"{event['LPN_conf']:.6f}",
                    "LPJ_conf": f"{event['LPJ_conf']:.6f}",
                    "LPT_conf": f"{event['LPT_conf']:.6f}",
                    "LPN_thresh": f"{thresholds[cam_key]['lpn']:.6f}",
                    "LPJ_thresh": f"{thresholds[cam_key]['lpj']:.6f}",
                    "LPT_thresh": f"{thresholds[cam_key]['lpt']:.6f}",
                    "decision": event["decision"],
                    "feedback_source": event["feedback_source"],
                    "error_label": event["error_label"] or "",
                    "overall_correct": int(event["overall_correct"]),
                }
            )
    return path


def thresholds_from_shared(shared: float) -> dict[str, dict[str, float]]:
    return {
        lane_key(l["lane_id"]): {field: clip_threshold(shared) for field in FIELDS}
        for l in LANES
    }


def graph_card(graph_id: str) -> html.Div:
    return html.Div(className="story-card", children=[dcc.Graph(id=graph_id, config={"displayModeBar": False})])


def make_score_meaning_block() -> html.Div:
    sample = {
        "plate": "MD 7BK2391",
        "lpn_score": 84,
        "lpj_score": 79,
        "lpt_score": 74,
        "lpn_thresh": 80,
        "lpj_thresh": 80,
        "lpt_thresh": 80,
    }
    checks = [
        ("LPJ", sample["lpj_score"], sample["lpj_thresh"]),
        ("LPN", sample["lpn_score"], sample["lpn_thresh"]),
        ("LPT", sample["lpt_score"], sample["lpt_thresh"]),
    ]
    check_rows = []
    for label, score, thresh in checks:
        passed = score >= thresh
        check_rows.append(
            html.Div(
                className=f"score-check-row field-{label.lower()}",
                children=[
                    html.Div(label, className="score-check-label"),
                    html.Div(f"confidence score: {score}", className="score-check-value"),
                    html.Div(f"threshold {thresh}", className="score-check-thresh"),
                    html.Div("PASS" if passed else "REVIEW", className=f"score-check-status {'pass' if passed else 'review'}"),
                ],
            )
        )

    return html.Div(
        className="story-card mt-3 score-meaning-interactive",
        children=[
            html.H6("Step 1: What the OCR score means", className="mb-2"),
            html.P(
                "Each plate read returns three model scores: number (LPN), jurisdiction (LPJ), and type (LPT). "
                "Each score is compared against its threshold for routing.",
                className="text-secondary mb-2",
            ),
            html.Div(
                className="score-plate-card mb-2",
                children=[
                    html.Div("Sample plate read", className="score-plate-kicker"),
                    html.Div(
                        className="score-plate-value score-plate-main",
                        children=[
                            html.Span(sample["plate"].split()[0], className="plate-piece token-lpj"),
                            html.Span(" "),
                            html.Span(sample["plate"].split()[1], className="plate-piece token-lpn"),
                        ],
                    ),
                    html.Div(
                        className="score-plate-type-wrap",
                        children=[html.Span("TYPE: PASSENGER", className="score-token token-lpt score-type-tag")],
                    ),
                ],
            ),
            html.Div(check_rows),
            html.Div(
                "Routing rule: if any one check fails, the whole event is routed to review.",
                className="text-secondary small mt-2 mb-0",
            ),
        ],
    )


def make_arbitrary_cutoff_figure(shared_threshold: float, canonical_cutoff: int = 80) -> go.Figure:
    """
    Explain cutoff semantics with a score ruler:
    one cutoff line splits the same reads into two routing zones.
    """
    rng = np.random.default_rng(902)
    conf = np.clip(rng.beta(5, 2, size=120), 0.02, 0.99)
    scores = np.round(conf * 99).astype(int)
    cutoff = canonical_cutoff
    y = rng.normal(0.0, 0.065, size=len(scores))
    # Simulated truth for explanation: high score can still occasionally be wrong.
    correct_flags = rng.random(len(conf)) < conf
    review_mask = scores < cutoff
    auto_mask = ~review_mask
    wrong_auto_mask = auto_mask & (~correct_flags)
    review_n = int(np.sum(review_mask))
    auto_n = int(np.sum(auto_mask))
    total_n = max(1, review_n + auto_n)
    review_rate = review_n / total_n
    auto_rate = auto_n / total_n

    fig = go.Figure()
    fig.add_vrect(x0=45, x1=cutoff, fillcolor="rgba(245,158,11,0.20)", line_width=0, layer="below")
    fig.add_vrect(x0=cutoff, x1=99, fillcolor="rgba(22,163,74,0.08)", line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=scores[review_mask],
            y=y[review_mask],
            mode="markers",
            name="Sent to review",
            marker=dict(size=10, color="#f59e0b", line=dict(color="rgba(15,23,42,0.2)", width=0.6)),
            hovertemplate="Score %{x}<br>Decision: review<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scores[auto_mask],
            y=y[auto_mask],
            mode="markers",
            name="Auto-clear",
            marker=dict(size=10, color=PAL_GREEN, line=dict(color="rgba(15,23,42,0.2)", width=0.6)),
            hovertemplate="Score %{x}<br>Decision: auto-clear<extra></extra>",
        )
    )
    if np.any(wrong_auto_mask):
        bad_idx = np.where(wrong_auto_mask)[0]
        sample_n = min(4, len(bad_idx))
        sampled_idx = bad_idx[:sample_n]
        x_bad_all = scores[sampled_idx].astype(float)
        y_bad_all = y[sampled_idx].astype(float)
        x_bad = float(x_bad_all[0])
        y_bad = float(y_bad_all[0])
        fig.add_trace(
            go.Scatter(
                x=x_bad_all,
                y=y_bad_all,
                mode="markers",
                name="Disputed auto-clear",
            marker=dict(size=12, color=PAL_ORANGE, symbol="x", line=dict(color="#9a3412", width=1.8)),
                hovertemplate="Score %{x:.0f}<br>Disputed auto-clear<extra></extra>",
            )
        )
        fig.add_annotation(
            x=x_bad,
            y=y_bad,
            ax=min(x_bad + 8, 96),
            ay=0.18,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.1,
            arrowcolor="#92400e",
            bgcolor="rgba(254,242,242,0.96)",
            bordercolor="rgba(220,38,38,0.55)",
            borderwidth=1,
            borderpad=5,
            text="Disputed auto-clear",
            font=dict(size=10, color="#dc2626"),
            align="left",
        )
    fig.add_vline(x=cutoff, line_color="rgba(51,65,85,0.45)", line_width=2.0, line_dash="dash")
    fig.add_annotation(x=cutoff, y=0.24, text=f"Cutoff = {cutoff}", showarrow=False, font=dict(size=11, color="#334155"))
    fig.add_annotation(x=61, y=0.24, text=f"Review zone: {review_n} reads ({pct(review_rate)})", showarrow=False, font=dict(size=10, color="#b45309"))
    fig.add_annotation(x=89, y=0.24, text=f"Auto zone: {auto_n} reads ({pct(auto_rate)})", showarrow=False, font=dict(size=10, color="#047857"))

    fig.update_layout(
        title="Cutoff explained: one line splits the score ruler into two decisions",
        height=300,
        margin=dict(l=12, r=12, t=54, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.13, x=0),
    )
    fig.update_xaxes(title="OCR score points (ordinal scale, not probability)", range=[45, 99], dtick=5, gridcolor=PAL_GRID)
    fig.update_yaxes(title="", range=[-0.3, 0.3], visible=False, showgrid=False, zeroline=False)
    return fig


def make_results_comparison_figure(shared_threshold: float, current_thresholds: dict[str, dict[str, float]], sim_step: int) -> go.Figure:
    events = generate_tolling_events_3field(step=max(sim_step, 1), n_per_camera=450)
    stagnant_counts = compute_outcome_counts(events, thresholds_from_shared(shared_threshold))
    adaptive_counts = compute_outcome_counts(events, current_thresholds)

    metrics = ["Auto-clear", "Review", "Type I", "Type II"]
    stagnant_vals = [
        stagnant_counts["auto_clear"],
        stagnant_counts["review"],
        stagnant_counts["type1"],
        stagnant_counts["type2"],
    ]
    adaptive_vals = [
        adaptive_counts["auto_clear"],
        adaptive_counts["review"],
        adaptive_counts["type1"],
        adaptive_counts["type2"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=stagnant_vals,
            name="Stagnant shared 80",
            marker=dict(color=PAL_BLUE),
            hovertemplate="%{x}<br>Stagnant: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=adaptive_vals,
            name="Adaptive thresholds",
            marker=dict(color=PAL_ORANGE),
            hovertemplate="%{x}<br>Adaptive: %{y}<extra></extra>",
        )
    )

    type1_delta = adaptive_counts["type1"] - stagnant_counts["type1"]
    type2_delta = adaptive_counts["type2"] - stagnant_counts["type2"]

    fig.update_layout(
        title=f"Results comparison: shared 80 vs adaptive thresholds<br><sup>Type I change: {type1_delta:+d} | Type II change: {type2_delta:+d} (negative is better)</sup>",
        barmode="group",
        height=360,
        margin=dict(l=12, r=12, t=114, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.07, x=0),
    )
    fig.update_yaxes(title="Event count", gridcolor=PAL_GRID, rangemode="tozero", automargin=True)
    fig.update_xaxes(showgrid=False)
    return fig


def make_shared_vs_adaptive_figure(shared_threshold: float, current_thresholds: dict[str, dict[str, float]], sim_step: int) -> go.Figure:
    events = generate_tolling_events_3field(step=max(sim_step, 1), n_per_camera=280)
    shared_metrics = evaluate_policy(events, thresholds_from_shared(shared_threshold))
    adaptive_metrics = evaluate_policy(events, current_thresholds)
    labels = ["Auto-clear", "Type I", "Review", "Utility"]
    shared_values = [
        shared_metrics["overall"]["auto_rate"],
        shared_metrics["overall"]["bad_auto_rate"],
        shared_metrics["overall"]["review_rate"],
        shared_metrics["overall"]["utility"],
    ]
    adaptive_values = [
        adaptive_metrics["overall"]["auto_rate"],
        adaptive_metrics["overall"]["bad_auto_rate"],
        adaptive_metrics["overall"]["review_rate"],
        adaptive_metrics["overall"]["utility"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=shared_values, name="Shared setting", marker=dict(color="#475569"), text=[pct(v) for v in shared_values], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=adaptive_values, name="Camera-specific settings", marker=dict(color="#0f766e"), text=[pct(v) for v in adaptive_values], textposition="outside"))
    fig.update_layout(
        title="Shared Setting vs Camera-Specific 3-Field Settings (Utility = higher is better)",
        barmode="group",
        height=340,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.14, x=0),
    )
    fig.update_yaxes(title="Rate", tickformat=".0%", range=[0, 1], gridcolor="#dbe4ec")
    fig.update_xaxes(showgrid=False)
    return fig


def make_shared_vs_adaptive_dumbbell_figure(shared_threshold: float, current_thresholds: dict[str, dict[str, float]], sim_step: int) -> go.Figure:
    events = generate_tolling_events_3field(step=max(sim_step, 1), n_per_camera=280)
    shared_metrics = evaluate_policy(events, thresholds_from_shared(shared_threshold))
    adaptive_metrics = evaluate_policy(events, current_thresholds)

    metrics = [
        ("Auto-clear", shared_metrics["overall"]["auto_rate"], adaptive_metrics["overall"]["auto_rate"]),
        ("Type I", shared_metrics["overall"]["bad_auto_rate"], adaptive_metrics["overall"]["bad_auto_rate"]),
        ("Review", shared_metrics["overall"]["review_rate"], adaptive_metrics["overall"]["review_rate"]),
        ("Utility", shared_metrics["overall"]["utility"], adaptive_metrics["overall"]["utility"]),
    ]
    labels = [m[0] for m in metrics]
    shared_vals = [m[1] for m in metrics]
    adaptive_vals = [m[2] for m in metrics]

    fig = go.Figure()
    for label, s_val, a_val in metrics:
        fig.add_trace(
            go.Scatter(
                x=[s_val, a_val],
                y=[label, label],
                mode="lines",
                line=dict(color="#94a3b8", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=shared_vals,
            y=labels,
            mode="markers+text",
            marker=dict(size=11, color="#475569"),
            text=[pct(v) for v in shared_vals],
            textposition="middle left",
            name="Shared 80",
            hovertemplate="%{y}<br>Shared 80: %{x:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=adaptive_vals,
            y=labels,
            mode="markers+text",
            marker=dict(size=11, color="#0f766e"),
            text=[pct(v) for v in adaptive_vals],
            textposition="middle right",
            name="Adaptive",
            hovertemplate="%{y}<br>Adaptive: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Dumbbell View: Shared 80 vs Adaptive",
        height=340,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.14, x=0),
    )
    fig.update_xaxes(title="Rate", tickformat=".0%", range=[0, 1], gridcolor="#dbe4ec")
    fig.update_yaxes(title="", categoryorder="array", categoryarray=list(reversed(labels)), gridcolor="#eef2f7")
    return fig


def make_shared_vs_adaptive_waterfall_figure(shared_threshold: float, current_thresholds: dict[str, dict[str, float]], sim_step: int) -> go.Figure:
    events = generate_tolling_events_3field(step=max(sim_step, 1), n_per_camera=450)
    shared_counts = compute_outcome_counts(events, thresholds_from_shared(shared_threshold))
    adaptive_counts = compute_outcome_counts(events, current_thresholds)

    delta = {
        "Auto-clear": adaptive_counts["auto_clear"] - shared_counts["auto_clear"],
        "Type I": adaptive_counts["type1"] - shared_counts["type1"],
        "Review": adaptive_counts["review"] - shared_counts["review"],
        "Type II": adaptive_counts["type2"] - shared_counts["type2"],
    }
    metrics = ["Auto-clear", "Type I", "Review", "Type II"]
    colors = []
    for metric in metrics:
        val = delta[metric]
        # For Type I / Type II / Review, lower is better.
        if metric in {"Type I", "Type II", "Review"}:
            colors.append(PAL_GREEN if val <= 0 else PAL_ORANGE)
        else:
            colors.append(PAL_GREEN if val >= 0 else PAL_ORANGE)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=[delta[m] for m in metrics],
            marker=dict(color=colors),
            text=[f"{int(d):+d}" for d in [delta[m] for m in metrics]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Adaptive - Shared 80: %{y:+.0f} events<extra></extra>",
            name="Net change in events",
        )
    )
    fig.add_hline(y=0, line_width=1.4, line_color=PAL_SLATE)
    fig.update_layout(
        title="Net change from adaptive thresholding (event counts)",
        height=340,
        margin=dict(l=12, r=12, t=56, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title="Change in event count", gridcolor=PAL_GRID, automargin=True)
    return fig


def make_machine_action_table(events: list[dict], n_rows: int = 10) -> html.Div:
    rows = []
    shown = events[:n_rows]
    for event in shown:
        rows.append(
            html.Tr(
                [
                    html.Td(event["camera_id"]),
                    html.Td(event["LPN"], className="text-nowrap"),
                    html.Td(event["LPN_OCRval"], className="text-nowrap"),
                    html.Td(f"{int(round(event['LPN_conf'] * 99))}"),
                    html.Td(f"{int(round(event['LPJ_conf'] * 99))}"),
                    html.Td(f"{int(round(event['LPT_conf'] * 99))}"),
                    html.Td("Review" if event["decision"] == "review" else "Auto-clear"),
                    html.Td(event["error_label"] or "unlabeled"),
                    html.Td(event["feedback_source"]),
                ]
            )
        )

    return html.Div(
        className="story-card",
        children=[
            html.H6("Machine in Action (sample batch)", className="mb-2"),
            html.Div(
                className="table-responsive",
                children=[
                    html.Table(
                        className="table table-sm align-middle mb-0",
                        children=[
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Camera"),
                                        html.Th("LPN"),
                                        html.Th("LPN OCR"),
                                        html.Th("LPN score"),
                                        html.Th("LPJ score"),
                                        html.Th("LPT score"),
                                        html.Th("Decision"),
                                        html.Th("Label"),
                                        html.Th("Feedback"),
                                    ]
                                )
                            ),
                            html.Tbody(rows),
                        ],
                    )
                ],
            ),
        ],
    )


def make_error_count_kpis(adaptive_counts: dict, fixed_counts: dict, sim_step: int) -> html.Div:
    adaptive_tiles = [
        ("Step", str(sim_step)),
        ("Auto-cleared", str(int(adaptive_counts["auto_clear"]))),
        ("Routed to review", str(int(adaptive_counts["review"]))),
        ("Type I", str(int(adaptive_counts["type1"]))),
        ("Type II", str(int(adaptive_counts["type2"]))),
    ]
    fixed_tiles = [
        ("Baseline cutoff", "80"),
        ("Auto-cleared", str(int(fixed_counts["auto_clear"]))),
        ("Routed to review", str(int(fixed_counts["review"]))),
        ("Type I", str(int(fixed_counts["type1"]))),
        ("Type II", str(int(fixed_counts["type2"]))),
    ]

    def panel(title: str, tiles: list[tuple[str, str]]) -> html.Div:
        return html.Div(
            className="story-card kpi-panel",
            children=[
                html.Div(title, className="small text-secondary mb-2 fw-semibold"),
                html.Div(
                    className="kpi-grid",
                    children=[
                        html.Div(
                            className="kpi-mini",
                            children=[
                                html.Div(label, className="kpi-mini-label"),
                                html.Div(value, className="kpi-mini-value"),
                            ],
                        )
                        for label, value in tiles
                    ],
                ),
            ],
        )

    return html.Div(
        className="row g-3",
        children=[
            html.Div(
                className="col-12 col-xl-6",
                children=[panel("Adaptive thresholds (current run)", adaptive_tiles)],
            ),
            html.Div(
                className="col-12 col-xl-6",
                children=[panel("Simulated under fixed shared cutoff = 80", fixed_tiles)],
            ),
        ],
    )


def policy_from_time_step(time_step: int, dispute_rate: float) -> tuple[str, float, float, str]:
    if time_step < 6:
        stage_label = "Initial calibration"
        base_rate = 0.20
    elif time_step < 16:
        stage_label = "Transitional tuning"
        base_rate = 0.10
    else:
        stage_label = "Steady-state monitoring"
        base_rate = 0.05

    rec_rate = min(0.25, base_rate + 0.05) if dispute_rate > 0.02 else base_rate
    trigger_label = "Escalated (dispute spike)" if rec_rate > base_rate else "Normal"
    return stage_label, base_rate, rec_rate, trigger_label


def make_label_source_figure(events: list[dict], time_step: int) -> go.Figure:
    routed_review_n = sum(1 for e in events if e.get("decision") == "review")
    dispute_n = sum(1 for e in events if e.get("feedback_source") == "dispute")
    total = max(1, len(events))
    dispute_rate = dispute_n / total
    _stage, _base, rec_rate, _trigger = policy_from_time_step(time_step, dispute_rate)

    # Time-dependent review labeling: only a sampled share of review-routed events is labeled.
    review_n = int(round(routed_review_n * rec_rate))
    review_n = max(0, min(review_n, routed_review_n))
    unlabeled_n = max(0, total - review_n - dispute_n)
    labeled_total = review_n + dispute_n

    fig = go.Figure()
    # Stage 1: total events into labeled vs unlabeled.
    fig.add_trace(
        go.Bar(
            x=[labeled_total, 0],
            y=["All events", "Labeled events"],
            orientation="h",
            name="Labeled",
            marker=dict(color=PAL_BLUE),
            text=[f"{labeled_total} ({labeled_total / total:.0%})", ""],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Labeled events<br>%{x} events<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[unlabeled_n, 0],
            y=["All events", "Labeled events"],
            orientation="h",
            name="Unlabeled",
            marker=dict(color=PAL_SLATE),
            text=[f"{unlabeled_n} ({unlabeled_n / total:.0%})", ""],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Unlabeled events<br>%{x} events<extra></extra>",
        )
    )
    # Stage 2: labeled events into review vs dispute sources.
    fig.add_trace(
        go.Bar(
            x=[0, review_n],
            y=["All events", "Labeled events"],
            orientation="h",
            name="Review labels",
            marker=dict(color=PAL_GREEN),
            text=["", f"{review_n}"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Review labels<br>%{x} events<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[0, dispute_n],
            y=["All events", "Labeled events"],
            orientation="h",
            name="Dispute labels",
            marker=dict(color=PAL_ORANGE),
            text=["", f"{dispute_n}"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Dispute labels<br>%{x} events<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Feedback pipeline this batch ({total} events)",
        barmode="stack",
        height=260,
        margin=dict(l=12, r=12, t=56, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.15, x=0),
    )
    fig.update_xaxes(title="Event count", rangemode="tozero", gridcolor=PAL_GRID)
    fig.update_yaxes(title="", showgrid=False)
    return fig


def make_audit_policy_panel(events: list[dict], time_step: int) -> html.Div:
    routed_review_n = sum(1 for e in events if e.get("decision") == "review")
    dispute_n = sum(1 for e in events if e.get("feedback_source") == "dispute")
    total = max(1, len(events))
    dispute_rate = dispute_n / total
    stage_label, _base_rate, rec_rate, trigger_label = policy_from_time_step(time_step, dispute_rate)
    review_n = int(round(routed_review_n * rec_rate))
    review_n = max(0, min(review_n, routed_review_n))

    return html.Div(
        className="story-card",
        children=[
            html.H6("Audit policy (learning signal)", className="mb-2"),
            html.Div(f"Time step: {time_step}", className="small text-secondary mb-1"),
            html.Div(f"Stage: {stage_label}", className="small text-secondary mb-1"),
            html.Div(f"Recommended audit rate: {int(round(rec_rate * 100))}%", className="fw-semibold mb-1"),
            html.Div(f"Policy status: {trigger_label}", className="small text-secondary mb-2"),
            html.Div(
                className="small text-secondary",
                children=[
                    html.Div(f"Reviewed labels this batch: {review_n}"),
                    html.Div(f"Review-routed events this batch: {routed_review_n}"),
                    html.Div(f"Dispute labels this batch: {dispute_n}"),
                    html.Div("Rule of thumb: higher audits early, lower after convergence, raise again when disputes rise."),
                ],
            ),
        ],
    )


def make_audit_rate_over_time_figure(max_step: int = 24, current_step: int = 0) -> go.Figure:
    steps = list(range(max_step + 1))
    # Synthetic dispute signal pattern to illustrate occasional escalation.
    dispute_series = [0.010 + (0.018 if s in {9, 17, 18} else 0.0) for s in steps]
    rates = [policy_from_time_step(s, d)[2] for s, d in zip(steps, dispute_series)]
    current_step = max(0, min(int(current_step), max_step))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps[: current_step + 1],
            y=rates[: current_step + 1],
            mode="lines+markers",
            line=dict(color=PAL_BLUE, width=3),
            marker=dict(size=8, color=PAL_ORANGE),
            name="Observed policy rate",
            hovertemplate="Step %{x}<br>Rate %{y:.0%}<extra></extra>",
        )
    )
    if current_step < max_step:
        fig.add_trace(
            go.Scatter(
                x=steps[current_step:],
                y=rates[current_step:],
                mode="lines",
                line=dict(color="rgba(99,102,241,0.35)", width=2, dash="dot"),
                name="Projected schedule",
                hovertemplate="Step %{x}<br>Rate %{y:.0%}<extra></extra>",
            )
        )

    # Stage bands
    fig.add_vrect(x0=0, x1=5.5, fillcolor="rgba(239,90,60,0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=5.5, x1=15.5, fillcolor="rgba(99,102,241,0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=15.5, x1=max_step, fillcolor="rgba(16,185,129,0.08)", line_width=0, layer="below")
    fig.add_annotation(x=2.5, y=0.235, text="Calibration", showarrow=False, font=dict(size=10, color="#9a3412"))
    fig.add_annotation(x=10.5, y=0.235, text="Transition", showarrow=False, font=dict(size=10, color="#4338ca"))
    fig.add_annotation(x=20.0, y=0.235, text="Steady state", showarrow=False, font=dict(size=10, color="#047857"))

    fig.update_layout(
        title="Recommended audit rate over time",
        height=300,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.14, x=0),
        updatemenus=[],
    )
    fig.update_xaxes(title="Time step", range=[0, max_step], dtick=3, gridcolor=PAL_GRID)
    fig.update_yaxes(title="Audit rate", range=[0, 0.25], tickformat=".0%", gridcolor=PAL_GRID)
    return fig


def make_threshold_matrix_table(threshold_store: dict) -> html.Div:
    rows = []
    for lane in LANES:
        key = lane_key(lane["lane_id"])
        cells = [html.Td(lane["camera"])]
        for field in FIELDS:
            tau = threshold_store["current"][key][field]
            delta = threshold_store["last_delta"][key][field]
            cells.append(html.Td(f"{tau:.3f} ({delta:+.4f})"))
        rows.append(html.Tr(cells))

    return html.Div(
        className="story-card",
        children=[
            html.H6("Per-Camera Threshold Matrix (LPN / LPJ / LPT)", className="mb-2"),
            html.Div(
                className="table-responsive",
                children=[
                    html.Table(
                        className="table table-sm align-middle mb-0",
                        children=[
                            html.Thead(
                                html.Tr([html.Th("Camera"), html.Th("LPN"), html.Th("LPJ"), html.Th("LPT")])
                            ),
                            html.Tbody(rows),
                        ],
                    )
                ],
            ),
        ],
    )


def make_field_threshold_trend_figure(threshold_store: dict, selected_camera: str) -> go.Figure:
    lane_id = int(selected_camera)
    key = lane_key(lane_id)
    fig = go.Figure()
    for field in FIELDS:
        y = threshold_store["history"][key][field]
        x = list(range(len(y)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=FIELD_LABELS[field],
                line=dict(width=2.5),
                marker=dict(size=6),
            )
        )
    fig.update_layout(
        title=f"Threshold Drift by Field - {next(l['camera'] for l in LANES if l['lane_id'] == lane_id)}",
        height=320,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.13, x=0),
    )
    fig.update_xaxes(title="Update steps", gridcolor="#dbe4ec")
    fig.update_yaxes(title="Threshold", range=[THRESHOLD_MIN, THRESHOLD_MAX], tickformat=".0%", gridcolor="#dbe4ec")
    return fig


app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.title = "OCR Threshold Optimization Story"

app.layout = html.Div(
    className="story-shell",
    children=[
        dcc.Store(id="store-thresholds-3field", data=initial_threshold_store_3field()),
        dcc.Store(id="store-feedback-3field", data=initial_feedback_store_3field()),
        dcc.Store(id="store-sim-step-3field", data=0),
        html.Div(className="story-orb orb-a"),
        html.Div(className="story-orb orb-b"),
        html.Main(
            className="container-xl py-4 py-md-5",
            children=[
                html.Section(
                    className="hero-block mb-5 mt-3",
                    children=[
                        html.Div(
                            className="hero-copy",
                            children=[
                                html.Div("OCR THRESHOLDING", className="kicker"),
                                html.H1("Find the optimal threshold between bad auto-passes and costly review", className="display-6 mb-3"),
                                html.P([html.Strong("Today:"), " one shared cutoff decides auto-clear vs review for LPN, LPJ, and LPT."], className="text-secondary mb-2"),
                                html.P([html.Strong("Problem:"), " a cutoff like 80 is a score point, not a true probability, and all cameras do not behave the same."], className="text-secondary mb-2"),
                                html.P([html.Strong("What we are testing:"), " update thresholds by camera and field using feedback to reduce misses without creating review overload."], className="text-secondary mb-0"),
                            ],
                        ),
                        html.Div(
                            className="story-card mt-3",
                            children=[
                                html.H6("Hypothesis", className="mb-1"),
                                html.P(
                                    "A single shared cutoff causes avoidable misses and avoidable reviews. "
                                    "Camera-specific, field-specific updates should improve the quality/workload balance.",
                                    className="text-secondary mb-0",
                                ),
                            ],
                        ),
                        make_score_meaning_block(),
                        html.Div(
                            className="story-card narrative-caption-card mt-3",
                            children=[
                                dcc.Graph(id="arbitrary-cutoff-graph", config={"displayModeBar": False}),
                                html.Div("80 is one line on a score distribution, not a probability of correctness.", className="narrative-caption"),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="story-section mb-5",
                    children=[
                        html.Div("Machine In Action", className="section-kicker"),
                        html.H2("Three fields per camera with adaptive thresholds", className="mb-3"),
                        html.Div(
                            className="story-card mb-3",
                            children=[
                                html.H6("How the updating works", className="mb-2"),
                                html.Div(
                                    className="text-secondary small",
                                    children=[
                                        html.Div("Each read gets 3 scores (LPN, LPJ, LPT). If any score is below threshold, it goes to review."),
                                        html.Div("We learn from reviewed reads and customer disputes."),
                                        html.Div("Too many wrong auto-clears pushes thresholds up; too many unnecessary reviews pushes them down, in small steps."),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="story-card mb-3",
                            children=[
                                html.Div(
                                    className="row g-3 align-items-end",
                                    children=[
                                        html.Div(
                                            className="col-12 col-lg-4",
                                            children=[
                                                html.Label("Comparison baseline", className="mb-2 fw-semibold"),
                                                html.Div("Fixed shared cutoff: 80 for all cameras", className="threshold-readout mt-2"),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-12 col-lg-3",
                                            children=[
                                                html.Label("Trend camera", htmlFor="trend-camera", className="mb-2 fw-semibold"),
                                                dcc.Dropdown(
                                                    id="trend-camera",
                                                    options=[{"label": f"{lane['camera']} ({lane['gantry']})", "value": str(lane["lane_id"])} for lane in LANES],
                                                    value="1",
                                                    clearable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="col-12 col-lg-5",
                                            children=[
                                                html.Div(
                                                    className="d-flex gap-2",
                                                    children=[
                                                        html.Button("Run next update", id="run-step-3field-btn", className="btn btn-primary w-100"),
                                                        html.Button("Reset", id="reset-3field-btn", className="btn btn-outline-secondary w-100"),
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(id="error-count-kpis", className="mb-3"),
                        html.Div(className="row g-3", children=[html.Div(className="col-12 col-xl-8", children=html.Div(id="machine-action-table")), html.Div(className="col-12 col-xl-4", children=html.Div(id="threshold-matrix-table"))]),
                        html.Div(className="mt-3", children=[graph_card("field-threshold-trend-graph")]),
                    ],
                ),
                html.Section(
                    className="story-section mb-5",
                    children=[
                        html.Div("Results", className="section-kicker"),
                        html.H2("Adaptive thresholding vs standard shared cutoff", className="mb-3"),
                        html.Div(
                            className="row g-3",
                            children=[
                                html.Div(className="col-12 col-xl-6", children=[graph_card("fixed-threshold-camera-outcomes")]),
                                html.Div(className="col-12 col-xl-6", children=[graph_card("shared-vs-adaptive-waterfall-graph")]),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="story-section mb-4",
                    children=[
                        html.Div("Learning Signal", className="section-kicker"),
                        html.H2("Audit coverage and policy", className="mb-3"),
                        html.Div(
                            className="row g-3",
                            children=[
                                html.Div(className="col-12 col-xl-6", children=[html.Div(id="audit-policy-note")]),
                                html.Div(
                                    id="audit-rate-over-time-wrap",
                                    className="col-12 col-xl-6",
                                    children=[graph_card("audit-rate-over-time-graph")],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="methodology-toggle-wrap",
                    children=[
                        html.Button("Methodology", id="methodology-toggle-btn", className="btn btn-outline-secondary methodology-corner-btn"),
                    ],
                ),
                html.Div(
                    id="methodology-panel",
                    style={"display": "none"},
                    children=[
                        html.Section(
                            className="story-section mt-3 mb-5",
                            children=[
                                html.Div("Methodology", className="section-kicker"),
                                html.H2("How the adaptive threshold method works", className="mb-3"),
                                html.Div(className="story-card mb-3", children=[dcc.Markdown(mathjax=True, children=r"""
#### 1) Event Model (Per Camera, Per Read)

For each event $e$ from camera $c$, we keep three OCR channels:

$$
f \in \{\mathrm{LPN}, \mathrm{LPJ}, \mathrm{LPT}\}
$$

$$
\big(y_{e,f},\ \hat{y}_{e,f},\ s_{e,f}\big)
$$

where:

- $y_{e,f}$: true field value
- $\hat{y}_{e,f}$: OCR predicted value
- $s_{e,f}$: OCR confidence score (ordinal score points, not probability)

Each camera has field-specific thresholds:

$$
\tau_{c,\mathrm{LPN}},\ \tau_{c,\mathrm{LPJ}},\ \tau_{c,\mathrm{LPT}}
$$
""")]),
                                html.Div(className="story-card mb-3", children=[dcc.Markdown(mathjax=True, children=r"""
#### 2) Routing Rule (Operational Decision)

An event is routed to review if **any** field fails threshold:

$$
r_e = \mathbf{1}\!\left[(s_{e,\mathrm{LPN}} < \tau_{c,\mathrm{LPN}})\ \lor\ (s_{e,\mathrm{LPJ}} < \tau_{c,\mathrm{LPJ}})\ \lor\ (s_{e,\mathrm{LPT}} < \tau_{c,\mathrm{LPT}})\right]
$$

$$
a_e = 1 - r_e
$$

This is an OR-gate design: one weak field can route the full event to review.
""")]),
                                html.Div(className="story-card mb-3", children=[dcc.Markdown(mathjax=True, children=r"""
#### 3) Error Definitions Used for Learning

Define strict event correctness:

$$
\mathrm{Correct}_e = \mathbf{1}\!\left[\bigwedge_{f \in \{\mathrm{LPN},\mathrm{LPJ},\mathrm{LPT}\}} (\hat{y}_{e,f} = y_{e,f})\right]
$$

Then:

$$
\mathrm{TypeI}_e = \mathbf{1}\!\left[a_e = 1 \land \mathrm{Correct}_e = 0\right]
$$

$$
\mathrm{TypeII}_e = \mathbf{1}\!\left[r_e = 1 \land \mathrm{Correct}_e = 1\right]
$$
""")]),
                                html.Div(className="story-card mb-3", children=[dcc.Markdown(mathjax=True, children=r"""
#### 4) Adaptive Update Rule (Per Camera, Per Field)

For camera $c$, field $f$, and step $t$:

$$
\tau_{c,f}^{(t+1)}=\mathrm{clip}\!\left(
\tau_{c,f}^{(t)} + \eta\left(w_1\,r_{c,f}^{\mathrm{I}} - w_2\,r_{c,f}^{\mathrm{II}}\right),
\tau_{\min},\tau_{\max}
\right)
$$

where:

- $\eta$: learning rate (step size)
- $w_1, w_2$: weights (usually $w_1 > w_2$ to prioritize Type I risk)
- $r_{c,f}^{\mathrm{I}}$: Type I pressure estimate
- $r_{c,f}^{\mathrm{II}}$: Type II pressure estimate
""")]),
                                html.Div(className="story-card", children=[dcc.Markdown(mathjax=True, children=r"""
#### 5) Standard Method vs Adaptive Method

Standard method:

$$
\tau_{c,f} = 80\ \ \forall c,f
$$

Adaptive method:

$$
\tau_{c,f}^{(t)}\ \text{updates by camera and field over time}
$$

Resulting objective tradeoff:

$$
\min_{\tau}\ \Big(\alpha\cdot \mathrm{TypeI}(\tau) + \beta\cdot \mathrm{ReviewLoad}(\tau)\Big)
$$
""")]),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("store-thresholds-3field", "data"),
    Output("store-feedback-3field", "data"),
    Output("store-sim-step-3field", "data"),
    Input("run-step-3field-btn", "n_clicks"),
    Input("reset-3field-btn", "n_clicks"),
    State("store-thresholds-3field", "data"),
    State("store-feedback-3field", "data"),
    State("store-sim-step-3field", "data"),
    prevent_initial_call=True,
)
def step_machine(
    _run_clicks: int | None,
    _reset_clicks: int | None,
    threshold_store: dict,
    feedback_store: dict,
    sim_step: int,
):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    if trigger == "reset-3field-btn":
        return initial_threshold_store_3field(), initial_feedback_store_3field(), 0
    if trigger != "run-step-3field-btn":
        return no_update, no_update, no_update

    threshold_store = threshold_store or initial_threshold_store_3field()
    feedback_store = feedback_store or initial_feedback_store_3field()
    sim_step = int(sim_step or 0) + 1
    thresholds = threshold_store["current"]

    events = generate_tolling_events_3field(step=sim_step, n_per_camera=BATCH_SIZE_PER_CAMERA)
    rng = np.random.default_rng(777 + sim_step)
    for event in events:
        cam_key = lane_key(event["lane_id"])
        decision, gate_fields = route_event_with_3field_thresholds(event, thresholds[cam_key])
        label_event_error_type(event, decision, gate_fields, rng)

    per_camera, per_camera_field, batch_counts = aggregate_feedback_3field(events)
    updated_thresholds, deltas = update_thresholds_3field(thresholds, per_camera_field)

    new_store = deepcopy(threshold_store)
    new_store["current"] = updated_thresholds
    new_store["last_delta"] = deltas
    for lane in LANES:
        key = lane_key(lane["lane_id"])
        for field in FIELDS:
            new_store["history"][key][field].append(updated_thresholds[key][field])
            if len(new_store["history"][key][field]) > 40:
                new_store["history"][key][field] = new_store["history"][key][field][-40:]

    new_feedback = deepcopy(feedback_store)
    for lane in LANES:
        key = lane_key(lane["lane_id"])
        for metric in ["type1", "type2", "correct", "review", "dispute", "auto_clear", "routed_review"]:
            new_feedback["per_camera"][key][metric] += per_camera[key][metric]
        for field in FIELDS:
            for metric in ["labeled", "type1", "type2"]:
                new_feedback["per_camera_field"][key][field][metric] += per_camera_field[key][field][metric]
    new_feedback["last_batch_events"] = events
    new_feedback["last_batch_counts"] = batch_counts
    return new_store, new_feedback, sim_step


@app.callback(
    Output("methodology-panel", "style"),
    Output("methodology-toggle-btn", "children"),
    Input("methodology-toggle-btn", "n_clicks"),
)
def toggle_methodology(n_clicks: int | None):
    is_open = bool(n_clicks and (n_clicks % 2 == 1))
    if is_open:
        return {"display": "block"}, "Hide methodology"
    return {"display": "none"}, "Methodology"


@app.callback(
    Output("arbitrary-cutoff-graph", "figure"),
    Output("fixed-threshold-camera-outcomes", "figure"),
    Output("shared-vs-adaptive-waterfall-graph", "figure"),
    Output("machine-action-table", "children"),
    Output("error-count-kpis", "children"),
    Output("audit-policy-note", "children"),
    Output("audit-rate-over-time-graph", "figure"),
    Output("threshold-matrix-table", "children"),
    Output("field-threshold-trend-graph", "figure"),
    Input("trend-camera", "value"),
    Input("store-thresholds-3field", "data"),
    Input("store-feedback-3field", "data"),
    Input("store-sim-step-3field", "data"),
)
def render_story(
    trend_camera: str,
    threshold_store: dict,
    feedback_store: dict,
    sim_step: int,
):
    threshold_store = threshold_store or initial_threshold_store_3field()
    feedback_store = feedback_store or initial_feedback_store_3field()
    sim_step = int(sim_step or 0)
    shared_threshold = 0.80
    audit_time_step = min(sim_step, 24)

    batch_events = feedback_store.get("last_batch_events") or generate_tolling_events_3field(step=0, n_per_camera=16)
    if not feedback_store.get("last_batch_events"):
        # Provide initial routed/label state for first render.
        rng = np.random.default_rng(42)
        for event in batch_events:
            cam_key = lane_key(event["lane_id"])
            decision, gate_fields = route_event_with_3field_thresholds(event, threshold_store["current"][cam_key])
            label_event_error_type(event, decision, gate_fields, rng)

    # Keep shared-80 baseline stagnant on a fixed reference batch.
    fixed_reference_events = generate_tolling_events_3field(step=0, n_per_camera=BATCH_SIZE_PER_CAMERA)
    fixed_shared_counts = simulate_batch_counts_for_policy(
        fixed_reference_events,
        thresholds_from_shared(shared_threshold),
        seed=0,
    )

    return (
        make_arbitrary_cutoff_figure(shared_threshold, canonical_cutoff=80),
        make_results_comparison_figure(shared_threshold, threshold_store["current"], sim_step),
        make_shared_vs_adaptive_waterfall_figure(shared_threshold, threshold_store["current"], sim_step),
        make_machine_action_table(batch_events, n_rows=10),
        make_error_count_kpis(
            feedback_store.get("last_batch_counts", {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}),
            fixed_shared_counts,
            sim_step,
        ),
        make_audit_policy_panel(batch_events, audit_time_step),
        make_audit_rate_over_time_figure(24, audit_time_step),
        make_threshold_matrix_table(threshold_store),
        make_field_threshold_trend_figure(threshold_store, trend_camera or "1"),
    )


server = app.server

if __name__ == "__main__":
    export_dataset_csv()
    app.run(debug=True)
