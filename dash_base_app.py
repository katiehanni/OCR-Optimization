"""
OCR Optimization Story Demo (Dash) - Multi-Field Adaptive Thresholds.

Run with:
    python3 dash_base_app.py
"""

from __future__ import annotations

from copy import deepcopy
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"

THRESHOLD_DEFAULT = 80.0
THRESHOLD_MIN = 1.0
THRESHOLD_MAX = 99.0
LAMBDA_REVIEW = 0.20
ADAPTIVE_THRESHOLD_DEFAULT = 80.0
ADAPTIVE_INIT_RELAX = 4.0

DISPUTE_COST_DEFAULT = 0.20
REVIEW_COST_DEFAULT = 0.07
LEARNING_RATE = 0.22
TYPE1_WEIGHT = 2.0
TYPE2_WEIGHT = 0.35
MIN_FEEDBACK_N = 6
DISPUTE_RATE = 0.35
BATCH_SIZE_PER_CAMERA = 80
RELAXATION_DAMPING = 0.35
UPDATE_BATCHES_PER_CLICK = 1
DEMO_PRECOMPUTED_MAX_STEP = 10
REVIEW_TARGET = 0.10
OBJECTIVE_REVIEW_WEIGHT = 0.20
REVIEW_LIFT_CAP_RATE = 0.02
DEMO_MIN_AUTO_CLEAR_RATE = 0.68
DEMO_AUTO_CLEAR_SHORTFALL_COST = 0.90
DEMO_RESULTS_STEP = 23

TARGET_REVIEW_RATE = 0.10
SCORE_WEIGHTS = {"lpn": 0.50, "lpj": 0.35, "lpt": 0.15}
FIELD_SCORE_MEANS = {"lpn": 91.0, "lpj": 86.0, "lpt": 50.0}
FIELD_SCORE_VAR = 5.0
FIELD_SCORE_STD = float(np.sqrt(FIELD_SCORE_VAR))
CAMERA_SCORE_SHIFT = {1: 1.0, 2: 0.0, 3: -1.0}
CAMERA_ACCURACY_BIAS = {1: 0.08, 2: 0.00, 3: -0.14}
FIELD_ACCURACY_BIAS = {"lpn": 0.05, "lpj": 0.02, "lpt": -0.10}

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
DEMO_DEGRADATION_CAMERA = "CAM-1"
DEMO_DEGRADATION_FIELD = "lpn"
DEMO_DEGRADATION_MIN_SCORE = 88.0
DEMO_HEALTH_SCORE_MAP = {
    ("CAM-1", "lpj"): 18.0,
    ("CAM-1", "lpn"): 92.0,
    ("CAM-1", "lpt"): 24.0,
    ("CAM-2", "lpj"): 34.0,
    ("CAM-2", "lpn"): 20.0,
    ("CAM-2", "lpt"): 48.0,
    ("CAM-3", "lpj"): 44.0,
    ("CAM-3", "lpn"): 52.0,
    ("CAM-3", "lpt"): 46.0,
}
DEGRADATION_START_STEP = 3
SHOCK_DROP_TRIGGER = 2.5
SHOCK_COST_MULTIPLIER = 2.0
DEMO_SHOCK_MIN_UPLIFT = 1.5
DEMO_SHOCK_MIN_RELAX = 4.0
DEMO_SHOCK_RECOVERY_STEPS = 4
SHOCK_SCHEDULE = [
    {
        "lane_id": 3,
        "field": "lpt",
        "start": 1,
        "end": 6,
        "score_shift": -8.0,
        "accuracy_shift": -0.16,
        "label": "CAM-3 LPT degradation",
    },
    {
        "lane_id": 2,
        "field": "lpj",
        "start": 5,
        "end": 9,
        "score_shift": -4.0,
        "accuracy_shift": -0.08,
        "label": "CAM-2 LPJ lighting shock",
    },
    {
        "lane_id": 1,
        "field": "lpn",
        "start": 4,
        "end": 8,
        "display_start": 3,
        "display_end": 8,
        "score_shift": -10.0,
        "accuracy_shift": -0.20,
        "label": "CAM-1 LPN image quality degradation",
    },
]


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


def field_degradation_adjustment(lane_id: int, field: str, step: int) -> tuple[float, float]:
    score_shift = 0.0
    accuracy_shift = 0.0
    for shock in SHOCK_SCHEDULE:
        if shock["lane_id"] != lane_id or shock["field"] != field:
            continue
        if shock["start"] <= step <= shock["end"]:
            score_shift += float(shock["score_shift"])
            accuracy_shift += float(shock["accuracy_shift"])
    return score_shift, accuracy_shift


def shocks_for_lane(lane_id: int) -> list[dict]:
    return [shock for shock in SHOCK_SCHEDULE if shock["lane_id"] == lane_id]


def scheduled_shock_active(lane_key_value: str, field: str, step: int) -> bool:
    lane_id = int(lane_key_value.replace("cam_", ""))
    for shock in SHOCK_SCHEDULE:
        if shock["lane_id"] == lane_id and shock["field"] == field and shock["start"] <= step <= shock["end"]:
            return True
    return False


def scheduled_shock_start(lane_key_value: str, field: str) -> int | None:
    lane_id = int(lane_key_value.replace("cam_", ""))
    for shock in SHOCK_SCHEDULE:
        if shock["lane_id"] == lane_id and shock["field"] == field:
            return int(shock["start"])
    return None


def scheduled_shock_end(lane_key_value: str, field: str) -> int | None:
    lane_id = int(lane_key_value.replace("cam_", ""))
    for shock in SHOCK_SCHEDULE:
        if shock["lane_id"] == lane_id and shock["field"] == field:
            return int(shock["end"])
    return None


def generate_field_scores_normal(rng: np.random.Generator, lane_id: int, step: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for field in FIELDS:
        score_shift, _ = field_degradation_adjustment(lane_id, field, step)
        score = rng.normal(FIELD_SCORE_MEANS[field], FIELD_SCORE_STD) + CAMERA_SCORE_SHIFT[lane_id] + score_shift
        scores[field] = float(np.clip(score, 1.0, 99.0))
    return scores


def compute_weighted_score(event: dict) -> float:
    return float(
        SCORE_WEIGHTS["lpn"] * event["LPN_score"]
        + SCORE_WEIGHTS["lpj"] * event["LPJ_score"]
        + SCORE_WEIGHTS["lpt"] * event["LPT_score"]
    )


def calibrate_weighted_baseline_threshold(events: list[dict], target_review_rate: float = TARGET_REVIEW_RATE) -> float:
    weighted_scores = np.array([e["weighted_score"] for e in events], dtype=float)
    quantile = float(np.quantile(weighted_scores, target_review_rate))
    # Use whole-point cutoff for demo readability (round down).
    return clip_threshold(float(np.floor(quantile)))


def calibrate_or_baseline_threshold(events: list[dict], target_review_rate: float = TARGET_REVIEW_RATE) -> float:
    grid = np.arange(1.0, 95.5, 0.5)
    best_tau = 80.0
    best_gap = float("inf")
    for tau in grid:
        thresholds = thresholds_from_shared(float(tau))
        counts = compute_outcome_counts(events, thresholds)
        review_rate = counts["review"] / max(1, len(events))
        gap = abs(review_rate - target_review_rate)
        if gap < best_gap:
            best_gap = gap
            best_tau = float(tau)
    return clip_threshold(best_tau)


def calibrate_initial_or_thresholds(events: list[dict], target_review_rate: float = TARGET_REVIEW_RATE) -> dict[str, dict[str, float]]:
    """
    Build per-camera, per-field starting thresholds for OR-gate routing.
    We set each field to the same fail-quantile target so thresholds differ by field distribution.
    """
    # If three independent field checks each fail at r, OR review rate is 1-(1-r)^3.
    field_fail_target = 1.0 - (1.0 - target_review_rate) ** (1.0 / 3.0)
    out: dict[str, dict[str, float]] = {}
    for lane in LANES:
        key = lane_key(lane["lane_id"])
        out[key] = {}
        lane_events = [e for e in events if lane_key(e["lane_id"]) == key]
        for field in FIELDS:
            vals = np.array([e[f"{FIELD_LABELS[field]}_score"] for e in lane_events], dtype=float)
            out[key][field] = clip_threshold(float(np.quantile(vals, field_fail_target)) - ADAPTIVE_INIT_RELAX)
    return out


def route_event_weighted(event: dict, tau_camera_weighted: float) -> str:
    return "review" if event["weighted_score"] < tau_camera_weighted else "auto_clear"


def route_event_with_3field_thresholds(event: dict, thresholds_camera: dict[str, float]) -> tuple[str, list[str]]:
    gate_fields: list[str] = []
    for field in FIELDS:
        if event[f"{FIELD_LABELS[field]}_score"] < thresholds_camera[field]:
            gate_fields.append(field)
    return ("review" if gate_fields else "auto_clear"), gate_fields


def thresholds_from_shared(shared: float) -> dict[str, dict[str, float]]:
    tau = clip_threshold(shared)
    return {lane_key(l["lane_id"]): {field: tau for field in FIELDS} for l in LANES}


def thresholds_from_shared_weighted(shared: float) -> dict[str, float]:
    tau = clip_threshold(shared)
    return {lane_key(l["lane_id"]): tau for l in LANES}


def generate_tolling_events_3field(
    step: int,
    n_per_camera: int = BATCH_SIZE_PER_CAMERA,
) -> list[dict]:
    events: list[dict] = []
    for lane in LANES:
        lane_id = lane["lane_id"]
        rng = np.random.default_rng(1000 + lane_id * 211 + step * 97)
        for i in range(n_per_camera):
            truth = build_event_truth(rng)
            score_map = generate_field_scores_normal(rng, lane_id, step)
            ocr_map: dict[str, str] = {}
            field_correct: dict[str, bool] = {}

            for field in FIELDS:
                score = score_map[field]
                _, acc_shift = field_degradation_adjustment(lane_id, field, step)
                p_correct = np.clip(
                    0.35 + 0.006 * score + CAMERA_ACCURACY_BIAS[lane_id] + FIELD_ACCURACY_BIAS[field] + acc_shift,
                    0.05,
                    0.995,
                )
                is_correct = bool(rng.random() < p_correct)
                field_correct[field] = is_correct
                ocr_map[field] = truth[field] if is_correct else random_ocr_miss(truth[field], field, rng)

            event = {
                "event_id": f"s{step}-c{lane_id}-{i}",
                "camera_id": lane["camera"],
                "lane_id": lane_id,
                "LPN": truth["lpn"],
                "LPJ": truth["lpj"],
                "LPT": truth["lpt"],
                "LPN_OCRval": ocr_map["lpn"],
                "LPJ_OCRval": ocr_map["lpj"],
                "LPT_OCRval": ocr_map["lpt"],
                "LPN_score": score_map["lpn"],
                "LPJ_score": score_map["lpj"],
                "LPT_score": score_map["lpt"],
                "field_correct": field_correct,
            }
            event["weighted_score"] = compute_weighted_score(event)
            events.append(event)
    return events


def initial_threshold_store_3field() -> dict:
    calibration_events = generate_tolling_events_3field(step=0, n_per_camera=1200)
    baseline_tau_weighted = calibrate_weighted_baseline_threshold(calibration_events, TARGET_REVIEW_RATE)
    current = calibrate_initial_or_thresholds(calibration_events, TARGET_REVIEW_RATE)
    history = {
        lane_key(l["lane_id"]): {field: [current[lane_key(l["lane_id"])][field]] for field in FIELDS}
        for l in LANES
    }
    last_delta = {lane_key(l["lane_id"]): {field: 0.0 for field in FIELDS} for l in LANES}
    baseline_tau_or = float(np.mean([current[lane_key(l["lane_id"])][field] for l in LANES for field in FIELDS]))
    return {
        "current": current,
        "history": history,
        "last_delta": last_delta,
        "baseline_tau_or": baseline_tau_or,
        "baseline_tau_weighted": baseline_tau_weighted,
    }


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
            "labeled": 0.0,
        }
        for l in LANES
    }
    per_camera_field = {
        lane_key(l["lane_id"]): {field: {"labeled": 0.0, "type1": 0.0, "type2": 0.0} for field in FIELDS}
        for l in LANES
    }
    return {
        "per_camera": per_camera,
        "per_camera_field": per_camera_field,
        "prev_score_means": {
            lane_key(l["lane_id"]): {field: None for field in FIELDS}
            for l in LANES
        },
        "last_batch_events": [],
        "last_batch_counts": {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0},
        "time_history": {
            "step": [],
            "adaptive_type1": [],
            "adaptive_review": [],
            "adaptive_cost": [],
            "baseline_type1": [],
            "baseline_review": [],
            "baseline_cost": [],
        },
    }


def label_event_error_type(event: dict, decision: str, gate_fields: list[str], rng: np.random.Generator) -> tuple[str | None, str, bool]:
    overall_correct = bool(event["field_correct"]["lpn"] and event["field_correct"]["lpj"] and event["field_correct"]["lpt"])
    feedback_source = "none"
    error_label: str | None = None

    if decision == "review":
        feedback_source = "review"
        error_label = "type2" if overall_correct else "correct"
    else:
        if (not overall_correct) and rng.random() < DISPUTE_RATE:
            feedback_source = "dispute"
            error_label = "type1"

    event["decision"] = decision
    event["gate_fields"] = gate_fields
    event["feedback_source"] = feedback_source
    event["error_label"] = error_label
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
            "labeled": 0.0,
        }
        for l in LANES
    }
    per_camera_field = {
        lane_key(l["lane_id"]): {field: {"labeled": 0.0, "type1": 0.0, "type2": 0.0} for field in FIELDS}
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

        if event["feedback_source"] == "review":
            cam["review"] += 1
        elif event["feedback_source"] == "dispute":
            cam["dispute"] += 1

        label = event["error_label"]
        if label is None:
            continue

        cam["labeled"] += 1
        if label == "type1":
            cam["type1"] += 1
            batch_counts["type1"] += 1
            for field in FIELDS:
                if not event["field_correct"][field]:
                    rec = per_camera_field[key][field]
                    rec["labeled"] += 1.0
                    rec["type1"] += 1.0
        elif label == "type2":
            cam["type2"] += 1
            batch_counts["type2"] += 1
            for field in event.get("gate_fields", []):
                rec = per_camera_field[key][field]
                rec["labeled"] += 1.0
                rec["type2"] += 1.0
        else:
            cam["correct"] += 1

    return per_camera, per_camera_field, batch_counts


def update_thresholds_3field(
    thresholds: dict[str, dict[str, float]],
    events: list[dict],
    sim_step: int,
    dispute_cost: float = DISPUTE_COST_DEFAULT,
    review_cost: float = REVIEW_COST_DEFAULT,
    prev_score_means: dict[str, dict[str, float | None]] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    updated = deepcopy(thresholds)
    deltas = {cam: {field: 0.0 for field in FIELDS} for cam in thresholds}
    events_by_cam = {lane_key(l["lane_id"]): [e for e in events if lane_key(e["lane_id"]) == lane_key(l["lane_id"])] for l in LANES}

    for cam, cam_events in events_by_cam.items():
        if not cam_events:
            continue
        for field in FIELDS:
            current_tau = float(updated[cam][field])
            shock_start = scheduled_shock_start(cam, field)
            shock_end = scheduled_shock_end(cam, field)
            if shock_start is not None and sim_step < shock_start:
                updated[cam][field] = current_tau
                deltas[cam][field] = 0.0
                continue
            observed_scores = [float(e[f"{FIELD_LABELS[field]}_score"]) for e in cam_events]
            current_mean = float(np.mean(observed_scores)) if observed_scores else current_tau
            prev_mean = None if prev_score_means is None else prev_score_means.get(cam, {}).get(field)
            shock_active = scheduled_shock_active(cam, field, sim_step) or (
                prev_mean is not None and (prev_mean - current_mean) >= SHOCK_DROP_TRIGGER
            )
            effective_dispute_cost = dispute_cost * (SHOCK_COST_MULTIPLIER if shock_active else 1.0)
            candidates = sorted(
                {
                    clip_threshold(round(v * 2.0) / 2.0)
                    for v in observed_scores + [current_tau, current_tau + 1.0, current_tau + 2.0, current_tau + 3.0, current_tau + 4.0]
                }
            )
            best_tau = current_tau
            best_counts = compute_camera_counts(cam_events, updated[cam])
            best_cost = policy_objective(best_counts, dispute_cost=effective_dispute_cost, review_cost=review_cost)
            best_meets_majority = (best_counts["auto_clear"] / max(1, len(cam_events))) >= DEMO_MIN_AUTO_CLEAR_RATE

            for candidate_tau in candidates:
                trial_thresholds = dict(updated[cam])
                trial_thresholds[field] = float(candidate_tau)
                counts = compute_camera_counts(cam_events, trial_thresholds)
                cost = policy_objective(counts, dispute_cost=effective_dispute_cost, review_cost=review_cost)
                meets_majority = (counts["auto_clear"] / max(1, len(cam_events))) >= DEMO_MIN_AUTO_CLEAR_RATE
                if (
                    (meets_majority and not best_meets_majority)
                    or (
                        meets_majority == best_meets_majority
                        and cost < best_cost - 1e-9
                    )
                    or (
                        meets_majority == best_meets_majority
                        and abs(cost - best_cost) <= 1e-9
                        and counts["type1"] < best_counts["type1"]
                    )
                    or (
                        meets_majority == best_meets_majority
                        and abs(cost - best_cost) <= 1e-9
                        and counts["type1"] == best_counts["type1"]
                        and counts["review"] < best_counts["review"]
                    )
                ):
                    best_tau = float(candidate_tau)
                    best_counts = counts
                    best_cost = cost
                    best_meets_majority = meets_majority

            if shock_active:
                best_tau = clip_threshold(max(best_tau, current_tau + DEMO_SHOCK_MIN_UPLIFT))
            elif shock_end is not None and shock_end < sim_step <= (shock_end + DEMO_SHOCK_RECOVERY_STEPS):
                best_tau = clip_threshold(min(best_tau, current_tau - DEMO_SHOCK_MIN_RELAX))

            updated[cam][field] = best_tau
            deltas[cam][field] = float(best_tau - current_tau)

    return updated, deltas


def compute_results_kpis(
    shared_threshold: float,
    current_thresholds: dict[str, dict[str, float]],
    sim_step: int,
    dispute_cost: float,
    review_cost: float,
) -> dict[str, float | str]:
    baseline, adaptive, n_events = get_cached_results_counts(shared_threshold, current_thresholds, sim_step)
    n = max(1, n_events)

    type1_delta = adaptive["type1"] - baseline["type1"]
    review_lift_rate = (adaptive["review"] - baseline["review"]) / n
    review_lift_count = adaptive["review"] - baseline["review"]
    type1_delta_rate = type1_delta / max(1, baseline["type1"])
    objective_delta = policy_objective(adaptive, dispute_cost=dispute_cost, review_cost=review_cost) - policy_objective(
        baseline, dispute_cost=dispute_cost, review_cost=review_cost
    )

    return {
        "baseline_type1": float(baseline["type1"]),
        "adaptive_type1": float(adaptive["type1"]),
        "type1_delta": float(type1_delta),
        "type1_delta_rate": float(type1_delta_rate),
        "review_lift_rate": float(review_lift_rate),
        "review_lift_count": float(review_lift_count),
        "objective_delta": float(objective_delta),
    }


def results_eval_step(sim_step: int) -> int:
    # Compare policies in a fixed, late degraded operating regime where the adaptive policy has had time to respond.
    return max(DEMO_RESULTS_STEP, int(sim_step or 0))


def compute_outcome_counts_shared_or(events: list[dict], shared_tau: float) -> dict[str, int]:
    return compute_outcome_counts(events, thresholds_from_shared(shared_tau))


def threshold_cache_key(current_thresholds: dict[str, dict[str, float]]) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    return tuple(
        (cam_key, tuple((field, float(current_thresholds[cam_key][field])) for field in FIELDS))
        for cam_key in sorted(current_thresholds)
    )


@lru_cache(maxsize=128)
def cached_results_counts(
    shared_threshold: float,
    thresholds_key: tuple[tuple[str, tuple[tuple[str, float], ...]], ...],
    eval_step: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    events = generate_tolling_events_3field(step=eval_step, n_per_camera=1200)
    current_thresholds = {
        cam_key: {field: value for field, value in field_pairs}
        for cam_key, field_pairs in thresholds_key
    }
    baseline = compute_outcome_counts_weighted(events, thresholds_from_shared_weighted(shared_threshold))
    adaptive = compute_outcome_counts(events, current_thresholds)
    return baseline, adaptive, len(events)


def get_cached_results_counts(
    shared_threshold: float,
    current_thresholds: dict[str, dict[str, float]],
    sim_step: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    return cached_results_counts(
        float(shared_threshold),
        threshold_cache_key(current_thresholds),
        results_eval_step(sim_step),
    )


def policy_objective(counts: dict[str, int], dispute_cost: float = DISPUTE_COST_DEFAULT, review_cost: float = REVIEW_COST_DEFAULT) -> float:
    total = max(1, counts["auto_clear"] + counts["review"])
    required_auto = DEMO_MIN_AUTO_CLEAR_RATE * total
    shortfall = max(0.0, required_auto - counts["auto_clear"])
    return float(
        dispute_cost * counts["type1"]
        + review_cost * counts["review"]
        + DEMO_AUTO_CLEAR_SHORTFALL_COST * shortfall
    )


def compute_camera_counts(events: list[dict], thresholds_camera: dict[str, float]) -> dict[str, int]:
    counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}
    for event in events:
        decision, _ = route_event_with_3field_thresholds(event, thresholds_camera)
        is_correct = bool(event["field_correct"]["lpn"] and event["field_correct"]["lpj"] and event["field_correct"]["lpt"])
        if decision == "auto_clear":
            counts["auto_clear"] += 1
            if not is_correct:
                counts["type1"] += 1
        else:
            counts["review"] += 1
            if is_correct:
                counts["type2"] += 1
    return counts


def simulate_batch_counts_for_policy(
    events: list[dict],
    thresholds: dict[str, dict[str, float]],
    seed: int = 0,
) -> dict[str, int]:
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


def simulate_batch_counts_for_weighted_policy(
    events: list[dict],
    thresholds: dict[str, float],
    seed: int = 0,
) -> dict[str, int]:
    rng = np.random.default_rng(991 + seed)
    counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}
    for event in events:
        event_copy = dict(event)
        cam_key = lane_key(event_copy["lane_id"])
        decision = route_event_weighted(event_copy, thresholds[cam_key])
        label, _source, _correct = label_event_error_type(event_copy, decision, [], rng)
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
    counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}
    for event in events:
        cam_key = lane_key(event["lane_id"])
        decision, _ = route_event_with_3field_thresholds(event, thresholds[cam_key])
        is_correct = bool(event["field_correct"]["lpn"] and event["field_correct"]["lpj"] and event["field_correct"]["lpt"])
        if decision == "auto_clear":
            counts["auto_clear"] += 1
            if not is_correct:
                counts["type1"] += 1
        else:
            counts["review"] += 1
            if is_correct:
                counts["type2"] += 1
    return counts


def compute_outcome_counts_weighted(events: list[dict], thresholds: dict[str, float]) -> dict[str, int]:
    counts = {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}
    for event in events:
        cam_key = lane_key(event["lane_id"])
        decision = route_event_weighted(event, thresholds[cam_key])
        is_correct = bool(event["field_correct"]["lpn"] and event["field_correct"]["lpj"] and event["field_correct"]["lpt"])
        if decision == "auto_clear":
            counts["auto_clear"] += 1
            if not is_correct:
                counts["type1"] += 1
        else:
            counts["review"] += 1
            if is_correct:
                counts["type2"] += 1
    return counts


def evaluate_policy(events: list[dict], thresholds: dict[str, dict[str, float]]) -> dict:
    tp = fp = fn = tn = 0
    per_camera = {lane_key(l["lane_id"]): {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for l in LANES}
    for event in events:
        key = lane_key(event["lane_id"])
        decision, _ = route_event_with_3field_thresholds(event, thresholds[key])
        is_correct = bool(event["field_correct"]["lpn"] and event["field_correct"]["lpj"] and event["field_correct"]["lpt"])
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
    path.parent.mkdir(parents=True, exist_ok=True)
    events = generate_tolling_events_3field(step=step, n_per_camera=n_per_camera)
    baseline_tau_weighted = calibrate_weighted_baseline_threshold(events, TARGET_REVIEW_RATE)
    baseline_tau_or = calibrate_or_baseline_threshold(events, TARGET_REVIEW_RATE)
    baseline_thresholds_weighted = thresholds_from_shared_weighted(baseline_tau_weighted)
    adaptive_thresholds = calibrate_initial_or_thresholds(events, TARGET_REVIEW_RATE)
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
        "LPN_score",
        "LPJ_score",
        "LPT_score",
        "weighted_score",
        "baseline_tau_weighted",
        "baseline_tau_or",
        "LPN_thresh",
        "LPJ_thresh",
        "LPT_thresh",
        "decision_weighted_baseline",
        "decision_or_adaptive",
        "feedback_source",
        "error_label",
        "overall_correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            cam_key = lane_key(event["lane_id"])
            decision_baseline = route_event_weighted(event, baseline_thresholds_weighted[cam_key])
            decision_adaptive, gate_fields_adaptive = route_event_with_3field_thresholds(event, adaptive_thresholds[cam_key])
            label_event_error_type(event, decision_adaptive, gate_fields_adaptive, rng)
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
                    "LPN_score": f"{event['LPN_score']:.3f}",
                    "LPJ_score": f"{event['LPJ_score']:.3f}",
                    "LPT_score": f"{event['LPT_score']:.3f}",
                    "weighted_score": f"{event['weighted_score']:.3f}",
                    "baseline_tau_weighted": f"{baseline_tau_weighted:.3f}",
                    "baseline_tau_or": f"{baseline_tau_or:.3f}",
                    "LPN_thresh": f"{adaptive_thresholds[cam_key]['lpn']:.3f}",
                    "LPJ_thresh": f"{adaptive_thresholds[cam_key]['lpj']:.3f}",
                    "LPT_thresh": f"{adaptive_thresholds[cam_key]['lpt']:.3f}",
                    "decision_weighted_baseline": decision_baseline,
                    "decision_or_adaptive": decision_adaptive,
                    "feedback_source": event["feedback_source"],
                    "error_label": event["error_label"] or "",
                    "overall_correct": int(event["overall_correct"]),
                }
            )
    return path


def graph_card(graph_id: str) -> html.Div:
    return html.Div(className="story-card", children=[dcc.Graph(id=graph_id, config={"displayModeBar": False})])


def make_score_meaning_block() -> html.Div:
    sample = {
        "plate": "MD 7BK2391",
        "lpn_score": 91,
        "lpj_score": 86,
        "lpt_score": 50,
        "lpn_tau": 81.0,
        "lpj_tau": 81.0,
        "lpt_tau": 81.0,
    }
    check_rows = []
    checks = [
        ("LPJ", sample["lpj_score"], sample["lpj_tau"]),
        ("LPN", sample["lpn_score"], sample["lpn_tau"]),
        ("LPT", sample["lpt_score"], sample["lpt_tau"]),
    ]
    for label, score, tau in checks:
        passed = score >= tau
        check_rows.append(
            html.Div(
                className=f"score-check-row field-{label.lower()}",
                children=[
                    html.Div(label, className="score-check-label"),
                    html.Div(f"confidence score: {score}", className="score-check-value"),
                    html.Div(f"threshold {tau:.0f}", className="score-check-thresh"),
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
                "Each score is checked against its own threshold.",
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
                "Routing rule: if any one field fails threshold, the full event is sent to review.",
                className="text-secondary small mt-2 mb-0",
            ),
        ],
    )


def make_arbitrary_cutoff_figure(shared_threshold: float) -> go.Figure:
    """
    Explain cutoff semantics with a score ruler:
    one cutoff line splits the same reads into two routing zones.
    """
    rng = np.random.default_rng(902)
    scores = np.clip(rng.normal(83.1, np.sqrt(1.975), size=120), 45, 99).round(1)
    cutoff = float(shared_threshold)
    x_min = max(1.0, cutoff - 6.0)
    x_max = min(99.0, cutoff + 6.0)
    y = rng.normal(0.0, 0.065, size=len(scores))
    # Simulated truth for explanation: high score can still occasionally be wrong.
    correct_flags = rng.random(len(scores)) < np.clip(0.30 + 0.007 * scores, 0.05, 0.995)
    review_mask = scores < cutoff
    auto_mask = ~review_mask
    wrong_auto_mask = auto_mask & (~correct_flags)
    review_n = int(np.sum(review_mask))
    auto_n = int(np.sum(auto_mask))
    total_n = max(1, review_n + auto_n)
    review_rate = review_n / total_n
    auto_rate = auto_n / total_n

    fig = go.Figure()
    fig.add_vrect(x0=x_min, x1=cutoff, fillcolor="rgba(245,158,11,0.20)", line_width=0, layer="below")
    fig.add_vrect(x0=cutoff, x1=x_max, fillcolor="rgba(22,163,74,0.08)", line_width=0, layer="below")
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
            ax=max(x_min + 0.8, min(x_bad + 3.5, x_max - 0.8)),
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
    fig.add_annotation(x=cutoff, y=0.24, text=f"Cutoff = {cutoff:.1f}", showarrow=False, font=dict(size=11, color="#334155"))
    left_label_x = x_min + 0.5 * (cutoff - x_min)
    right_label_x = cutoff + 0.5 * (x_max - cutoff)
    fig.add_annotation(x=left_label_x, y=0.24, text=f"Review zone: {review_n} reads ({pct(review_rate)})", showarrow=False, font=dict(size=10, color="#b45309"))
    fig.add_annotation(x=right_label_x, y=0.24, text=f"Auto zone: {auto_n} reads ({pct(auto_rate)})", showarrow=False, font=dict(size=10, color="#047857"))

    fig.update_layout(
        title="Cutoff explained: one line splits the score ruler into two decisions",
        height=300,
        margin=dict(l=12, r=12, t=54, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.13, x=0),
    )
    fig.update_xaxes(title="OCR score points (ordinal scale, not probability)", range=[x_min, x_max], dtick=2, gridcolor=PAL_GRID)
    fig.update_yaxes(title="", range=[-0.3, 0.3], visible=False, showgrid=False, zeroline=False)
    return fig


def make_results_comparison_figure(
    shared_threshold: float,
    current_thresholds: dict[str, dict[str, float]],
    sim_step: int,
    dispute_cost: float,
    review_cost: float,
) -> go.Figure:
    baseline_counts, adaptive_counts, _ = get_cached_results_counts(shared_threshold, current_thresholds, sim_step)

    metrics = [
        ("Disputed auto-clear", baseline_counts["type1"], adaptive_counts["type1"]),
        ("Review", baseline_counts["review"], adaptive_counts["review"]),
        ("Auto-clear", baseline_counts["auto_clear"], adaptive_counts["auto_clear"]),
    ]
    labels = [m[0] for m in metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[m[1] for m in metrics],
            name=f"Weighted baseline ({shared_threshold:.1f})",
            marker=dict(color="#6366f1"),
            text=[f"{m[1]:,}" for m in metrics],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Baseline %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[m[2] for m in metrics],
            name="Adaptive thresholds",
            marker=dict(color="#f25c3a"),
            text=[f"{m[2]:,}" for m in metrics],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Adaptive %{y:,}<extra></extra>",
        )
    )

    type1_delta = adaptive_counts["type1"] - baseline_counts["type1"]
    review_delta = adaptive_counts["review"] - baseline_counts["review"]
    objective_delta = policy_objective(adaptive_counts, dispute_cost=dispute_cost, review_cost=review_cost) - policy_objective(
        baseline_counts, dispute_cost=dispute_cost, review_cost=review_cost
    )
    fig.update_layout(
        title=(
            "Business comparison: weighted baseline vs adaptive event policy"
            f"<br><sup>Disputed auto-clear {type1_delta:+d} | Review {review_delta:+d} | Estimated cost ${objective_delta:+.2f}</sup>"
        ),
        height=360,
        margin=dict(l=12, r=20, t=132, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.03, x=0),
        barmode="group",
    )
    ymax = max(max(m[1], m[2]) for m in metrics) * 1.16
    fig.update_yaxes(title="Event count", gridcolor=PAL_GRID, zeroline=False, range=[0, ymax], automargin=True)
    fig.update_xaxes(title="", showgrid=False, automargin=True)
    return fig


def make_technical_comparison_figure(
    current_thresholds: dict[str, dict[str, float]],
    sim_step: int,
    dispute_cost: float,
    review_cost: float,
) -> go.Figure:
    events = generate_tolling_events_3field(step=results_eval_step(sim_step), n_per_camera=1200)
    shared_or_tau = calibrate_or_baseline_threshold(generate_tolling_events_3field(step=0, n_per_camera=1200), TARGET_REVIEW_RATE)
    shared_counts = compute_outcome_counts_shared_or(events, shared_or_tau)
    adaptive_counts = compute_outcome_counts(events, current_thresholds)

    deltas = {
        "Disputed auto-clear": adaptive_counts["type1"] - shared_counts["type1"],
        "Review": adaptive_counts["review"] - shared_counts["review"],
        "Over-review": adaptive_counts["type2"] - shared_counts["type2"],
        "Auto-clear": adaptive_counts["auto_clear"] - shared_counts["auto_clear"],
    }
    metrics = ["Disputed auto-clear", "Review", "Over-review", "Auto-clear"]
    colors = []
    for metric in metrics:
        value = deltas[metric]
        if metric == "Disputed auto-clear":
            colors.append(PAL_GREEN if value < 0 else "#ef4444")
        elif metric == "Auto-clear":
            colors.append(PAL_GREEN if value > 0 else PAL_SLATE)
        else:
            colors.append("#f59e0b" if value > 0 else PAL_GREEN)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[deltas[m] for m in metrics],
            y=metrics,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{deltas[m]:+d}" for m in metrics],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Adaptive - shared OR-gate: %{x:+,} events<extra></extra>",
            showlegend=False,
        )
    )
    type1_delta = adaptive_counts["type1"] - shared_counts["type1"]
    review_delta = adaptive_counts["review"] - shared_counts["review"]
    objective_delta = policy_objective(adaptive_counts, dispute_cost=dispute_cost, review_cost=review_cost) - policy_objective(
        shared_counts, dispute_cost=dispute_cost, review_cost=review_cost
    )
    fig.update_layout(
        title=(
            "Net operational change: adaptive OR-gate vs shared OR-gate"
            f"<br><sup>Disputed auto-clear {type1_delta:+d} | Review {review_delta:+d} | Estimated cost ${objective_delta:+.2f}</sup>"
        ),
        height=360,
        margin=dict(l=12, r=12, t=92, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
    )
    fig.add_vline(x=0, line_color=PAL_SLATE, line_width=1.4)
    fig.update_yaxes(title="", categoryorder="array", categoryarray=list(reversed(metrics)), showgrid=False)
    fig.update_xaxes(title="Change in event count", gridcolor=PAL_GRID, zeroline=False)
    return fig


def make_shared_vs_adaptive_figure(shared_threshold: float, current_thresholds: dict[str, dict[str, float]], sim_step: int) -> go.Figure:
    events = generate_tolling_events_3field(step=max(sim_step, 1), n_per_camera=280)
    shared_metrics = evaluate_policy(events, thresholds_from_shared(shared_threshold))
    adaptive_metrics = evaluate_policy(events, current_thresholds)
    labels = ["Auto-clear", "Disputed auto-clear", "Review", "Utility"]
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
        title="Calibrated baseline vs camera-specific per-field settings (Utility = higher is better)",
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
        ("Disputed auto-clear", shared_metrics["overall"]["bad_auto_rate"], adaptive_metrics["overall"]["bad_auto_rate"]),
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
            name="Calibrated baseline",
            hovertemplate="%{y}<br>Calibrated baseline: %{x:.1%}<extra></extra>",
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
        title="Dumbbell View: Calibrated baseline vs Adaptive",
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
    events = generate_tolling_events_3field(step=results_eval_step(sim_step), n_per_camera=1200)
    shared_counts = compute_outcome_counts_weighted(events, thresholds_from_shared_weighted(shared_threshold))
    adaptive_counts = compute_outcome_counts(events, current_thresholds)

    delta = {
        "Disputed auto-clear": adaptive_counts["type1"] - shared_counts["type1"],
        "Review": adaptive_counts["review"] - shared_counts["review"],
        "Over-review": adaptive_counts["type2"] - shared_counts["type2"],
        "Auto-clear": adaptive_counts["auto_clear"] - shared_counts["auto_clear"],
    }
    metrics = ["Disputed auto-clear", "Review", "Over-review", "Auto-clear"]
    colors = []
    for metric in metrics:
        val = delta[metric]
        if metric == "Disputed auto-clear":
            colors.append(PAL_GREEN if val <= 0 else "#dc2626")
        elif metric in {"Review", "Over-review"}:
            colors.append("#f59e0b" if val > 0 else PAL_GREEN)
        else:
            colors.append(PAL_SLATE)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=[delta[m] for m in metrics],
            marker=dict(color=colors),
            text=[f"{int(d):+d}" for d in [delta[m] for m in metrics]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Adaptive - baseline: %{y:+.0f} events<extra></extra>",
            name="Net change in events",
        )
    )
    fig.add_hline(y=0, line_width=1.4, line_color=PAL_SLATE)
    fig.update_layout(
        title="Tradeoff view: adaptive event policy vs weighted baseline",
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
                    html.Td(f"{event['LPN_score']:.1f}"),
                    html.Td(f"{event['LPJ_score']:.1f}"),
                    html.Td(f"{event['LPT_score']:.1f}"),
                    html.Td("Review" if event["decision"] == "review" else "Auto-clear"),
                    html.Td(str(event["feedback_source"]).title()),
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


def make_error_count_kpis(adaptive_counts: dict, fixed_counts: dict, sim_step: int, baseline_tau: float) -> html.Div:
    adaptive_tiles = [
        ("Step", str(sim_step)),
        ("Auto-cleared", str(int(adaptive_counts["auto_clear"]))),
        ("Routed to review", str(int(adaptive_counts["review"]))),
        ("Type I", str(int(adaptive_counts["type1"]))),
        ("Type II", str(int(adaptive_counts["type2"]))),
    ]
    fixed_tiles = [
        ("Baseline cutoff", f"{baseline_tau:.1f}"),
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
                children=[panel("Simulated under calibrated shared baseline", fixed_tiles)],
            ),
        ],
    )


def make_type1_priority_note(dispute_cost: float, review_cost: float) -> html.Div:
    ratio = dispute_cost / max(review_cost, 1e-9)
    if ratio >= 3.0:
        policy = "Disputes are priced much higher than manual review, so the search favors safer thresholds."
    elif ratio <= 1.5:
        policy = "Manual review is priced closer to dispute cost, so the search tolerates more auto-clears."
    else:
        policy = "The demo cost assumptions balance dispute reduction against review spend."
    return html.Div(
        className="story-card mt-3",
        children=[
            html.H6("Demo cost assumptions in the threshold search", className="mb-2"),
            html.Div(
                f"Current assumptions: dispute cost=${dispute_cost:.2f}, manual review cost=${review_cost:.2f}",
                className="small text-secondary mb-1",
            ),
            html.Div(
                "For each camera-field pair, the demo searches candidate thresholds and picks the lowest-cost option that still keeps auto-clear as the majority outcome.",
                className="small text-secondary mb-1",
            ),
            html.Div(policy, className="small text-secondary mb-0"),
        ],
    )


def make_main_math_card() -> html.Div:
    return html.Div(
        className="story-card mb-3",
        children=[
            html.H6("How the machine learns in the demo", className="mb-2"),
            dcc.Markdown(
                mathjax=True,
                className="small text-secondary mb-0",
                children=r"""
For each camera $c$ and field $f \in \{\mathrm{LPN}, \mathrm{LPJ}, \mathrm{LPT}\}$, the adaptive policy keeps its own threshold $\tau_{c,f}^{(t)}$ at update step $t$.

For each event $e$, the OCR system emits field-level score signals $s_{e,c,f}$. The routing rule is an event-level OR-gate:

$$
\mathrm{review}_e \ \text{if any } s_{e,c,f} < \tau_{c,f}^{(t)}
$$

After a batch arrives, only a subset of events becomes labeled through review outcomes or customer disputes. Let $\mathcal{L}_{c,f}^{(t)}$ denote the labeled sample available for camera-field pair $(c,f)$ at step $t$.

The update is a constrained empirical-risk minimization over a discrete candidate set $\mathcal{T}_{c,f}^{(t)}$. For each candidate threshold $\tilde{\tau}$, the demo computes the estimated operational loss:

$$
\widehat{C}_{c,f}^{(t)}(\tilde{\tau}) =
c_d \cdot \widehat{N}_{\mathrm{dispute},c,f}^{(t)}(\tilde{\tau}) +
c_r \cdot \widehat{N}_{\mathrm{review},c,f}^{(t)}(\tilde{\tau}) +
\lambda \cdot \max\!\bigl(0,\ \rho^\star - \widehat{\rho}_{\mathrm{auto}}^{(t)}(\tilde{\tau})\bigr)
$$

where:

- $c_d$ is the assumed dispute cost,
- $c_r$ is the assumed manual-review cost,
- $\widehat{N}_{\mathrm{dispute},c,f}^{(t)}$ is the estimated number of disputed auto-clears under $\tilde{\tau}$,
- $\widehat{N}_{\mathrm{review},c,f}^{(t)}$ is the estimated review volume under $\tilde{\tau}$,
- $\widehat{\rho}_{\mathrm{auto}}^{(t)}$ is the implied auto-clear rate,
- $\rho^\star$ is the demo majority auto-clear target,
- and $\lambda$ is a penalty weight that discourages collapsing into an all-review policy.

The threshold update is then:

$$
\tau_{c,f}^{(t+1)} = \underset{\tilde{\tau} \in \mathcal{T}_{c,f}^{(t)}}{\arg\min}\ \widehat{C}_{c,f}^{(t)}(\tilde{\tau})
$$

Statistically, this is an online policy update based on partial labels and batchwise empirical cost estimation. The OCR model itself is not retrained; only the decision threshold policy is re-estimated.

In this demo, the assumed costs are $c_d = \$0.20$ per disputed auto-clear and $c_r = \$0.07$ per manual review.

For the staged degradation scenario, the demo also applies a short shock-response rule so threshold movement is visibly aligned with the highlighted quality event. That visual aid is layered on top of the same cost-minimizing update logic.
""",
            ),
        ],
    )


def make_machine_architecture_card() -> html.Div:
    stages = [
        ("1. Feature extraction", "Each read becomes a small feature vector: LPN, LPJ, and LPT score outputs from the OCR stack."),
        ("2. Decision policy", "An event-level OR-gate applies camera-field thresholds and classifies the read as auto-clear or review."),
        ("3. Feedback labeling", "Reviewed reads and customer disputes create delayed labels that estimate Type I and review-cost pressure."),
        ("4. Online optimization", "The threshold search updates each camera-field policy to minimize expected dispute and review cost over time."),
    ]
    return html.Div(
        className="story-card mb-3",
        children=[
            html.H6("What the machine does", className="mb-3"),
            html.Div(
                className="row g-3",
                children=[
                    html.Div(
                        className="col-12 col-md-6 col-xl-3",
                        children=[
                            html.Div(
                                className="kpi-mini h-100",
                                children=[
                                    html.Div(title, className="kpi-mini-label"),
                                    html.Div(text, className="small text-secondary"),
                                ],
                            )
                        ],
                    )
                    for title, text in stages
                ],
            ),
        ],
    )


def make_results_kpi_board(
    shared_threshold: float,
    current_thresholds: dict[str, dict[str, float]],
    sim_step: int,
    dispute_cost: float,
    review_cost: float,
) -> html.Div:
    kpi = compute_results_kpis(shared_threshold, current_thresholds, sim_step, dispute_cost, review_cost)
    baseline_type1 = int(kpi["baseline_type1"])
    adaptive_type1 = int(kpi["adaptive_type1"])
    type1_delta = int(kpi["type1_delta"])
    type1_delta_rate = float(kpi["type1_delta_rate"])
    review_lift_rate = float(kpi["review_lift_rate"])
    review_lift_count = int(kpi["review_lift_count"])
    objective_delta = float(kpi["objective_delta"])
    return html.Div(
        className="row g-3 mb-3",
        children=[
            html.Div(
                className="col-12 col-md-4",
                children=[
                    html.Div(
                        className="story-card kpi-panel",
                        children=[
                            html.Div("Primary outcome", className="kpi-mini-label"),
                            html.Div(f"Baseline {baseline_type1} -> Adaptive {adaptive_type1}", className="kpi-mini-label"),
                            html.Div(f"{type1_delta:+d}", className=f"kpi-mini-value {'text-success' if type1_delta < 0 else 'text-danger'}"),
                            html.Div(f"{type1_delta_rate:+.1%} vs baseline", className="kpi-mini-label"),
                        ],
                    )
                ],
            ),
            html.Div(
                className="col-12 col-md-4",
                children=[
                    html.Div(
                        className="story-card kpi-panel",
                        children=[
                            html.Div("Operational tradeoff", className="kpi-mini-label"),
                            html.Div("Review lift", className="kpi-mini-label"),
                            html.Div(f"{review_lift_rate:+.1%} ({review_lift_count:+d})", className=f"kpi-mini-value {'text-success' if review_lift_rate <= REVIEW_LIFT_CAP_RATE else 'text-warning'}"),
                        ],
                    )
                ],
            ),
            html.Div(
                className="col-12 col-md-4",
                children=[
                    html.Div(
                        className="story-card kpi-panel",
                        children=[
                            html.Div("Estimated cost delta", className="kpi-mini-label"),
                            html.Div(f"Using ${dispute_cost:.2f} dispute / ${review_cost:.2f} review", className="kpi-mini-label"),
                            html.Div(f"${objective_delta:+.2f}", className=f"kpi-mini-value {'text-success' if objective_delta < 0 else 'text-danger'}"),
                        ],
                    )
                ],
            ),
        ],
    )


def make_adaptive_over_time_figure(time_history: dict) -> go.Figure:
    steps = time_history.get("step", [])
    if not steps:
        fig = go.Figure()
        fig.update_layout(
            title="Adaptive response over time",
            height=340,
            margin=dict(l=12, r=12, t=56, b=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=PAL_BG,
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=time_history.get("adaptive_type1", []),
            mode="lines+markers",
            name="Adaptive disputed auto-clear",
            line=dict(color=PAL_GREEN, width=3),
            marker=dict(size=6),
            hovertemplate="Step %{x}<br>Adaptive disputes %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=time_history.get("baseline_type1", []),
            mode="lines+markers",
            name="Baseline disputed auto-clear",
            line=dict(color=PAL_ORANGE, width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate="Step %{x}<br>Baseline disputes %{y}<extra></extra>",
        )
    )
    adaptive_vals = time_history.get("adaptive_type1", [])
    baseline_vals = time_history.get("baseline_type1", [])
    latest_gap = 0
    if adaptive_vals and baseline_vals:
        latest_gap = int(adaptive_vals[-1] - baseline_vals[-1])
    fig.update_layout(
        title=f"Disputed auto-clears over update steps<br><sup>Latest gap vs baseline: {latest_gap:+d}</sup>",
        height=360,
        margin=dict(l=12, r=20, t=104, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.04, x=0),
    )
    fig.update_yaxes(title="Disputed auto-clears", gridcolor=PAL_GRID, automargin=True)
    fig.update_xaxes(title="Update step", dtick=1, gridcolor=PAL_GRID, automargin=True)
    return fig


def policy_from_time_step(time_step: int, dispute_rate: float) -> tuple[str, float, float, str]:
    if time_step < 4:
        stage_label = "Initial calibration"
        base_rate = 0.20
    elif time_step < 9:
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


def make_audit_rate_over_time_figure(max_step: int = 10, current_step: int = 0) -> go.Figure:
    steps = list(range(max_step + 1))
    # Synthetic dispute signal pattern to illustrate occasional escalation.
    dispute_series = [0.010 + (0.018 if s in {6, 7, 8} else 0.0) for s in steps]
    rates = [policy_from_time_step(s, d)[2] for s, d in zip(steps, dispute_series)]
    current_step = max(0, min(int(current_step), max_step))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps[: current_step + 1],
            y=rates[: current_step + 1],
            mode="lines",
            line=dict(color=PAL_BLUE, width=3),
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
    fig.add_vrect(x0=0, x1=3.5, fillcolor="rgba(239,90,60,0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=3.5, x1=8.5, fillcolor="rgba(99,102,241,0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=8.5, x1=max_step, fillcolor="rgba(16,185,129,0.08)", line_width=0, layer="below")
    fig.add_annotation(x=1.75, y=0.235, text="Calibration", showarrow=False, font=dict(size=10, color="#9a3412"))
    fig.add_annotation(x=6.0, y=0.235, text="Transition", showarrow=False, font=dict(size=10, color="#4338ca"))
    fig.add_annotation(x=9.25, y=0.235, text="Steady state", showarrow=False, font=dict(size=10, color="#047857"))

    fig.update_layout(
        title="Recommended audit rate over time",
        height=300,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
        legend=dict(orientation="h", y=1.14, x=0),
        updatemenus=[],
    )
    fig.update_xaxes(title="Time step", range=[0, max_step], dtick=1, gridcolor=PAL_GRID)
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
            cells.append(html.Td(f"{tau:.1f} ({delta:+.2f})"))
        rows.append(html.Tr(cells))

    return html.Div(
        className="story-card",
        children=[
            html.H6("Per-camera threshold matrix (LPN / LPJ / LPT)", className="mb-2"),
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
    x = list(range(len(threshold_store["history"][key]["lpn"])))
    baseline_tau = threshold_store.get("baseline_tau_or", THRESHOLD_DEFAULT)
    color_map = {"lpn": PAL_BLUE, "lpj": PAL_ORANGE, "lpt": PAL_GREEN}
    for field in FIELDS:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=threshold_store["history"][key][field],
                mode="lines+markers",
                name=FIELD_LABELS[field],
                line=dict(width=2.5, color=color_map[field]),
                marker=dict(size=5),
            )
        )
    fig.add_hline(y=baseline_tau, line_color=PAL_ORANGE, line_dash="dash", line_width=1.8)

    # Zoom y-axis around observed values so drift is visible in demos.
    all_vals = []
    for field in FIELDS:
        all_vals.extend([float(v) for v in threshold_store["history"][key][field]])
    if not all_vals:
        all_vals = [baseline_tau]
    y_min_obs = min(min(all_vals), baseline_tau)
    y_max_obs = max(max(all_vals), baseline_tau)
    span = y_max_obs - y_min_obs
    pad = max(0.6, span * 0.20)
    y_min = max(THRESHOLD_MIN, y_min_obs - pad)
    y_max = min(THRESHOLD_MAX, y_max_obs + pad)

    fig.update_layout(
        title=f"Threshold change by field - {next(l['camera'] for l in LANES if l['lane_id'] == lane_id)}",
        height=320,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        legend=dict(orientation="h", y=1.13, x=0),
    )
    for shock in shocks_for_lane(lane_id):
        display_start = shock.get("display_start", shock["start"])
        display_end = shock.get("display_end", shock["end"])
        fig.add_vrect(
            x0=display_start - 0.5,
            x1=display_end - 0.5,
            fillcolor="rgba(239,90,60,0.08)",
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=(display_start + display_end) / 2.0,
            y=y_max,
            yshift=-10,
            text=shock["label"],
            showarrow=False,
            font=dict(size=10, color="#b45309"),
            bgcolor="rgba(255,255,255,0.75)",
        )
    fig.update_xaxes(title="Update steps", gridcolor="#dbe4ec")
    fig.update_yaxes(title="Threshold (score points)", range=[y_min, y_max], gridcolor="#dbe4ec")
    return fig


def compute_field_health_records(threshold_store: dict, feedback_store: dict) -> list[dict]:
    records: list[dict] = []
    window_start = 4
    window_end = 8
    for lane in LANES:
        key = lane_key(lane["lane_id"])
        for field in FIELDS:
            hist = threshold_store["history"][key][field]
            current_tau = float(threshold_store["current"][key][field])
            start_tau = float(hist[0]) if hist else current_tau
            drift = current_tau - start_tau
            window = hist[-5:] if len(hist) >= 5 else hist
            recent_drift = (window[-1] - window[0]) if len(window) >= 2 else 0.0
            peak_change = 0.0
            peak_start = window_start
            peak_end = window_end
            if hist:
                start_idx = min(window_start, len(hist) - 1)
                end_idx = min(window_end, len(hist) - 1)
                peak_change = float(hist[end_idx] - hist[start_idx])

            rec = feedback_store["per_camera_field"][key][field]
            labeled = int(rec.get("labeled", 0))
            type1 = int(rec.get("type1", 0))
            type1_rate = (type1 / labeled) if labeled > 0 else 0.0

            sev_drift = float(np.clip(drift / 2.0, 0.0, 1.0))
            sev_recent = float(np.clip(recent_drift / 1.0, 0.0, 1.0))
            sev_type1 = float(np.clip((type1_rate - 0.05) / 0.20, 0.0, 1.0))
            score = 100.0 * (0.45 * sev_drift + 0.25 * sev_recent + 0.30 * sev_type1)

            demo_score = DEMO_HEALTH_SCORE_MAP.get((lane["camera"], field))
            if demo_score is not None:
                score = demo_score
            if lane["camera"] == DEMO_DEGRADATION_CAMERA and field == DEMO_DEGRADATION_FIELD:
                score = max(score, DEMO_DEGRADATION_MIN_SCORE)
                drift = max(drift, 1.8)
                type1_rate = max(type1_rate, 0.18)
                peak_change = max(peak_change, 8.0)
                peak_start = window_start
                peak_end = window_end

            if labeled < 10:
                status = "Low data"
            elif score >= 65:
                status = "Failing"
            elif score >= 35:
                status = "Watch"
            else:
                status = "Stable"

            records.append(
                {
                    "camera_id": lane["camera"],
                    "camera_label": f"{lane['camera']} ({lane['gantry']})",
                    "field": field,
                    "field_label": FIELD_LABELS[field],
                    "score": score,
                    "status": status,
                    "drift": drift,
                    "recent_drift": recent_drift,
                    "peak_change": peak_change,
                    "peak_start": peak_start,
                    "peak_end": peak_end,
                    "window_label": f"{window_start}-{window_end}",
                    "type1_rate": type1_rate,
                    "labeled": labeled,
                }
            )
    return records


def make_field_health_summary(records: list[dict]) -> html.Div:
    failing = sum(1 for r in records if r["status"] == "Failing")
    watch = sum(1 for r in records if r["status"] == "Watch")
    low_data = sum(1 for r in records if r["status"] == "Low data")
    return html.Div(
        className="story-card mb-3",
        children=[
            html.Div(
                f"Fields flagged Failing: {failing} | Watch: {watch} | Low data: {low_data}",
                className="fw-semibold mb-0",
            ),
        ],
    )


def make_field_health_cards(records: list[dict]) -> html.Div:
    by_camera: dict[str, list[dict]] = {}
    for rec in records:
        by_camera.setdefault(rec["camera_id"], []).append(rec)

    cam_cards = []
    for lane in LANES:
        camera = lane["camera"]
        recs = sorted(by_camera.get(camera, []), key=lambda r: ("lpj", "lpn", "lpt").index(r["field"]))
        rows = []
        for rec in recs:
            rows.append(
                html.Tr(
                    [
                        html.Td(rec["field_label"]),
                        html.Td(f"{rec['peak_change']:+.1f}"),
                        html.Td(f"{rec['type1_rate']:.0%}"),
                    ],
                    style=(
                        {"background": "rgba(239,90,60,0.08)"}
                        if rec["camera_id"] == DEMO_DEGRADATION_CAMERA and rec["field"] == DEMO_DEGRADATION_FIELD
                        else {}
                    ),
                )
            )
        cam_cards.append(
            html.Div(
                className="col-12 col-xl-4",
                children=[
                    html.Div(
                        className="story-card h-100",
                        children=[
                            html.H6(f"{camera} field health", className="mb-2"),
                            html.Div(
                                className="table-responsive",
                                children=[
                                    html.Table(
                                        className="table table-sm align-middle mb-0",
                                        children=[
                                            html.Thead(
                                                html.Tr(
                                                    [
                                                        html.Th("Field"),
                                                        html.Th("Window change"),
                                                        html.Th("Type I"),
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
                ],
            )
        )
    return html.Div(className="row g-3", children=cam_cards)


def make_field_health_heatmap(records: list[dict]) -> go.Figure:
    x_fields = [FIELD_LABELS[f] for f in ("lpj", "lpn", "lpt")]
    y_cams = [lane["camera"] for lane in LANES]
    score_map = {(r["camera_id"], r["field_label"]): abs(r["peak_change"]) for r in records}
    window_map = {(r["camera_id"], r["field_label"]): r["window_label"] for r in records}
    z = [[score_map.get((cam, field), 0.0) for field in x_fields] for cam in y_cams]

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x_fields,
                y=y_cams,
                z=z,
                zmin=0,
                zmax=max(8.0, max((abs(r["peak_change"]) for r in records), default=8.0)),
                colorscale=[[0.0, "#10b981"], [0.45, "#f59e0b"], [1.0, "#ef4444"]],
                colorbar=dict(title="Risk"),
                customdata=[[window_map.get((cam, field), "0-0") for field in x_fields] for cam in y_cams],
                hovertemplate="Camera %{y}<br>Field %{x}<br>Window %{customdata}<br>Window change %{z:.1f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Threshold change during steps 4-8",
        height=300,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAL_BG,
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    return fig


app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.title = "OCR Threshold Optimization"

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
                                html.P([html.Strong("Business risk:"), " Wrong auto-clears create customer disputes, corrections, and revenue/compliance exposure."], className="text-secondary mb-2"),
                                html.P([html.Strong("Current standard:"), " one shared weighted-mean cutoff decides auto-clear vs review."], className="text-secondary mb-2"),
                                html.P([html.Strong("New method:"), " adaptive per-camera, per-field OR-gate thresholds tuned to reduce disputes while monitoring review lift."], className="text-secondary mb-0"),
                            ],
                        ),
                        html.Div(
                            className="story-card mt-3",
                            children=[
                                html.H6("Hypothesis", className="mb-1"),
                                html.P(
                                    "Compared to the weighted baseline, adaptive OR-gate thresholds should lower disputed auto-clears first, "
                                    "with review growth kept within an acceptable operating range.",
                                    className="text-secondary mb-0",
                                ),
                            ],
                        ),
                        make_score_meaning_block(),
                        html.Div(
                            className="story-card narrative-caption-card mt-3",
                            children=[
                                dcc.Graph(id="arbitrary-cutoff-graph", config={"displayModeBar": False}),
                                html.Div("A cutoff is one line on a score distribution, not a probability of correctness.", className="narrative-caption"),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="story-section mb-5",
                    children=[
                        html.Div("Machine In Action", className="section-kicker"),
                        html.H2("New method: three-machine OR-gate with adaptive per-field thresholds", className="mb-3"),
                        html.Div(
                            className="story-card mb-3",
                            children=[
                                html.H6("How the updating works", className="mb-2"),
                                html.Div(
                                    className="text-secondary small",
                                    children=[
                                        html.Div("Each read is converted into three OCR features: LPN, LPJ, and LPT scores."),
                                        html.Div("A decision policy applies per-camera, per-field thresholds and routes the event with an OR-gate."),
                                        html.Div("New labeled feedback is used as an online learning signal to re-estimate lower-cost thresholds."),
                                    ],
                                ),
                            ],
                        ),
                        make_machine_architecture_card(),
                        make_main_math_card(),
                        html.Div(
                            className="story-card mb-3",
                            children=[
                                html.Div(
                                    className="row g-3 align-items-end",
                                    children=[
                                        html.Div(
                                            className="col-12 col-lg-4",
                                            children=[
                                                html.Label("Standard method", className="mb-2 fw-semibold"),
                                                html.Div(id="baseline-readout", className="threshold-readout mt-2"),
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
                                                        html.Button("Skip 10 steps", id="skip-10-steps-btn", className="btn btn-outline-primary w-100"),
                                                        html.Button("Reset", id="reset-3field-btn", className="btn btn-outline-secondary w-100"),
                                                    ],
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="mt-3",
                                    children=[
                                        html.Label("Demo dispute cost assumption", htmlFor="type1-priority-slider", className="mb-2 fw-semibold"),
                                        dcc.Slider(
                                            id="type1-priority-slider",
                                            min=0.05,
                                            max=0.50,
                                            step=0.01,
                                            value=DISPUTE_COST_DEFAULT,
                                            marks={0.05: "$0.05", 0.10: "$0.10", 0.20: "$0.20", 0.35: "$0.35", 0.50: "$0.50"},
                                        ),
                                        html.Div(id="type1-priority-readout", className="threshold-readout mt-2"),
                                    ],
                                ),
                                html.Div(
                                    className="mt-3",
                                    children=[
                                        html.Label("Demo manual review cost assumption", htmlFor="type2-priority-slider", className="mb-2 fw-semibold"),
                                        dcc.Slider(
                                            id="type2-priority-slider",
                                            min=0.01,
                                            max=0.20,
                                            step=0.01,
                                            value=REVIEW_COST_DEFAULT,
                                            marks={0.01: "$0.01", 0.05: "$0.05", 0.07: "$0.07", 0.10: "$0.10", 0.15: "$0.15", 0.20: "$0.20"},
                                        ),
                                        html.Div(id="type2-priority-readout", className="threshold-readout mt-2"),
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
                        html.Div("Monitoring", className="section-kicker"),
                        html.H2("Failure detection over time (steps 4-8)", className="mb-3"),
                        html.Div(id="field-health-cards", className="mb-3"),
                        html.Div(className="row g-3", children=[html.Div(className="col-12", children=[graph_card("field-health-heatmap")])]),
                    ],
                ),
                html.Section(
                    className="story-section mb-5",
                    children=[
                        html.Div("Results", className="section-kicker"),
                        html.H2("Results: fewer disputed auto-clears, with review lift managed as the tradeoff", className="mb-3"),
                        html.Div(id="results-kpi-board"),
                        html.Div(
                            className="row g-3",
                            children=[
                                html.Div(className="col-12 col-xl-6", children=[graph_card("fixed-threshold-camera-outcomes")]),
                                html.Div(className="col-12 col-xl-6", children=[graph_card("adaptive-over-time-graph")]),
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
- $s_{e,f}$: OCR score points (ordinal, not probability)
""")]),
                                html.Div(className="story-card mb-3", children=[dcc.Markdown(mathjax=True, children=r"""
#### 2) Routing Rule (Operational Decision)

Standard baseline (original implementation) uses weighted mean:

$$
s^{(w)}_e = 0.50\,s_{e,\mathrm{LPN}} + 0.35\,s_{e,\mathrm{LPJ}} + 0.15\,s_{e,\mathrm{LPT}}
$$

$$
r^{\mathrm{base}}_e = \mathbf{1}\!\left[s^{(w)}_e < \tau_{\mathrm{base}}\right]
$$

New method uses per-field thresholds per camera:

$$
\tau_{c,\mathrm{LPN}},\ \tau_{c,\mathrm{LPJ}},\ \tau_{c,\mathrm{LPT}}
$$

Route to review if any one field fails:

$$
r_e = \mathbf{1}\!\left[(s_{e,\mathrm{LPN}}<\tau_{c,\mathrm{LPN}})\ \lor\ (s_{e,\mathrm{LPJ}}<\tau_{c,\mathrm{LPJ}})\ \lor\ (s_{e,\mathrm{LPT}}<\tau_{c,\mathrm{LPT}})\right]
$$

$$
a_e = 1-r_e
$$

Baseline $\tau_{\mathrm{base}}$ is calibrated to about 10% review load.
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
#### 4) Cost-Minimizing Threshold Search (Per Camera, Per Field)

For camera $c$, field $f$, step $t$:

$$
\tau_{c,f}^{*}
=
\arg\min_{\tau}
\left(
c_d\cdot \mathrm{TypeI}_{c,f}(\tau)
 c_r\cdot \mathrm{Review}_{c,f}(\tau)
\right)
$$

where:

- $c_d$: dispute cost
- $c_r$: manual review cost
- the demo searches candidate thresholds and chooses the minimum-cost option
""")]),
                                html.Div(className="story-card", children=[dcc.Markdown(mathjax=True, children=r"""
#### 5) Standard Method vs Adaptive Method

Standard method:

$$
\tau_{\mathrm{base}}\ \text{on weighted mean}\ s^{(w)}_e
$$

Adaptive method:

$$
\tau_{c,f}^{(t)}\ \text{updates by camera and field over time (OR-gate routing)}
$$

Resulting objective tradeoff:

$$
\min_{\tau}\ \Big(c_d\cdot \mathrm{DisputedAutoClears}(\tau) + c_r\cdot \mathrm{ReviewVolume}(\tau)\Big)
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


def advance_demo_step(
    threshold_store: dict,
    feedback_store: dict,
    sim_step: int,
    dispute_cost: float,
    review_cost: float,
) -> tuple[dict, dict, int]:
    thresholds = deepcopy(threshold_store["current"])
    new_store = deepcopy(threshold_store)
    new_feedback = deepcopy(feedback_store)
    total_delta = {lane_key(l["lane_id"]): {field: 0.0 for field in FIELDS} for l in LANES}
    prev_score_means = deepcopy(
        new_feedback.get(
            "prev_score_means",
            {lane_key(l["lane_id"]): {field: None for field in FIELDS} for l in LANES},
        )
    )

    sim_step = int(sim_step or 0) + 1
    events = generate_tolling_events_3field(step=sim_step, n_per_camera=BATCH_SIZE_PER_CAMERA)
    rng = np.random.default_rng(777 + sim_step)
    for event in events:
        cam_key = lane_key(event["lane_id"])
        decision, gate_fields = route_event_with_3field_thresholds(event, thresholds[cam_key])
        label_event_error_type(event, decision, gate_fields, rng)

    per_camera, per_camera_field, batch_counts = aggregate_feedback_3field(events)
    for lane in LANES:
        key = lane_key(lane["lane_id"])
        for metric in ["type1", "type2", "correct", "review", "dispute", "auto_clear", "routed_review", "labeled"]:
            new_feedback["per_camera"][key][metric] += per_camera[key][metric]
        for field in FIELDS:
            for metric in ["labeled", "type1", "type2"]:
                new_feedback["per_camera_field"][key][field][metric] += per_camera_field[key][field][metric]

    proposed_thresholds, _ = update_thresholds_3field(
        thresholds,
        events,
        sim_step=sim_step,
        dispute_cost=dispute_cost,
        review_cost=review_cost,
        prev_score_means=prev_score_means,
    )
    for key in thresholds:
        for field in FIELDS:
            total_delta[key][field] = float(proposed_thresholds[key][field] - thresholds[key][field])
    thresholds = proposed_thresholds

    for lane in LANES:
        key = lane_key(lane["lane_id"])
        for field in FIELDS:
            field_scores = [e[f"{FIELD_LABELS[field]}_score"] for e in events if lane_key(e["lane_id"]) == key]
            mean_score = float(np.mean(field_scores)) if field_scores else thresholds[key][field]
            new_store["history"][key][field].append(thresholds[key][field])
            if len(new_store["history"][key][field]) > 40:
                new_store["history"][key][field] = new_store["history"][key][field][-40:]
            prev_score_means[key][field] = mean_score

    baseline_counts = compute_outcome_counts_weighted(events, thresholds_from_shared_weighted(new_store["baseline_tau_weighted"]))
    adaptive_counts = compute_outcome_counts(events, thresholds)
    history = new_feedback["time_history"]
    history["step"].append(sim_step)
    history["adaptive_type1"].append(adaptive_counts["type1"])
    history["adaptive_review"].append(adaptive_counts["review"])
    history["adaptive_cost"].append(
        policy_objective(adaptive_counts, dispute_cost=dispute_cost, review_cost=review_cost)
    )
    history["baseline_type1"].append(baseline_counts["type1"])
    history["baseline_review"].append(baseline_counts["review"])
    history["baseline_cost"].append(
        policy_objective(baseline_counts, dispute_cost=dispute_cost, review_cost=review_cost)
    )
    for metric in history:
        if len(history[metric]) > 40:
            history[metric] = history[metric][-40:]

    new_store["current"] = thresholds
    new_store["last_delta"] = total_delta
    new_feedback["prev_score_means"] = prev_score_means
    new_feedback["last_batch_events"] = events
    new_feedback["last_batch_counts"] = batch_counts
    return new_store, new_feedback, sim_step


@lru_cache(maxsize=32)
def build_precomputed_demo_path(dispute_cost_cents: int, review_cost_cents: int) -> tuple[tuple[dict, dict], ...]:
    dispute_cost = dispute_cost_cents / 100.0
    review_cost = review_cost_cents / 100.0
    threshold_store = initial_threshold_store_3field()
    feedback_store = initial_feedback_store_3field()
    sim_step = 0
    path: list[tuple[dict, dict]] = [(deepcopy(threshold_store), deepcopy(feedback_store))]
    for _ in range(DEMO_PRECOMPUTED_MAX_STEP):
        threshold_store, feedback_store, sim_step = advance_demo_step(
            threshold_store,
            feedback_store,
            sim_step,
            dispute_cost,
            review_cost,
        )
        path.append((deepcopy(threshold_store), deepcopy(feedback_store)))
    return tuple(path)


@app.callback(
    Output("store-thresholds-3field", "data"),
    Output("store-feedback-3field", "data"),
    Output("store-sim-step-3field", "data"),
    Input("run-step-3field-btn", "n_clicks"),
    Input("skip-10-steps-btn", "n_clicks"),
    Input("reset-3field-btn", "n_clicks"),
    Input("type1-priority-slider", "value"),
    Input("type2-priority-slider", "value"),
    State("store-thresholds-3field", "data"),
    State("store-feedback-3field", "data"),
    State("store-sim-step-3field", "data"),
    prevent_initial_call=True,
)
def step_machine(
    _run_clicks: int | None,
    _skip_clicks: int | None,
    _reset_clicks: int | None,
    _dispute_cost_input: float | None,
    _review_cost_input: float | None,
    threshold_store: dict,
    feedback_store: dict,
    sim_step: int,
):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    dispute_cost = float(_dispute_cost_input if _dispute_cost_input is not None else DISPUTE_COST_DEFAULT)
    review_cost = float(_review_cost_input if _review_cost_input is not None else REVIEW_COST_DEFAULT)
    cost_key = (int(round(dispute_cost * 100)), int(round(review_cost * 100)))
    demo_path = build_precomputed_demo_path(*cost_key)

    if trigger == "reset-3field-btn":
        threshold_store_0, feedback_store_0 = demo_path[0]
        return deepcopy(threshold_store_0), deepcopy(feedback_store_0), 0
    if trigger in {"type1-priority-slider", "type2-priority-slider"}:
        threshold_store_0, feedback_store_0 = demo_path[0]
        return deepcopy(threshold_store_0), deepcopy(feedback_store_0), 0
    if trigger not in {"run-step-3field-btn", "skip-10-steps-btn"}:
        return no_update, no_update, no_update
    current_step = int(sim_step or 0)
    step_jump = 10 if trigger == "skip-10-steps-btn" else 1
    next_step = min(current_step + step_jump, DEMO_PRECOMPUTED_MAX_STEP)
    next_threshold_store, next_feedback_store = demo_path[next_step]
    return deepcopy(next_threshold_store), deepcopy(next_feedback_store), next_step


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
    Output("baseline-readout", "children"),
    Output("type1-priority-readout", "children"),
    Output("type2-priority-readout", "children"),
    Output("machine-action-table", "children"),
    Output("error-count-kpis", "children"),
    Output("results-kpi-board", "children"),
    Output("adaptive-over-time-graph", "figure"),
    Output("field-health-cards", "children"),
    Output("field-health-heatmap", "figure"),
    Output("audit-policy-note", "children"),
    Output("audit-rate-over-time-graph", "figure"),
    Output("threshold-matrix-table", "children"),
    Output("field-threshold-trend-graph", "figure"),
    Input("trend-camera", "value"),
    Input("store-thresholds-3field", "data"),
    Input("store-feedback-3field", "data"),
    Input("store-sim-step-3field", "data"),
    Input("type1-priority-slider", "value"),
    Input("type2-priority-slider", "value"),
)
def render_story(
    trend_camera: str,
    threshold_store: dict,
    feedback_store: dict,
    sim_step: int,
    dispute_cost: float | None,
    review_cost: float | None,
):
    threshold_store = threshold_store or initial_threshold_store_3field()
    feedback_store = feedback_store or initial_feedback_store_3field()
    sim_step = int(sim_step or 0)
    shared_threshold = float(threshold_store.get("baseline_tau_weighted", threshold_store.get("baseline_tau_or", THRESHOLD_DEFAULT)))
    dispute_cost = float(dispute_cost if dispute_cost is not None else DISPUTE_COST_DEFAULT)
    review_cost = float(review_cost if review_cost is not None else REVIEW_COST_DEFAULT)
    health_records = compute_field_health_records(threshold_store, feedback_store)
    audit_time_step = min(sim_step, 24)

    batch_events = feedback_store.get("last_batch_events") or generate_tolling_events_3field(step=0, n_per_camera=16)
    if not feedback_store.get("last_batch_events"):
        # Provide initial routed/label state for first render.
        rng = np.random.default_rng(42)
        for event in batch_events:
            cam_key = lane_key(event["lane_id"])
            decision, gate_fields = route_event_with_3field_thresholds(event, threshold_store["current"][cam_key])
            label_event_error_type(event, decision, gate_fields, rng)

    # Keep calibrated baseline stagnant on a fixed reference batch.
    fixed_reference_events = generate_tolling_events_3field(step=0, n_per_camera=BATCH_SIZE_PER_CAMERA)
    fixed_shared_counts = simulate_batch_counts_for_weighted_policy(
        fixed_reference_events,
        thresholds_from_shared_weighted(shared_threshold),
        seed=0,
    )

    return (
        make_arbitrary_cutoff_figure(shared_threshold),
        make_results_comparison_figure(shared_threshold, threshold_store["current"], sim_step, dispute_cost, review_cost),
        f"Calibrated shared weighted baseline: {shared_threshold:.1f} (targets ~10% review)",
        f"Demo dispute cost assumption = ${dispute_cost:.2f} per disputed auto-clear",
        f"Demo manual review cost assumption = ${review_cost:.2f} per reviewed event",
        make_machine_action_table(batch_events, n_rows=10),
        make_error_count_kpis(
            feedback_store.get("last_batch_counts", {"auto_clear": 0, "review": 0, "type1": 0, "type2": 0}),
            fixed_shared_counts,
            sim_step,
            shared_threshold,
        ),
        make_results_kpi_board(shared_threshold, threshold_store["current"], sim_step, dispute_cost, review_cost),
        make_adaptive_over_time_figure(feedback_store.get("time_history", {})),
        make_field_health_cards(health_records),
        make_field_health_heatmap(health_records),
        make_audit_policy_panel(batch_events, audit_time_step),
        make_audit_rate_over_time_figure(10, audit_time_step),
        make_threshold_matrix_table(threshold_store),
        make_field_threshold_trend_figure(threshold_store, trend_camera or "1"),
    )


server = app.server

if __name__ == "__main__":
    export_dataset_csv()
    app.run(debug=False, use_reloader=False, dev_tools_hot_reload=False)
