"""
OCR Optimization Dashboard (Dash)1.

Run with:
    python3 dash_base_app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"

THRESHOLD = 0.70

LANES = [
    {"lane_id": 1, "camera": "CAM-1", "pass_rate": 0.92},
    {"lane_id": 2, "camera": "CAM-2", "pass_rate": 0.64},
    {"lane_id": 3, "camera": "CAM-3", "pass_rate": 0.84},
]


def status_for_rate(rate: float) -> str:
    if rate < 0.70:
        return "Poor"
    if rate < 0.85:
        return "Watch"
    return "Good"


def badge_class_for_status(status: str) -> str:
    if status == "Poor":
        return "text-bg-danger"
    if status == "Watch":
        return "text-bg-warning"
    return "text-bg-success"


def build_lane_points(pass_rate: float, seed: int, n: int = 22) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = np.arange(1, n + 1)
    above = rng.random(n) < pass_rate
    confidence = np.where(
        above,
        rng.uniform(THRESHOLD, 0.98, n),
        rng.uniform(0.25, THRESHOLD - 0.02, n),
    )
    return [
        {"x": int(xi), "confidence": float(ci), "correct": bool(ok)}
        for xi, ci, ok in zip(x, confidence, above)
    ]


LANE_POINTS = {lane["lane_id"]: build_lane_points(lane["pass_rate"], seed=100 + lane["lane_id"]) for lane in LANES}


for lane in LANES:
    lane["status"] = status_for_rate(lane["pass_rate"])


def make_mini_scatter(lane_id: int) -> go.Figure:
    points = LANE_POINTS[lane_id]
    x = [p["x"] for p in points]
    y = [p["confidence"] for p in points]
    c = ["Above threshold" if p["correct"] else "Below threshold" for p in points]
    color = ["#16a34a" if p["correct"] else "#dc2626" for p in points]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=8, color=color, line=dict(width=0.8, color="#0f172a")),
                customdata=c,
                hovertemplate=(
                    "Vehicle #: %{x}<br>"
                    "Confidence: %{y:.0%}<br>"
                    "Threshold " + f"{THRESHOLD:.0%}: %{{customdata}}<extra></extra>"
                ),
                showlegend=False,
            )
        ]
    )
    fig.add_hline(y=THRESHOLD, line_width=1.5, line_dash="dash", line_color="#64748b")
    fig.update_layout(
        margin=dict(l=8, r=8, t=4, b=8),
        height=180,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False, range=[0.2, 1.0])
    return fig


def make_lane_card(lane: dict) -> html.Div:
    lane_id = lane["lane_id"]
    return html.Div(
        className="col-12 col-md-6 col-lg-4",
        children=[
            html.Div(
                className="card h-100 shadow-sm border-0",
                children=[
                    html.Div(
                        className="card-header bg-white border-0 pb-0",
                        children=[
                            html.Div(
                                className="d-flex justify-content-between align-items-start",
                                children=[
                                    html.H5(lane["camera"], className="mb-1"),
                                    html.Span(lane["status"], className=f"badge {badge_class_for_status(lane['status'])}"),
                                ],
                            ),
                            html.Div(
                                f"Pass rate at {THRESHOLD:.0%} threshold: {lane['pass_rate']:.0%}",
                                className="text-secondary small",
                            ),
                        ],
                    ),
                    html.Div(
                        className="card-body pt-2",
                        children=[
                            dcc.Graph(
                                id={"type": "lane-mini-graph", "index": lane_id},
                                figure=make_mini_scatter(lane_id),
                                config={"displayModeBar": False},
                            )
                        ],
                    ),
                ],
            )
        ],
    )


def build_summary_metrics() -> html.Div:
    avg_pass = float(np.mean([lane["pass_rate"] for lane in LANES]))
    poorest = min(LANES, key=lambda l: l["pass_rate"])
    return html.Div(
        className="row g-3 mb-2",
        children=[
            html.Div(
                className="col-12 col-md-6",
                children=[
                    html.Div(
                        className="card border-0 shadow-sm",
                        children=[
                            html.Div(
                                className="card-body",
                                children=[
                                    html.Div("Threshold (all gantries)", className="text-secondary small"),
                                    html.H4(f"{THRESHOLD:.0%}", className="mb-0"),                                ],
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                className="col-12 col-md-6",
                children=[
                    html.Div(
                        className="card border-0 shadow-sm",
                        children=[
                            html.Div(
                                className="card-body",
                                children=[
                                    html.Div("Failing gantry (lowest pass rate)", className="text-secondary small"),
                                    html.H4(f"{poorest['camera']} ({poorest['pass_rate']:.0%} pass)", className="mb-0"),                                ],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )


app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.title = "OCR Optimization Demo"

app.layout = html.Div(
    className="container py-4",
    children=[
        html.Div(
            className="mb-3",
            children=[
                html.H2("OCR Optimization Demo", className="mb-1"),
                html.P(
                    "All gantries use the same threshold value. However, accuracy varies by camera, showing that the given threshold is arbitrary.",
                    className="text-secondary mb-0",
                ),
            ],
        ),
        build_summary_metrics(),
        html.Div(
            className="d-flex justify-content-end mb-3",
            children=[
                dcc.Dropdown(
                    id="lane-filter",
                    options=[
                        {"label": "All lanes", "value": "all"},
                        *[{"label": lane["camera"], "value": lane["lane_id"]} for lane in LANES],
                    ],
                    value="all",
                    clearable=False,
                    style={"width": "220px"},
                )
            ],
        ),
        html.Div(id="lane-card-grid", className="row g-3 mb-4"),
    ],
)


@app.callback(
    Output("lane-card-grid", "children"),
    Input("lane-filter", "value"),
)
def render_dashboard(lane_filter: str | int) -> list[html.Div]:
    if lane_filter == "all":
        visible_lanes = LANES
    else:
        visible_lanes = [lane for lane in LANES if lane["lane_id"] == int(lane_filter)]
    return [make_lane_card(lane) for lane in visible_lanes]


if __name__ == "__main__":
    app.run(debug=True)
