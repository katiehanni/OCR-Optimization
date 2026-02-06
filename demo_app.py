"""
OCR Decision Demo for Tolling Operations

A presentation-friendly Streamlit app for non-technical audiences.
Run with: streamlit run demo_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

st.set_page_config(
    page_title="OCR Decision Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def build_demo_data(seed: int = 42, n: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stable demo data with an OCR score, true correctness, and reviewer labels."""
    rng = np.random.default_rng(seed)
    true_tau = 80
    steepness = 5.0
    reviewer_accuracy = {"A": 0.95, "B": 0.85, "C": 0.70}

    def p_correct(score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(score - true_tau) / steepness))

    scores = rng.integers(0, 100, size=n)
    true_prob = p_correct(scores)
    is_correct = rng.binomial(1, true_prob)

    reviewers = rng.choice(list(reviewer_accuracy.keys()), size=n)
    review_label = np.array([
        true if rng.random() <= reviewer_accuracy[r] else (1 - true)
        for true, r in zip(is_correct, reviewers)
    ])

    timestamp = pd.to_datetime("2025-08-01") + pd.to_timedelta(rng.integers(0, 60, size=n), unit="D")

    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "score": scores,
            "is_correct": is_correct,
            "reviewer_id": reviewers,
            "review_label": review_label,
            "timestamp": timestamp,
        }
    ).sort_values("timestamp").reset_index(drop=True)

    split = int(0.8 * len(df))
    train = df.iloc[:split].copy()
    validation = df.iloc[split:].copy()
    return df, train, validation


def optimize_threshold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    review_penalty: float,
) -> tuple[pd.DataFrame, int, pd.Series, str]:
    """Calibrate score->correctness and choose threshold maximizing utility."""
    weighted = train.copy()
    weighted["agree"] = weighted["review_label"] == weighted["is_correct"]

    reviewer_stats = weighted.groupby("reviewer_id").agg(n=("id", "count"), agree=("agree", "sum"))
    reviewer_stats["weight"] = (1 + reviewer_stats["agree"]) / (2 + reviewer_stats["n"])
    weight_map = reviewer_stats["weight"].to_dict()

    weighted["sample_weight"] = weighted["reviewer_id"].map(weight_map).fillna(1.0)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(weighted["score"], weighted["is_correct"], sample_weight=weighted["sample_weight"])

    lr = LogisticRegression(max_iter=1000)
    lr.fit(
        weighted["score"].to_numpy().reshape(-1, 1) / 99.0,
        weighted["is_correct"],
        sample_weight=weighted["sample_weight"],
    )

    x_val = validation["score"].to_numpy().reshape(-1, 1)
    p_iso = np.clip(iso.predict(validation["score"]), 1e-6, 1 - 1e-6)
    p_platt = np.clip(lr.predict_proba(x_val / 99.0)[:, 1], 1e-6, 1 - 1e-6)

    brier_iso = brier_score_loss(validation["is_correct"], p_iso)
    brier_platt = brier_score_loss(validation["is_correct"], p_platt)
    chosen_calibrator = "Isotonic" if brier_iso <= brier_platt else "Platt"

    score = validation["score"].to_numpy()
    truth = validation["is_correct"].to_numpy()

    rows = []
    for tau in range(100):
        auto_accept = score >= tau
        review_rate = (score < tau).mean()
        accuracy = truth[auto_accept].mean() if auto_accept.any() else 0.0
        utility = accuracy - review_penalty * review_rate
        rows.append(
            {
                "tau": tau,
                "accuracy": accuracy,
                "review_rate": review_rate,
                "utility": utility,
                "auto_accept_rate": 1 - review_rate,
            }
        )

    curve = pd.DataFrame(rows)
    best_row = curve.iloc[curve["utility"].idxmax()]
    best_tau = int(best_row["tau"])
    return curve, best_tau, best_row, chosen_calibrator


def roi_estimate(
    volume_per_day: int,
    cost_per_review: float,
    baseline_review_rate: float,
    optimized_review_rate: float,
) -> tuple[float, float]:
    baseline_daily = volume_per_day * baseline_review_rate * cost_per_review
    optimized_daily = volume_per_day * optimized_review_rate * cost_per_review
    return baseline_daily, baseline_daily - optimized_daily


# Load and optimize
base_df, train_df, val_df = build_demo_data()

st.sidebar.header("Demo Controls")
lambda_penalty = st.sidebar.slider("Review cost priority (lambda)", 0.00, 0.60, 0.20, 0.01)
baseline_threshold = st.sidebar.slider("Current threshold (today)", 0, 99, 85, 1)
daily_volume = st.sidebar.number_input("Images per day", min_value=1000, max_value=500000, value=100000, step=5000)
review_unit_cost = st.sidebar.number_input("Cost per manual review ($)", min_value=0.10, max_value=10.00, value=0.75, step=0.05)

curve_df, optimal_threshold, optimal_metrics, chosen_calibrator = optimize_threshold(train_df, val_df, lambda_penalty)

score = val_df["score"].to_numpy()
truth = val_df["is_correct"].to_numpy()

baseline_mask = score >= baseline_threshold
baseline_review_rate = (score < baseline_threshold).mean()
baseline_accuracy = truth[baseline_mask].mean() if baseline_mask.any() else 0.0

optimized_review_rate = float(optimal_metrics["review_rate"])
optimized_accuracy = float(optimal_metrics["accuracy"])

baseline_daily_cost, daily_savings = roi_estimate(
    volume_per_day=int(daily_volume),
    cost_per_review=float(review_unit_cost),
    baseline_review_rate=float(baseline_review_rate),
    optimized_review_rate=optimized_review_rate,
)


st.title("OCR Confidence Threshold Demo")
st.caption("Use this story-driven demo to show how data chooses the best point between accuracy and manual workload.")

intro_c1, intro_c2 = st.columns([2, 1])
with intro_c1:
    st.markdown(
        """
        **Business question:** For each tolling image, should we auto-accept the OCR result or send it to manual review?

        This demo shows a transparent, data-driven way to set that decision threshold.
        """
    )
with intro_c2:
    st.info(
        f"Chosen calibrator: {chosen_calibrator}\n\n"
        f"Optimal threshold: {optimal_threshold}"
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("Recommended threshold", optimal_threshold)
k2.metric("Auto-accept accuracy", f"{optimized_accuracy:.1%}")
k3.metric("Manual review rate", f"{optimized_review_rate:.1%}")
k4.metric("Estimated daily savings", f"${daily_savings:,.0f}")

st.divider()

st.subheader("1) Current State vs Recommended State")

compare = pd.DataFrame(
    {
        "Scenario": ["Current", "Recommended"],
        "Threshold": [baseline_threshold, optimal_threshold],
        "Accuracy": [baseline_accuracy, optimized_accuracy],
        "Review Rate": [baseline_review_rate, optimized_review_rate],
    }
)

compare_long = compare.melt(
    id_vars=["Scenario", "Threshold"],
    value_vars=["Accuracy", "Review Rate"],
    var_name="Metric",
    value_name="Value",
)

bar_fig = px.bar(
    compare_long,
    x="Scenario",
    y="Value",
    color="Metric",
    barmode="group",
    text=compare_long["Value"].map(lambda v: f"{v:.1%}"),
    color_discrete_map={"Accuracy": "#2f6fdb", "Review Rate": "#f39c34"},
)
bar_fig.update_layout(height=380, yaxis_title="Percent", xaxis_title="")
bar_fig.update_traces(textposition="outside")
st.plotly_chart(bar_fig, use_container_width=True)

summary_text = (
    f"At the current threshold ({baseline_threshold}), about {baseline_accuracy:.1%} of auto-accepted reads are correct "
    f"and {baseline_review_rate:.1%} of images require manual review. "
    f"The recommended threshold ({optimal_threshold}) shifts this to {optimized_accuracy:.1%} accuracy with "
    f"{optimized_review_rate:.1%} manual review."
)
st.write(summary_text)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("2) Confidence Score Distribution")
    hist = px.histogram(base_df, x="score", nbins=20, color_discrete_sequence=["#2f6fdb"])
    hist.add_vline(x=baseline_threshold, line_dash="dot", line_color="#444", annotation_text="Current")
    hist.add_vline(x=optimal_threshold, line_dash="dash", line_color="#1a8f4b", annotation_text="Recommended")
    hist.update_layout(height=360, xaxis_title="OCR confidence score", yaxis_title="Image count")
    st.plotly_chart(hist, use_container_width=True)

with right:
    st.subheader("3) Utility by Threshold")
    utility_fig = go.Figure()
    utility_fig.add_trace(
        go.Scatter(
            x=curve_df["tau"],
            y=curve_df["utility"],
            mode="lines",
            line=dict(color="#2f6fdb", width=3),
            name="Utility",
        )
    )
    utility_fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="#1a8f4b",
        annotation_text=f"Best = {optimal_threshold}",
    )
    utility_fig.update_layout(height=360, xaxis_title="Threshold", yaxis_title="Business utility")
    st.plotly_chart(utility_fig, use_container_width=True)

st.divider()

st.subheader("4) Financial Impact Snapshot")
fin1, fin2, fin3 = st.columns(3)
fin1.metric("Current daily review cost", f"${baseline_daily_cost:,.0f}")
fin2.metric("Projected daily savings", f"${daily_savings:,.0f}")
fin3.metric("Projected annual savings", f"${daily_savings * 365:,.0f}")

st.caption(
    "Savings estimate uses: daily volume, cost per manual review, and reduction in review rate from current to recommended threshold."
)

with st.expander("Technical appendix"):
    st.markdown(
        "Calibration maps OCR scores to estimated correctness probability using isotonic or Platt scaling. "
        "Threshold is selected by maximizing utility: `accuracy - lambda * review_rate`, with reviewer quality weighting."
    )
    st.dataframe(base_df[["score", "is_correct", "reviewer_id", "timestamp"]].head(15), use_container_width=True)
