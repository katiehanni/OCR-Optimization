# OCR Threshold Optimization — v1.0

## Overview  
This project introduces a data-driven calibration and threshold optimization pipeline for Optical Character Recognition (OCR) confidence scores.  
The goal is to automatically determine and adapt an optimal confidence threshold \( \tau \) that balances **accuracy** and **human review cost**, replacing prior heuristic or Bayesian thresholding methods.

---

## 1. Motivation  

Traditional OCR systems rely on raw model confidence scores to decide whether to **auto-accept** or **send to manual review**.  
However, these scores are *ordinal* and not directly probabilistic — a score of 85 does not necessarily mean an 85% chance of correctness.  

The original idea was that a Bayesian prior could be used to adjust this, but it was static and required manual tuning.  
Our new data-driven approach directly learns calibration and threshold behavior from observed performance data.

---

## 2. Approach Summary  

### Step 1 — Generate or Ingest OCR Data

Each document *i* has:

- OCR confidence score **sᵢ ∈ [0, 99]**  
- Ground-truth correctness label **yᵢ ∈ {0, 1}**  
- Optional reviewer label **rᵢ** and timestamp **tᵢ**  

Synthetic data is used for initial testing, with a configurable **“true” sigmoid probability curve** controlling the likelihood of correctness.

---

### Step 2 — Calibration  
We transform ordinal OCR scores into **calibrated probabilities** \( \hat{p}_i = P(y_i = 1 \mid s_i) \) using two approaches:

| Calibrator | Description | Type |
|-------------|--------------|------|
| **Isotonic Regression** | Non-parametric, piecewise-constant monotonic mapping | Non-parametric |
| **Platt Scaling** | Logistic regression: \( f_{\text{platt}}(s) = \frac{1}{1 + e^{-(\alpha + \beta s)}} \) | Parametric |

The chosen calibrator minimizes **Brier score** and **log-loss** on held-out validation data.

---

### Step 3 — Threshold Optimization

For each candidate threshold **τ**:

$$
\text{Utility}(\tau) = \text{Accuracy}(\tau) - \lambda \times \text{ReviewRate}(\tau)
$$

where:

- **τ**: decision threshold  
- **λ ∈ [0, 1]**: review-cost penalty  
- **Accuracy(τ)**: proportion of auto-accepted documents that are correct  
- **ReviewRate(τ)**: fraction of documents routed to review  

The optimal threshold **τ\*** maximizes this utility function.  
We typically find **τ\* ≈ 78–80** balances performance for moderate review cost (**λ = 0.2**).

---

### Step 4 — Online Updating

A rolling **30-day calibration window** allows the model to adapt dynamically to production drift:

- ↓ **τ** → OCR accuracy is improving  
- ↑ **τ** → OCR confidence reliability is decaying  
- Summary metrics (**Brier**, **LogLoss**) track calibration quality  
- Reviewer reliability weights are updated using **Beta–Bernoulli posteriors**

---

## 3. Key Results  

| Metric | Value | Notes |
|---------|--------|-------|
| **Optimal threshold (τ\*) | ≈ 78 | Balances 84% accuracy, 65% review rate |
| Accuracy at τ | 0.84 | On validation set |
| Review rate | 0.65 | Documents sent for manual review |
| Utility | 0.71 | Combined metric |
| Calibrator | Isotonic | Chosen by lowest log-loss |
| Validation utility loss | 0% | Perfect generalization (synthetic) |

**Interpretation:**  
A threshold near 78 auto-accepts ~35% of documents while maintaining ~84% accuracy, achieving an optimal utility trade-off.

---

## 4. Example Visuals  

| Visualization | Description |
|----------------|--------------|
| **Histogram of OCR Scores** | Distribution of confidence outputs |
| **Empirical Accuracy by Score Bin** | Shows non-linearity before calibration |
| **Utility vs. Threshold τ** | Identifies utility-optimal τ |
| **τ Drift Over Time** | Monitors model calibration drift |
| **λ Sensitivity Curve** | Effect of increasing review-cost penalty |

---

## 5. Next Steps  

- Freeze this configuration as **Baseline v1.0**  
- Implement **online recalibration** using production OCR logs  
- Extend **reviewer reliability tracking** (via Bayesian posteriors)  
- Add **adaptive λ tuning** for variable cost environments  
- Integrate into **QuickSight or Streamlit dashboard** for real-time monitoring  

---

## 7. Dependencies  

- Python ≥ 3.10  
- pandas, numpy  
- scikit-learn (for calibration models)  
- plotly (for visualization)  
- joblib (for model serialization)

---

## 8. Mathematical Appendix

### Calibration

Raw OCR scores are ordinal. A **monotonic calibration function** maps them to probabilities:

$$
\hat{p}_i = f(s_i) = P(y_i = 1 \mid s_i)
$$

Two calibration methods are supported:

- **Isotonic Regression** — piecewise-constant, non-parametric mapping  
- **Platt Scaling** — parametric logistic form  

$$
f_{\text{platt}}(s) = \frac{1}{1 + e^{-(\alpha + \beta s)}}
$$

The model that minimizes the **Brier score** and **log-loss** on validation data is selected.

---

### Reviewer Weighting

Reviewer reliability is estimated via a **Beta–Bernoulli model**:

$$
\alpha_r = 1 + \text{count(correct reviews by reviewer } r)
$$

$$
\beta_r = 1 + \text{count(incorrect reviews by reviewer } r)
$$

Reviewer weight:

$$
w_r = \frac{\alpha_r}{\alpha_r + \beta_r}
$$

---

### Threshold Utility

Decision rule based on maximizing expected utility:

$$
U(\tau) = \text{Accuracy}(\tau) - \lambda \times \text{ReviewRate}(\tau)
$$

Optimal threshold:

$$
\tau^* = \arg\max_{\tau} U(\tau)
$$

---

## 9. Authors & Notes  
**Author:** Katie Hannigan  
**Contributors:** David Casale 
**Last Updated:** November 2025  
