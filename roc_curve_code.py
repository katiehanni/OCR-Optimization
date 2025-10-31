# ROC Curve Analysis for OCR System
# Copy and paste this code into a new cell in your notebook

from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import numpy as np

# Calculate ROC curve for the validation set
y_true = val["is_correct"]
y_scores = val["score"]  # Using raw OCR scores as confidence

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Create ROC curve plot
fig_roc = go.Figure()

# Add ROC curve
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC Curve (AUC = {roc_auc:.3f})',
    line=dict(color='blue', width=3)
))

# Add diagonal line (random classifier)
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(color='red', dash='dash', width=2)
))

# Add optimal threshold point
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
fig_roc.add_trace(go.Scatter(
    x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
    mode='markers',
    name=f'Optimal Threshold (τ={optimal_threshold:.0f})',
    marker=dict(color='green', size=12, symbol='star')
))

# Update layout
fig_roc.update_layout(
    title='ROC Curve for OCR System Performance',
    xaxis_title='False Positive Rate (FPR)',
    yaxis_title='True Positive Rate (TPR)',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
    template='plotly_white',
    hovermode='x unified'
)

# Add annotations
fig_roc.add_annotation(
    x=0.6, y=0.2,
    text=f'AUC = {roc_auc:.3f}<br>Optimal τ = {optimal_threshold:.0f}',
    showarrow=True,
    arrowhead=2,
    bgcolor='rgba(255,255,255,0.8)',
    bordercolor='black',
    borderwidth=1
)

fig_roc.show()

# Print performance metrics
print("=== ROC Curve Analysis ===")
print(f"AUC Score: {roc_auc:.3f}")
print(f"Optimal Threshold: {optimal_threshold:.0f}")
print(f"At optimal threshold:")
print(f"  - True Positive Rate: {tpr[optimal_idx]:.3f}")
print(f"  - False Positive Rate: {fpr[optimal_idx]:.3f}")
print(f"  - You correctly identify {tpr[optimal_idx]:.1%} of correct OCR results")
print(f"  - You incorrectly accept {fpr[optimal_idx]:.1%} of incorrect OCR results")

# Compare with your utility-optimized threshold
utility_threshold = best["tau"]
print(f"\nUtility-optimized threshold: {utility_threshold}")
print(f"ROC-optimized threshold: {optimal_threshold:.0f}")
print(f"Difference: {abs(utility_threshold - optimal_threshold):.0f} points")



