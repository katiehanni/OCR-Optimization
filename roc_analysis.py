# ROC Curve Analysis for OCR System
# ---------------------------------
# This creates a ROC curve to evaluate OCR performance across all thresholds

from sklearn.metrics import roc_curve, auc, roc_auc_score
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_roc_curve(y_true, y_scores, title="ROC Curve for OCR System"):
    """
    Create a ROC curve for OCR system evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (0 or 1)
    y_scores : array-like  
        OCR confidence scores
    title : str
        Title for the plot
        
    Returns:
    --------
    tuple: (fig, auc_score, optimal_threshold)
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC curve plot
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=3)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Add optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    fig.add_trace(go.Scatter(
        x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
        mode='markers',
        name=f'Optimal Threshold (τ={optimal_threshold:.0f})',
        marker=dict(color='green', size=12, symbol='star')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Add annotations
    fig.add_annotation(
        x=0.6, y=0.2,
        text=f'AUC = {roc_auc:.3f}<br>Optimal τ = {optimal_threshold:.0f}',
        showarrow=True,
        arrowhead=2,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    return fig, roc_auc, optimal_threshold

# Example usage (run this in your notebook after creating val and best variables):
"""
# ROC Curve Analysis
fig_roc, auc_score, optimal_threshold = create_roc_curve(
    val["is_correct"], 
    val["score"], 
    "ROC Curve for OCR System Performance"
)

fig_roc.show()

# Print performance metrics
print("=== ROC Curve Analysis ===")
print(f"AUC Score: {auc_score:.3f}")
print(f"Optimal Threshold: {optimal_threshold:.0f}")

# Compare with utility-optimized threshold
utility_threshold = best["tau"]
print(f"\nUtility-optimized threshold: {utility_threshold}")
print(f"ROC-optimized threshold: {optimal_threshold:.0f}")
print(f"Difference: {abs(utility_threshold - optimal_threshold):.0f} points")
"""
