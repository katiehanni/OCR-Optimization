# First Step: Enhance Your Threshold Optimization

## Current Situation

You have a `choose_threshold` function that uses:
```python
utility = accuracy - λ × review_rate
```

This only accounts for review cost, not different error types.

## First Step: Add Cost-Aware Version

Add this enhanced function to your notebook. It's **backward compatible** - works with or without cost information.

### Step 1: Add This Cell to Your Notebook

```python
def choose_threshold_enhanced(
    scores, 
    labels, 
    lam=0.2,
    cost_fp=None,      # Cost of false positive (accepting bad OCR)
    cost_fn=None,      # Cost of false negative (rejecting good OCR)  
    cost_review=None,  # Cost per review
    objective='utility'  # 'utility', 'cost', 'f1', 'youdens_j'
):
    """
    Enhanced threshold optimization with cost-aware options.
    
    Parameters:
    -----------
    scores : array-like
        OCR confidence scores
    labels : array-like
        Ground truth (1 = correct, 0 = incorrect)
    lam : float
        Review cost weight (for 'utility' objective)
    cost_fp : float, optional
        False positive cost (accepting bad OCR)
    cost_fn : float, optional
        False negative cost (rejecting good OCR)
    cost_review : float, optional
        Cost per review
    objective : str
        'utility' (original), 'cost', 'f1', 'youdens_j'
    
    Returns:
    --------
    results_df : DataFrame with all thresholds
    best : dict with optimal threshold and metrics
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    thresholds = np.arange(0, 100)
    
    # Default costs based on class prevalence (if not provided)
    pos_rate = labels.mean()
    neg_rate = 1 - pos_rate
    
    if cost_fp is None:
        cost_fp = 1.0 / neg_rate if neg_rate > 0 else 1.0
    if cost_fn is None:
        cost_fn = 1.0 / pos_rate if pos_rate > 0 else 1.0
    if cost_review is None:
        cost_review = (cost_fp + cost_fn) / 2
    
    results = []
    best = None
    best_value = -np.inf
    
    for t in thresholds:
        preds = (scores >= t).astype(int)
        
        # Confusion matrix
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        total = len(scores)
        
        # Metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        review_rate = (scores < t).mean()
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Youden's J
        youdens_j = recall - fpr
        
        # Objective value
        if objective == 'utility':
            obj_value = accuracy - lam * review_rate
        elif objective == 'cost':
            # Expected cost (negative because we maximize)
            total_cost = cost_fp * fp + cost_fn * fn + cost_review * (fn + tn)
            obj_value = -total_cost
        elif objective == 'f1':
            obj_value = f1
        elif objective == 'youdens_j':
            obj_value = youdens_j
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        result_row = {
            'tau': t,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'youdens_j': youdens_j,
            'review_rate': review_rate,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'objective_value': obj_value
        }
        
        if objective == 'utility':
            result_row['utility'] = obj_value
        
        results.append(result_row)
        
        if obj_value > best_value:
            best_value = obj_value
            best = result_row.copy()
            best['tau_star'] = t
    
    return pd.DataFrame(results), best
```

### Step 2: Test It

```python
# Test 1: Original behavior (backward compatible)
results_df, best = choose_threshold_enhanced(
    val["score"].to_numpy(), 
    val["is_correct"].to_numpy(), 
    lam=0.2,
    objective='utility'
)
print(f"Original method: tau* = {best['tau_star']}")

# Test 2: Cost-aware (with explicit costs)
results_df, best = choose_threshold_enhanced(
    val["score"].to_numpy(), 
    val["is_correct"].to_numpy(),
    cost_fp=50.0,      # Accepting bad OCR is expensive
    cost_fn=5.0,       # Rejecting good OCR is less expensive
    cost_review=10.0,  # Review cost
    objective='cost'
)
print(f"Cost-aware: tau* = {best['tau_star']}")
print(f"Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}")

# Test 3: No cost info needed (F1 score)
results_df, best = choose_threshold_enhanced(
    val["score"].to_numpy(), 
    val["is_correct"].to_numpy(),
    objective='f1'
)
print(f"F1-optimal: tau* = {best['tau_star']}, F1 = {best['f1']:.3f}")
```

### Step 3: Compare Methods

```python
# Compare different objectives
objectives = ['utility', 'cost', 'f1', 'youdens_j']
comparison = []

for obj in objectives:
    _, best = choose_threshold_enhanced(
        val["score"].to_numpy(), 
        val["is_correct"].to_numpy(),
        lam=0.2,
        cost_fp=50.0, cost_fn=5.0, cost_review=10.0,
        objective=obj
    )
    comparison.append({
        'Method': obj,
        'tau_star': best['tau_star'],
        'Precision': best['precision'],
        'Recall': best['recall'],
        'F1': best['f1'],
        'Review_Rate': best['review_rate']
    })

df_comparison = pd.DataFrame(comparison)
print(df_comparison)
```

## What This Gives You

✅ **Backward compatible** - Your existing code still works  
✅ **Cost-aware** - Distinguishes FP vs FN costs  
✅ **No cost info needed** - Use F1 or Youden's J  
✅ **Multiple objectives** - Compare different approaches  
✅ **Full metrics** - Precision, recall, F1, confusion matrix  

## Next Steps After This

1. **Sensitivity analysis** - See how costs affect optimal threshold
2. **Multi-objective** - Find Pareto-optimal solutions
3. **Visualization** - Compare different objectives

But start with this - it's a simple drop-in replacement that gives you immediate cost-aware capabilities!

