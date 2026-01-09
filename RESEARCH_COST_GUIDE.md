# Research-Oriented Cost Framework Guide

## Overview

This guide explains how to use the generalized cost framework for OCR threshold optimization in research settings where costs may be unknown, variable, or need to be explored.

## Key Design Principles

1. **Works with or without explicit costs** - Defaults based on class prevalence
2. **Multiple optimization objectives** - Not just cost minimization
3. **Sensitivity analysis** - Explore how costs affect optimal thresholds
4. **Multi-objective optimization** - Pareto frontiers when multiple goals exist
5. **Research-friendly** - Easy to experiment with different approaches

---

## Use Cases

### 1. **No Cost Information Available**

When you don't know the costs, use **cost-agnostic objectives**:

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective

# Option 1: F1 score (balanced precision/recall)
opt = ThresholdOptimizer(objective=OptimizationObjective.F_BETA, beta=1.0)
result = opt.optimize(scores, labels)

# Option 2: Youden's J (maximizes TPR - FPR)
opt = ThresholdOptimizer(objective=OptimizationObjective.YOUDENS_J)
result = opt.optimize(scores, labels)

# Option 3: Precision-Recall harmonic mean
opt = ThresholdOptimizer(objective=OptimizationObjective.PRECISION_RECALL)
result = opt.optimize(scores, labels)
```

**When to use:**
- Exploratory research
- Comparing different calibration methods
- When costs are truly unknown
- Baseline comparisons

---

### 2. **Partial Cost Information**

When you know some costs but not others, use **default cost estimation**:

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective, CostStructure

# Only know review cost, use defaults for FP/FN
cost_struct = CostStructure(
    cost_review=5.0,  # Known: $5 per review
    # cost_fp and cost_fn will use prevalence-based defaults
)

opt = ThresholdOptimizer(
    objective=OptimizationObjective.COST_SENSITIVE,
    cost_structure=cost_struct
)
result = opt.optimize(scores, labels)
```

**Default cost logic:**
- Costs inversely proportional to class prevalence
- Rare events (errors) get higher default costs
- Makes intuitive sense: rare but important events cost more to miss

---

### 3. **Full Cost Information Available**

When you have complete cost information:

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective, CostStructure

cost_struct = CostStructure(
    cost_fp=50.0,      # Accepting bad OCR costs $50
    cost_fn=5.0,       # Rejecting good OCR costs $5
    cost_review=10.0,  # Each review costs $10
    cost_tp=0.0,       # Accepting good OCR is free
    cost_tn=0.0        # Rejecting bad OCR is free
)

opt = ThresholdOptimizer(
    objective=OptimizationObjective.COST_SENSITIVE,
    cost_structure=cost_struct
)
result = opt.optimize(scores, labels)
```

---

### 4. **Cost Sensitivity Analysis**

Explore how optimal threshold changes with different cost assumptions:

```python
from cost_framework import cost_sensitivity_analysis
import numpy as np

# Test different cost ratios
cost_fp_range = np.array([1, 5, 10, 20, 50, 100])  # FP cost range
cost_fn_range = np.array([1, 5, 10])               # FN cost range

results = cost_sensitivity_analysis(
    scores, labels,
    cost_fp_range=cost_fp_range,
    cost_fn_range=cost_fn_range,
    cost_review=5.0
)

# Visualize how tau* changes with costs
import plotly.express as px
fig = px.scatter(
    results, x='cost_fp', y='tau_star', 
    color='cost_fn', size='objective_value',
    title='Optimal Threshold vs Cost Parameters'
)
fig.show()
```

**Research questions this answers:**
- How sensitive is the optimal threshold to cost assumptions?
- What cost ratios lead to different threshold choices?
- Is the system robust to cost uncertainty?

---

### 5. **Multi-Objective Optimization**

When you have multiple competing goals (e.g., accuracy AND review rate):

```python
from cost_framework import multi_objective_optimization, OptimizationObjective

# Optimize for both F1 score and review rate
results = multi_objective_optimization(
    scores, labels,
    objectives=[
        OptimizationObjective.F_BETA,
        OptimizationObjective.REVIEW_RATE
    ]
)

# Get Pareto-optimal solutions
pareto = results[results['pareto_optimal'] == True]

# Visualize Pareto frontier
import plotly.express as px
fig = px.scatter(
    pareto, x='f_beta', y='review_rate',
    hover_data=['tau'],
    title='Pareto Frontier: F1 vs Review Rate'
)
fig.show()
```

**Use cases:**
- Trade-off analysis
- Finding multiple "good" solutions
- Understanding objective conflicts
- Stakeholder decision-making

---

### 6. **F-Beta Score (Precision/Recall Trade-off)**

Control precision vs recall trade-off without explicit costs:

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective

# F1 score (balanced)
opt = ThresholdOptimizer(
    objective=OptimizationObjective.F_BETA,
    beta=1.0
)

# F2 score (favors recall - catch more errors)
opt = ThresholdOptimizer(
    objective=OptimizationObjective.F_BETA,
    beta=2.0
)

# F0.5 score (favors precision - fewer false alarms)
opt = ThresholdOptimizer(
    objective=OptimizationObjective.F_BETA,
    beta=0.5
)

result = opt.optimize(scores, labels)
```

**Beta interpretation:**
- β = 1.0: Equal weight to precision and recall
- β > 1.0: More weight on recall (catch errors)
- β < 1.0: More weight on precision (avoid false alarms)

---

### 7. **Constrained Optimization**

Minimize review rate subject to accuracy constraints:

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective

# Minimize review rate, but ensure precision >= 0.95
opt = ThresholdOptimizer(
    objective=OptimizationObjective.REVIEW_RATE,
    min_precision=0.95
)

result = opt.optimize(scores, labels)
```

---

## Research Workflow Examples

### Example 1: Exploratory Analysis (No Costs)

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective
import pandas as pd

# Try multiple objectives
objectives = [
    OptimizationObjective.F_BETA,
    OptimizationObjective.YOUDENS_J,
    OptimizationObjective.PRECISION_RECALL,
    OptimizationObjective.UTILITY
]

results = {}
for obj in objectives:
    opt = ThresholdOptimizer(objective=obj)
    results[obj.value] = opt.optimize(scores, labels)

# Compare results
comparison = pd.DataFrame({
    name: {
        'tau_star': r['tau_star'],
        'precision': r['metrics']['precision'],
        'recall': r['metrics']['recall'],
        'f1': r['metrics']['f_beta'],
        'review_rate': r['metrics']['review_rate']
    }
    for name, r in results.items()
}).T

print(comparison)
```

### Example 2: Cost Uncertainty Analysis

```python
from cost_framework import cost_sensitivity_analysis
import numpy as np

# Test wide range of cost assumptions
cost_fp_range = np.logspace(0, 2, 20)  # 1 to 100
cost_fn_range = np.logspace(0, 1.5, 15)  # 1 to ~32

results = cost_sensitivity_analysis(
    scores, labels,
    cost_fp_range=cost_fp_range,
    cost_fn_range=cost_fn_range
)

# Analyze robustness
tau_std = results.groupby('cost_fp')['tau_star'].std()
print(f"Threshold stability: std = {tau_std.mean():.2f}")

# Find cost ranges where threshold is stable
stable_regions = results.groupby('tau_star').agg({
    'cost_fp': ['min', 'max'],
    'cost_fn': ['min', 'max']
})
```

### Example 3: Multi-Objective Trade-off

```python
from cost_framework import multi_objective_optimization, OptimizationObjective

# Optimize for cost AND review rate
results = multi_objective_optimization(
    scores, labels,
    objectives=[
        OptimizationObjective.COST_SENSITIVE,
        OptimizationObjective.REVIEW_RATE
    ]
)

pareto = results[results['pareto_optimal']]

# Find knee point (best trade-off)
# (simplified - in practice use more sophisticated methods)
pareto['tradeoff_score'] = (
    pareto['cost_sensitive'] / pareto['cost_sensitive'].max() +
    (1 - pareto['review_rate'])  # Higher is better
)
best = pareto.loc[pareto['tradeoff_score'].idxmax()]
```

---

## Choosing the Right Approach

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| **No cost information** | F-Beta or Youden's J | Standard, interpretable metrics |
| **Partial cost info** | Cost-sensitive with defaults | Uses available info, estimates rest |
| **Full cost info** | Cost-sensitive with explicit costs | Most accurate for business |
| **Cost uncertainty** | Sensitivity analysis | Understand robustness |
| **Multiple goals** | Multi-objective optimization | Find trade-offs |
| **Precision critical** | F-Beta with β < 1.0 | Prioritize precision |
| **Recall critical** | F-Beta with β > 1.0 | Prioritize catching errors |
| **Minimize reviews** | Review rate optimization with constraints | Direct optimization |

---

## Integration with Existing Code

### Replace Current Utility Function

**Old approach:**
```python
utility = acc - lam * review_rate
```

**New approach (drop-in replacement):**
```python
from cost_framework import optimize_with_defaults

# Simple replacement
result = optimize_with_defaults(scores, labels, method="utility")
tau_star = result['tau_star']

# Or with explicit lambda
from cost_framework import ThresholdOptimizer, OptimizationObjective
opt = ThresholdOptimizer(
    objective=OptimizationObjective.UTILITY,
    lambda_param=0.2  # Your existing lambda
)
result = opt.optimize(scores, labels)
```

### Enhanced Analysis

```python
# Instead of just finding optimal threshold, get full analysis
from cost_framework import ThresholdOptimizer, OptimizationObjective

opt = ThresholdOptimizer(objective=OptimizationObjective.COST_SENSITIVE)
result = opt.optimize(scores, labels)

# Access full metrics
print(f"Optimal threshold: {result['tau_star']}")
print(f"Precision: {result['metrics']['precision']:.3f}")
print(f"Recall: {result['metrics']['recall']:.3f}")
print(f"F1: {result['metrics']['f_beta']:.3f}")
print(f"Review rate: {result['metrics']['review_rate']:.3f}")

# Get full curve for visualization
curve = result['curve']
```

---

## Best Practices for Research

1. **Start with cost-agnostic methods** - Establish baselines
2. **Use sensitivity analysis** - Understand cost impact
3. **Compare multiple objectives** - See which aligns with goals
4. **Report multiple metrics** - Don't just report optimal threshold
5. **Use Pareto frontiers** - When multiple objectives matter
6. **Document cost assumptions** - Even if using defaults

---

## Example: Complete Research Pipeline

```python
import numpy as np
import pandas as pd
from cost_framework import (
    ThresholdOptimizer, OptimizationObjective, CostStructure,
    cost_sensitivity_analysis, multi_objective_optimization
)

# 1. Baseline: Cost-agnostic methods
baseline_results = {}
for obj_name in ['F_BETA', 'YOUDENS_J', 'PRECISION_RECALL']:
    opt = ThresholdOptimizer(objective=OptimizationObjective[obj_name])
    baseline_results[obj_name] = opt.optimize(scores, labels)

# 2. Cost-sensitive with defaults
opt = ThresholdOptimizer(objective=OptimizationObjective.COST_SENSITIVE)
default_cost_result = opt.optimize(scores, labels)

# 3. Sensitivity analysis
sensitivity = cost_sensitivity_analysis(
    scores, labels,
    cost_fp_range=np.array([1, 5, 10, 20, 50]),
    cost_fn_range=np.array([1, 5, 10])
)

# 4. Multi-objective
pareto = multi_objective_optimization(
    scores, labels,
    objectives=[
        OptimizationObjective.COST_SENSITIVE,
        OptimizationObjective.REVIEW_RATE
    ]
)

# 5. Report findings
print("=== Research Summary ===")
print(f"Baseline F1 optimal tau: {baseline_results['F_BETA']['tau_star']}")
print(f"Cost-sensitive optimal tau: {default_cost_result['tau_star']}")
print(f"Threshold stability (std): {sensitivity['tau_star'].std():.2f}")
print(f"Pareto-optimal solutions: {pareto['pareto_optimal'].sum()}")
```

---

## Summary

The cost framework provides:

✅ **Flexibility** - Works with or without explicit costs  
✅ **Multiple objectives** - Not limited to cost minimization  
✅ **Research-friendly** - Easy experimentation  
✅ **Sensitivity analysis** - Understand robustness  
✅ **Multi-objective** - Handle competing goals  
✅ **Backward compatible** - Can replace existing utility function  

This makes it ideal for research where you want to explore different optimization strategies without requiring specific business cost information.

