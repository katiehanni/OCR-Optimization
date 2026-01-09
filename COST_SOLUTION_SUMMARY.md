# Generalized Cost Framework for Research

## Problem

The current implementation has limited cost accounting - it only considers review cost via λ, but doesn't distinguish between different error types (false positives vs false negatives).

For **research purposes**, you need a flexible framework that:
- Works with or without explicit cost information
- Supports multiple optimization objectives
- Allows sensitivity analysis
- Is easy to experiment with

## Solution

I've created a **generalized cost framework** (`cost_framework.py`) that provides:

### 1. **Multiple Optimization Objectives**

Not just cost minimization - choose what matters for your research:

- **Cost-Sensitive**: Minimize expected cost (with or without explicit costs)
- **F-Beta Score**: Control precision/recall trade-off (β parameter)
- **Youden's J**: Maximize TPR - FPR
- **Precision-Recall**: Harmonic mean
- **Utility**: Your existing function (accuracy - λ × review_rate)
- **Review Rate**: Minimize reviews with constraints
- **Accuracy**: Simple accuracy maximization

### 2. **Works Without Explicit Costs**

When costs are unknown, the framework:
- Uses **prevalence-based defaults** (rare events get higher default costs)
- Provides **cost-agnostic objectives** (F1, Youden's J, etc.)
- Allows **sensitivity analysis** to explore cost impact

### 3. **Flexible Cost Structures**

```python
# No costs - use defaults
opt = ThresholdOptimizer(objective=OptimizationObjective.COST_SENSITIVE)

# Partial costs - defaults for missing ones
cost_struct = CostStructure(cost_review=5.0)  # FP/FN use defaults

# Full costs - explicit everything
cost_struct = CostStructure(
    cost_fp=50.0, cost_fn=5.0, cost_review=10.0
)
```

### 4. **Research Tools**

- **Sensitivity Analysis**: How does threshold change with costs?
- **Multi-Objective Optimization**: Pareto frontiers for competing goals
- **Multiple Metrics**: Get precision, recall, F1, review rate, etc.

## Quick Start

### Basic Usage (No Costs)

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective

# Use F1 score (no cost info needed)
opt = ThresholdOptimizer(objective=OptimizationObjective.F_BETA, beta=1.0)
result = opt.optimize(scores, labels)
print(f"Optimal threshold: {result['tau_star']}")
```

### With Costs

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective, CostStructure

# Explicit costs
cost_struct = CostStructure(
    cost_fp=50.0,      # Accepting bad OCR
    cost_fn=5.0,        # Rejecting good OCR
    cost_review=10.0    # Review cost
)

opt = ThresholdOptimizer(
    objective=OptimizationObjective.COST_SENSITIVE,
    cost_structure=cost_struct
)
result = opt.optimize(scores, labels)
```

### Sensitivity Analysis

```python
from cost_framework import cost_sensitivity_analysis
import numpy as np

# Test different cost assumptions
results = cost_sensitivity_analysis(
    scores, labels,
    cost_fp_range=np.array([1, 5, 10, 20, 50]),
    cost_fn_range=np.array([1, 5, 10])
)

# See how threshold changes
print(results[['cost_fp', 'cost_fn', 'tau_star']])
```

## Files Created

1. **`cost_framework.py`** - Main framework implementation
2. **`RESEARCH_COST_GUIDE.md`** - Comprehensive usage guide
3. **`example_cost_usage.py`** - Working examples
4. **`COST_SOLUTION_SUMMARY.md`** - This file

## Key Advantages for Research

✅ **No cost info required** - Use F1, Youden's J, etc.  
✅ **Partial cost info OK** - Defaults fill in the gaps  
✅ **Full cost info supported** - Explicit cost structures  
✅ **Sensitivity analysis** - Understand robustness  
✅ **Multiple objectives** - Compare different approaches  
✅ **Easy experimentation** - Simple API  
✅ **Backward compatible** - Can replace existing utility function  

## Integration

### Replace Current Approach

**Old:**
```python
utility = acc - lam * review_rate
```

**New (drop-in):**
```python
from cost_framework import optimize_with_defaults
result = optimize_with_defaults(scores, labels, method="utility")
```

### Enhanced Analysis

```python
from cost_framework import ThresholdOptimizer, OptimizationObjective

opt = ThresholdOptimizer(objective=OptimizationObjective.COST_SENSITIVE)
result = opt.optimize(scores, labels)

# Get full metrics
print(result['metrics'])  # precision, recall, F1, review_rate, etc.
print(result['confusion_matrix'])  # TP, FP, TN, FN
print(result['curve'])  # Full curve for all thresholds
```

## Research Workflow

1. **Start cost-agnostic** - Use F1 or Youden's J for baselines
2. **Add cost sensitivity** - Use default costs if unknown
3. **Sensitivity analysis** - Explore cost impact
4. **Compare objectives** - See which aligns with goals
5. **Multi-objective** - If multiple goals matter

## Next Steps

1. Read `RESEARCH_COST_GUIDE.md` for detailed examples
2. Run `example_cost_usage.py` to see it in action
3. Integrate into your notebooks using the examples
4. Experiment with different objectives for your research

The framework is designed to be **research-friendly** - you can explore different optimization strategies without requiring specific business cost information, while still supporting cost-sensitive optimization when that information is available.


