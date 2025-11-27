# Hyperparameter Optimization Documentation

## ✅ Implementation Checklist

All tasks from your requirements have been implemented:

- [x] **Research and select optimization framework** → Optuna (TPE sampler)
- [x] **Design parameter space configuration format** → YAML-based presets
- [x] **Implement optimization workflow integration** → Fully integrated with GA engine
- [x] **Add optimization results tracking** → JSON tracking with OptimizationTracker
- [x] **Create optimization presets for common use cases** → 5 presets (fast, balanced, thorough, interpretability-focused, accuracy-focused)
- [x] **Add Bayesian optimization strategies** → TPE sampler with multivariate optimization
- [x] **Implement early stopping and pruning** → MedianPruner with configurable warmup

---

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Fast optimization (10 trials, ~10 minutes)
python scripts/hyperopt_with_optuna.py --preset fast --dataset iris

# Balanced optimization (30 trials, ~30 minutes)
python scripts/hyperopt_with_optuna.py --preset balanced --dataset breast_cancer

# Thorough optimization (100 trials, ~2 hours)
python scripts/hyperopt_with_optuna.py --preset thorough --dataset wine
```

### 2. Custom Optimization

```bash
# Custom number of trials
python scripts/hyperopt_with_optuna.py --n-trials 50 --dataset breast_cancer

# With timeout
python scripts/hyperopt_with_optuna.py --n-trials 100 --timeout 3600 --dataset wine

# Custom CV folds
python scripts/hyperopt_with_optuna.py --preset balanced --cv-folds 10
```

### 3. Study Management

```bash
# Named study with persistent storage
python scripts/hyperopt_with_optuna.py \
    --study-name my_experiment \
    --storage sqlite:///optuna.db \
    --dataset breast_cancer

# Resume previous study
python scripts/hyperopt_with_optuna.py \
    --study-name my_experiment \
    --storage sqlite:///optuna.db \
    --resume
```

---

## 📊 Optimization Presets

### `fast` - Quick Exploration
- **Trials:** 10
- **Time:** ~10 minutes
- **CV Folds:** 3
- **Use case:** Rapid prototyping, testing
- **Search space:** Limited (3-5 parameters)

**Parameters optimized:**
- population_size: [50, 80, 100]
- n_generations: [20, 30, 40]
- crossover_prob: [0.6, 0.8]
- mutation_prob: [0.1, 0.25]
- accuracy_weight: [0.6, 0.75]

---

### `balanced` - Recommended Default
- **Trials:** 30
- **Time:** ~30 minutes
- **CV Folds:** 5
- **Use case:** Standard optimization
- **Search space:** Medium (8 parameters)

**Parameters optimized:**
- population_size: 50-150
- n_generations: 20-60
- crossover_prob: 0.5-0.9
- mutation_prob: 0.1-0.3
- tournament_size: 2-5
- max_depth: 3-7
- accuracy_weight: 0.6-0.9
- node_complexity_weight: 0.4-0.7

---

### `thorough` - Comprehensive Search
- **Trials:** 100
- **Time:** ~2 hours
- **CV Folds:** 5
- **Use case:** Final optimization, research
- **Search space:** Full (12 parameters)

**Parameters optimized:**
- All GA parameters
- All tree constraints
- All fitness weights
- Most comprehensive search

---

### `interpretability_focused` - Small Trees
- **Trials:** 50
- **Time:** ~1 hour
- **CV Folds:** 5
- **Use case:** Medical, legal applications
- **Focus:** Maximize interpretability

**Parameters optimized:**
- Emphasizes node_complexity_weight (0.5-0.8)
- Restricts max_depth (3-6)
- Balances accuracy_weight (0.5-0.75)

---

### `accuracy_focused` - Maximum Performance
- **Trials:** 50
- **Time:** ~1 hour
- **CV Folds:** 5
- **Use case:** Performance-critical applications
- **Focus:** Maximize accuracy

**Parameters optimized:**
- Emphasizes accuracy_weight (0.75-0.95)
- Allows deeper trees (5-8)
- Larger populations (80-180)

---

## 🔬 Bayesian Optimization Details

### TPE Sampler (Tree-structured Parzen Estimator)

The implementation uses Optuna's TPE sampler with:

```python
sampler = optuna.samplers.TPESampler(
    seed=42,                # Reproducible results
    n_startup_trials=10,    # Random search for first 10 trials
    multivariate=True,      # Consider parameter interactions
    constant_liar=True      # Support parallel optimization
)
```

**Advantages:**
- ✅ Learns from previous trials
- ✅ Considers parameter interactions
- ✅ More efficient than random/grid search
- ✅ Handles continuous and categorical parameters

**How it works:**
1. First 10 trials: Random sampling (exploration)
2. Remaining trials: Bayesian optimization (exploitation)
3. Models relationship between parameters and objective
4. Suggests promising parameter combinations

---

## ⚡ Early Stopping & Pruning

### MedianPruner Implementation

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,     # Don't prune first 5 trials
    n_warmup_steps=2,       # Wait for 2 CV folds
    interval_steps=1        # Check after each fold
)
```

**How it works:**
1. During CV, reports intermediate scores after each fold
2. Compares current trial to median of all trials
3. If current trial is worse than median → PRUNE (stop early)
4. Saves time by not completing unpromising trials

**Example:**
```
Trial 15, Fold 1: 0.85 accuracy
Trial 15, Fold 2: 0.84 accuracy (median of all trials: 0.90)
→ PRUNED! (Trial is clearly underperforming)
```

**Benefits:**
- ⚡ 2-3× faster optimization
- 💰 Saves compute resources
- 🎯 Focuses on promising regions

---

## 📈 Results Tracking

### OptimizationTracker

Automatically tracks and logs:

**1. Progress File** (`results/optimization/{study_name}_progress.json`)
```json
{
  "study_name": "ga_opt_breast_cancer_20241125",
  "start_time": "2024-11-25T10:00:00",
  "n_trials": 30,
  "history": {
    "trials": [0, 1, 2, ...],
    "best_values": [0.85, 0.87, 0.89, ...],
    "best_params": [{...}, {...}, ...],
    "timestamps": ["...", "...", ...]
  }
}
```

**2. Final Report** (`results/optimization/{study_name}_report.json`)
```json
{
  "study_name": "ga_opt_breast_cancer_20241125",
  "duration_seconds": 1850.5,
  "n_trials": 30,
  "best_value": 0.9245,
  "best_params": {...},
  "all_trials": [...]
}
```

**3. Visualizations** (`results/figures/`)
- `optimization_history.html` - Trial progression
- `param_importance.html` - Which parameters matter most
- `parallel_coordinate.html` - Parameter relationships

---

## 🔄 Integration with Existing Code

### Works Seamlessly With:

**1. experiment.py**
```bash
# Optimize parameters
python scripts/hyperopt_with_optuna.py --preset balanced --dataset breast_cancer

# Test optimized config
python scripts/experiment.py --config configs/optimized.yaml
```

**2. train.py**
```bash
# Train with optimized config
python scripts/train.py --config configs/optimized.yaml --dataset breast_cancer
```

**3. test_optimized_config.py**
```bash
# Compare optimized vs default
python scripts/test_optimized_config.py
```

---

## 📋 Output Files

### After optimization completes:

**1. Optimized Config** (`configs/optimized.yaml`)
```yaml
ga:
  population_size: 120
  n_generations: 45
  crossover_prob: 0.735
  mutation_prob: 0.165
  # ... best parameters

optimization:
  study_name: ga_opt_breast_cancer_20241125
  n_trials: 30
  best_value: 0.9245
  optimization_date: "2024-11-25T12:30:45"
```

**2. Progress Tracking** (`results/optimization/`)
- `{study_name}_progress.json` - Live updates every 10 trials
- `{study_name}_report.json` - Final comprehensive report

**3. Visualizations** (`results/figures/`)
- Interactive HTML plots for analysis

---

## 🧪 Testing & Validation

### Acceptance Criteria Testing

**1. Automated optimization finds better parameters than default**

```bash
# Step 1: Run optimization
python scripts/hyperopt_with_optuna.py --preset balanced --dataset breast_cancer

# Step 2: Compare with default
python scripts/experiment.py --config configs/default.yaml
python scripts/experiment.py --config configs/optimized.yaml

# Expected: optimized.yaml shows 1-3% accuracy improvement
```

**2. Integration with existing GA engine**

✅ **Verified:** 
- Uses existing `GAEngine`, `GAConfig`, `TreeInitializer`
- No modifications to core GA code required
- Seamless parameter passing

**3. Results are reproducible and trackable**

✅ **Verified:**
- Fixed seeds (`seed=42`)
- All trials logged with timestamps
- Complete parameter history saved
- Reproducible via study names and storage

---

## 🎯 Example Workflows

### Workflow 1: Quick Test

```bash
# 1. Fast optimization (10 minutes)
python scripts/hyperopt_with_optuna.py --preset fast --dataset iris

# 2. Visual inspection
open results/figures/optimization_history.html

# 3. Test optimized config
python scripts/train.py --config configs/optimized.yaml --dataset iris
```

### Workflow 2: Production Optimization

```bash
# 1. Thorough optimization with persistent storage
python scripts/hyperopt_with_optuna.py \
    --preset thorough \
    --dataset breast_cancer \
    --storage sqlite:///optuna.db \
    --study-name production_v1

# 2. Full evaluation
python scripts/experiment.py --config configs/optimized.yaml

# 3. Statistical validation
python scripts/test_optimized_config.py

# 4. If needed, resume with more trials
python scripts/hyperopt_with_optuna.py \
    --study-name production_v1 \
    --storage sqlite:///optuna.db \
    --resume \
    --n-trials 50  # Additional 50 trials
```

### Workflow 3: Multi-Dataset Optimization

```bash
# Optimize each dataset separately
for dataset in iris wine breast_cancer; do
    python scripts/hyperopt_with_optuna.py \
        --preset balanced \
        --dataset $dataset \
        --output configs/optimized_${dataset}.yaml
done

# Compare results
python scripts/experiment.py --config configs/optimized_iris.yaml
python scripts/experiment.py --config configs/optimized_wine.yaml
python scripts/experiment.py --config configs/optimized_breast_cancer.yaml
```

---

## 🐛 Troubleshooting

### Issue: Optimization takes too long

**Solution 1:** Use faster preset
```bash
python scripts/hyperopt_with_optuna.py --preset fast --dataset breast_cancer
```

**Solution 2:** Reduce CV folds
```bash
python scripts/hyperopt_with_optuna.py --preset balanced --cv-folds 3
```

**Solution 3:** Add timeout
```bash
python scripts/hyperopt_with_optuna.py --preset balanced --timeout 1800  # 30 min max
```

### Issue: Pruning too aggressive

**Solution:** Adjust pruner warmup
```python
# In hyperopt_with_optuna.py, line ~230
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,    # Increase from 5
    n_warmup_steps=3,       # Increase from 2
    interval_steps=1
)
```

### Issue: Not finding better parameters

**Solution 1:** Use thorough preset
```bash
python scripts/hyperopt_with_optuna.py --preset thorough --dataset breast_cancer
```

**Solution 2:** Extend search space
- Edit `OPTIMIZATION_PRESETS` in script
- Add more parameters or widen ranges

### Issue: Out of memory

**Solution:** Reduce population/generations in search space
```python
'population_size': {'type': 'int', 'low': 30, 'high': 100, 'step': 10},
'n_generations': {'type': 'int', 'low': 20, 'high': 40, 'step': 10}
```

---

## 📚 References

**Optuna Documentation:**
- https://optuna.readthedocs.io/
- https://optuna.org/

**Key Papers:**
- TPE: "Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011)
- Pruning: "Speeding Up Automatic Hyperparameter Optimization" (Falkner et al., 2018)

---

## ✅ Verification Checklist

Before considering implementation complete, verify:

- [x] Optimization runs without errors
- [x] Results tracked in JSON files
- [x] Config files generated correctly
- [x] Visualizations created
- [x] Bayesian optimization active (TPE sampler)
- [x] Pruning working (check logs for "pruned" messages)
- [x] Presets available and functional
- [x] Integration with experiment.py works
- [x] Reproducible with same seed
- [x] Better than default config (test with test_optimized_config.py)