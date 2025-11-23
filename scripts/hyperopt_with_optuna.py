"""Automatic hyperparameter tuning with Optuna - FIXED VERSION."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import optuna
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ga_trees.ga.engine import GAEngine, GAConfig, TreeInitializer, Mutation
from ga_trees.fitness.calculator import FitnessCalculator, TreePredictor


def objective(trial):
    """Optuna objective function - FIXED."""
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Suggest hyperparameters
    population_size = trial.suggest_int('population_size', 50, 150, step=10)
    n_generations = trial.suggest_int('n_generations', 20, 60, step=10)
    crossover_prob = trial.suggest_float('crossover_prob', 0.5, 0.9)
    mutation_prob = trial.suggest_float('mutation_prob', 0.1, 0.3)
    tournament_size = trial.suggest_int('tournament_size', 2, 5)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
    accuracy_weight = trial.suggest_float('accuracy_weight', 0.6, 0.9)
    
    # Setup
    n_features = X.shape[1]
    feature_ranges = {i: (X[:, i].min(), X[:, i].max()) for i in range(n_features)}
    
    ga_config = GAConfig(
        population_size=population_size,
        n_generations=n_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        tournament_size=tournament_size
    )
    
    # FIXED: Added missing arguments
    initializer = TreeInitializer(
        n_features=n_features,
        n_classes=2,
        max_depth=max_depth,
        min_samples_split=min_samples_split,  # ADDED
        min_samples_leaf=min_samples_leaf     # ADDED
    )
    
    fitness_calc = FitnessCalculator(
        accuracy_weight=accuracy_weight,
        interpretability_weight=1.0 - accuracy_weight
    )
    
    mutation = Mutation(n_features=n_features, feature_ranges=feature_ranges)
    
    # Train and evaluate with 3-fold CV (faster than 5-fold)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            ga_engine = GAEngine(ga_config, initializer, 
                               fitness_calc.calculate_fitness, mutation)
            best_tree = ga_engine.evolve(X_train, y_train, verbose=False)
            
            predictor = TreePredictor()
            y_pred = predictor.predict(best_tree, X_test)
            
            scores.append(accuracy_score(y_test, y_pred))
        except Exception as e:
            # If training fails, return low score
            print(f"Trial failed: {e}")
            return 0.5
    
    return np.mean(scores)


def main():
    """Run hyperparameter optimization."""
    print("="*70)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*70)
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    print("\nStarting optimization...")
    print("This will run 30 trials (estimated time: 15-20 minutes)")
    print("Each trial trains GA with 3-fold CV\n")
    
    try:
        study.optimize(objective, n_trials=30, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Saving partial results...")
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nBest Trial:")
    print(f"  Value (Accuracy): {study.best_trial.value:.4f}")
    print(f"  Number: {study.best_trial.number}")
    
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key:20s}: {value}")
    
    # Show top 5 trials
    print(f"\n{'='*70}")
    print("TOP 5 TRIALS")
    print(f"{'='*70}")
    
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nlargest(5, 'value')
    
    print(top_trials[['number', 'value', 'params_population_size', 
                      'params_n_generations', 'params_max_depth']].to_string(index=False))
    
    # Visualizations (if plotly available)
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        print("\nCreating visualizations...")
        
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html('results/figures/optuna_history.html')
        print("✓ Saved: optuna_history.html")
        
        # Parameter importance
        fig2 = plot_param_importances(study)
        fig2.write_html('results/figures/optuna_importance.html')
        print("✓ Saved: optuna_importance.html")
        
        # Parallel coordinate plot
        try:
            from optuna.visualization import plot_parallel_coordinate
            fig3 = plot_parallel_coordinate(study)
            fig3.write_html('results/figures/optuna_parallel.html')
            print("✓ Saved: optuna_parallel.html")
        except:
            pass
            
    except ImportError:
        print("\nNote: Install plotly for visualizations: pip install plotly")
    
    # Save best params to YAML
    try:
        import yaml
        
        config = {
            'ga': {
                'population_size': study.best_params['population_size'],
                'n_generations': study.best_params['n_generations'],
                'crossover_prob': study.best_params['crossover_prob'],
                'mutation_prob': study.best_params['mutation_prob'],
                'tournament_size': study.best_params['tournament_size'],
            },
            'tree': {
                'max_depth': study.best_params['max_depth'],
                'min_samples_split': study.best_params['min_samples_split'],
                'min_samples_leaf': study.best_params['min_samples_leaf'],
            },
            'fitness': {
                'accuracy_weight': study.best_params['accuracy_weight'],
                'interpretability_weight': 1.0 - study.best_params['accuracy_weight'],
            }
        }
        
        Path('configs').mkdir(exist_ok=True)
        with open('configs/optimized.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("\n✓ Saved best params to: configs/optimized.yaml")
        
    except ImportError:
        print("\nNote: Install pyyaml to save config: pip install pyyaml")
    
    # Save study
    import pickle
    with open('results/optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)
    print("✓ Saved study to: results/optuna_study.pkl")
    
    print("\n" + "="*70)
    print("Hyperparameter Optimization Complete! 🎉")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Review visualizations in results/figures/")
    print("2. Use optimized config: configs/optimized.yaml")
    print("3. Re-run experiments with optimized hyperparameters")


if __name__ == '__main__':
    main()