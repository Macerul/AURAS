"""
Utility Metrics: Measures ML model performance on real vs augmented data
Enhanced with 6 additional ML models
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import time
from sklearn.pipeline import Pipeline
import copy

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available. Using GradientBoostingClassifier as fallback.")

def calculate_utility_metrics(df_original, df_augmented, target_column, model_configs,
                              use_cv=False, cv_folds=5, cv_metric='accuracy'):
    """
    Calculate utility metrics by training ML models
    """
    results = {}

    if target_column and target_column in df_original.columns and target_column in df_augmented.columns:
        pass
    elif 'category' in df_original.columns and 'category' in df_augmented.columns:
        target_column = 'category'
    else:
        last_orig = df_original.columns[-1]
        last_aug = df_augmented.columns[-1]
        if last_orig != last_aug:
            return {
                'error': f'Last column mismatch between original ("{last_orig}") and augmented ("{last_aug}") datasets'}
        target_column = last_orig

    # Prepare data
    try:
        X_orig, y_orig = prepare_data(df_original, target_column)
        X_aug, y_aug = prepare_data(df_augmented, target_column)

        # Combine datasets for mixed training
        X_combined = pd.concat([X_orig, X_aug], ignore_index=True)
        y_combined = pd.concat([y_orig, y_aug], ignore_index=True)

        # Split original data for testing (held-out test set)
        stratify_arg = y_orig if len(np.unique(y_orig)) > 1 else None
        X_train_orig, X_test, y_train_orig, y_test = train_test_split(
            X_orig, y_orig, test_size=0.3, random_state=42, stratify=stratify_arg
        )

        # Train models on different datasets
        model_results = []

        for config in model_configs:
            model_type = config.get('type')
            params = config.get('params', {}) or {}

            # Per-model cv override (if provided)
            per_model_cv = int(config.get('cv', cv_folds))

            print(f"Training {model_type} (cv={per_model_cv}, use_cv={use_cv}, requested_metric={cv_metric})...")

            # Train on original only
            result_orig = train_and_evaluate(
                model_type, params, X_train_orig, y_train_orig, X_test, y_test,
                'Original Only', cv_folds=per_model_cv, cv_metric=cv_metric, use_cv=use_cv
            )

            # Train on augmented only
            stratify_aug = y_aug if len(np.unique(y_aug)) > 1 else None
            X_train_aug, _, y_train_aug, _ = train_test_split(
                X_aug, y_aug, test_size=0.3, random_state=42, stratify=stratify_aug
            )
            result_aug = train_and_evaluate(
                model_type, params, X_train_aug, y_train_aug, X_test, y_test,
                'Augmented Only', cv_folds=per_model_cv, cv_metric=cv_metric, use_cv=use_cv
            )

            # Train on combined (original + augmented)
            stratify_comb = y_combined if len(np.unique(y_combined)) > 1 else None
            X_train_comb, _, y_train_comb, _ = train_test_split(
                X_combined, y_combined, test_size=0.3, random_state=42,
                stratify=stratify_comb
            )
            result_comb = train_and_evaluate(
                model_type, params, X_train_comb, y_train_comb, X_test, y_test,
                'Original + Augmented', cv_folds=per_model_cv, cv_metric=cv_metric, use_cv=use_cv
            )

            # Normalize missing CV fields
            def _extract_cv_flat(res):
                cv = res.get('cv', {})
                if isinstance(cv, dict):
                    req = cv.get('requested_metric')
                    if isinstance(req, dict) and 'mean' in req:
                        res['cv_mean'] = float(req.get('mean')) if req.get('mean') is not None else None
                        res['cv_std'] = float(req.get('std')) if req.get('std') is not None else None
                    else:
                        if 'accuracy_mean' in cv:
                            res['cv_mean'] = float(cv.get('accuracy_mean', None))
                            res['cv_std'] = float(cv.get('accuracy_std', None))
                        else:
                            res['cv_mean'] = None
                            res['cv_std'] = None
                else:
                    res['cv_mean'] = None
                    res['cv_std'] = None

            _extract_cv_flat(result_orig)
            _extract_cv_flat(result_aug)
            _extract_cv_flat(result_comb)

            # Compare results
            comparison = {
                'model_type': model_type,
                'parameters': params,
                'original_only': result_orig,
                'augmented_only': result_aug,
                'combined': result_comb,
                'improvement': {
                    'accuracy': result_comb.get('accuracy', 0) - result_orig.get('accuracy', 0),
                    'f1': result_comb.get('f1_weighted', 0) - result_orig.get('f1_weighted', 0)
                },
                'quality_assessment': assess_utility_quality(result_orig, result_aug, result_comb)
            }

            model_results.append(comparison)

        if len(model_results) == 0:
            return {'error': 'No models configured'}

        best_model = max(model_results, key=lambda x: x['combined'].get('accuracy', 0))

        results = {
            'model_results': model_results,
            'best_model': {
                'type': best_model.get('model_type'),
                'accuracy': best_model.get('combined', {}).get('accuracy', 0),
                'f1': best_model.get('combined', {}).get('f1_weighted', 0)
            },
            'average_improvement': {
                'accuracy': float(np.mean([m['improvement']['accuracy'] for m in model_results])),
                'f1': float(np.mean([m['improvement']['f1'] for m in model_results]))
            },
            'dataset_info': {
                'original_samples': int(len(X_orig)),
                'augmented_samples': int(len(X_aug)),
                'test_samples': int(len(X_test)),
                'n_features': int(X_orig.shape[1]),
                'n_classes': int(len(np.unique(y_orig)))
            },
            'overall_utility_score': float(calculate_overall_utility_score(model_results)),
            'recommendation': generate_utility_recommendation(model_results),
            'use_cv': bool(use_cv),
            'cv_folds': int(cv_folds),
            'cv_metric': cv_metric
        }

        return results

    except Exception as e:
        return {'error': str(e)}


# Aggiorna la funzione get_model() per includere i nuovi modelli
def get_model(model_type, params):
    """Get sklearn model instance with enhanced model support"""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, BaggingClassifier
    from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier


    models = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'decision_tree': DecisionTreeClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'extra_trees': ExtraTreesClassifier,
        'mlp': MLPClassifier,
        'linear_svc': LinearSVC,
        'naive_bayes': GaussianNB,
        'ada_boost': AdaBoostClassifier,
        'hist_gradient_boosting': HistGradientBoostingClassifier,
        'ridge_classifier': RidgeClassifier,
        'sgd_classifier': SGDClassifier,
        'lda': LinearDiscriminantAnalysis,
        'qda': QuadraticDiscriminantAnalysis,
        'passive_aggressive': PassiveAggressiveClassifier,
        'perceptron': Perceptron,
        'bagging': BaggingClassifier,
        'dummy': DummyClassifier,
        'gaussian_process': GaussianProcessClassifier
    }

    # Handle optional models with fallbacks
    if model_type == 'xgboost':
        if XGB_AVAILABLE:
            return xgb.XGBClassifier(**params)
        else:
            print("⚠️ Using GradientBoostingClassifier as XGBoost fallback")
            return GradientBoostingClassifier(**params)

    if model_type == 'calibrated_cv':
        # Usa Logistic Regression come base per calibrated classifier
        base_estimator = LogisticRegression(max_iter=1000, random_state=42)
        return CalibratedClassifierCV(base_estimator, **params)

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    # Copy params so we don't mutate input dict
    params_copy = copy.deepcopy(params) if params is not None else {}

    # Add default parameters for different models
    if model_type == 'logistic_regression':
        params_copy.setdefault('solver', 'lbfgs')
        params_copy.setdefault('max_iter', 1000)

    elif model_type == 'mlp':
        params_copy.setdefault('hidden_layer_sizes', (100,))
        params_copy.setdefault('max_iter', 1000)
        params_copy.setdefault('random_state', 42)

    elif model_type == 'linear_svc':
        params_copy.setdefault('max_iter', 1000)
        params_copy.setdefault('random_state', 42)

    elif model_type in ['random_forest', 'extra_trees', 'gradient_boosting', 'ada_boost', 'bagging']:
        params_copy.setdefault('n_estimators', 100)
        params_copy.setdefault('random_state', 42)

    elif model_type == 'decision_tree':
        params_copy.setdefault('random_state', 42)

    elif model_type == 'svm':
        params_copy.setdefault('kernel', 'rbf')
        params_copy.setdefault('gamma', 'scale')
        params_copy.setdefault('random_state', 42)

    elif model_type == 'hist_gradient_boosting':
        params_copy.setdefault('max_iter', 100)
        params_copy.setdefault('random_state', 42)

    elif model_type == 'ridge_classifier':
        params_copy.setdefault('alpha', 1.0)
        params_copy.setdefault('random_state', 42)

    elif model_type == 'sgd_classifier':
        params_copy.setdefault('max_iter', 1000)
        params_copy.setdefault('random_state', 42)

    elif model_type in ['passive_aggressive', 'perceptron']:
        params_copy.setdefault('max_iter', 1000)
        params_copy.setdefault('random_state', 42)

    elif model_type == 'calibrated_cv':
        params_copy.setdefault('cv', 3)
        params_copy.setdefault('method', 'sigmoid')

    elif model_type == 'dummy':
        params_copy.setdefault('strategy', 'stratified')

    elif model_type == 'gaussian_process':
        params_copy.setdefault('random_state', 42)

    elif model_type == 'isolation_forest':
        params_copy.setdefault('n_estimators', 100)
        params_copy.setdefault('random_state', 42)
        params_copy.setdefault('contamination', 'auto')

    return models[model_type](**params_copy)

def train_and_evaluate(model_type, params, X_train, y_train, X_test, y_test, dataset_name,
                       cv_folds=5, cv_metric='accuracy', use_cv=False):
    """
    Train model and evaluate on test set AND perform cross-validation
    """
    model = get_model(model_type, params or {})
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    # Define scoring metrics for cross-validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }

    cv_results = {}
    if use_cv and cv_folds and cv_folds >= 2 and len(np.unique(y_train)) > 1 and len(y_train) >= cv_folds:
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_res = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=skf,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1
            )

            # Aggregate means and stds
            cv_results = {
                'n_folds': int(cv_folds),
                'accuracy_mean': float(np.mean(cv_res['test_accuracy'])),
                'accuracy_std': float(np.std(cv_res['test_accuracy'])),
                'precision_mean': float(np.mean(cv_res['test_precision'])),
                'precision_std': float(np.std(cv_res['test_precision'])),
                'recall_mean': float(np.mean(cv_res['test_recall'])),
                'recall_std': float(np.std(cv_res['test_recall'])),
                'f1_mean': float(np.mean(cv_res['test_f1'])),
                'f1_std': float(np.std(cv_res['test_f1'])),
                'raw_cv_scores': {
                    'accuracy': cv_res['test_accuracy'].tolist(),
                    'precision': cv_res['test_precision'].tolist(),
                    'recall': cv_res['test_recall'].tolist(),
                    'f1': cv_res['test_f1'].tolist()
                }
            }

            # Map requested cv_metric
            metric_key_map = {
                'accuracy': 'test_accuracy',
                'f1_weighted': 'test_f1',
                'precision_weighted': 'test_precision',
                'recall_weighted': 'test_recall'
            }
            requested_key = metric_key_map.get(cv_metric, None)
            if requested_key and requested_key in cv_res:
                vals = np.array(cv_res[requested_key])
                cv_results['requested_metric'] = {
                    'name': cv_metric,
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'per_fold': vals.tolist()
                }
            else:
                if cv_metric == 'f1_weighted' and 'test_f1' in cv_res:
                    vals = np.array(cv_res['test_f1'])
                    cv_results['requested_metric'] = {'name': cv_metric, 'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'per_fold': vals.tolist()}
                else:
                    cv_results['requested_metric'] = {'name': cv_metric, 'mean': None, 'std': None, 'per_fold': []}

        except Exception as e:
            cv_results = {'error': f'CV failed: {str(e)}'}
    else:
        if use_cv:
            cv_results = {'warning': 'Not enough samples/classes for CV or cv_folds < 2'}
        else:
            cv_results = {'info': 'CV disabled'}

    # Fit on training set and evaluate on test set
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    average = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    result = {
        'dataset': dataset_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_weighted': float(f1),
        'training_time_seconds': float(training_time),
        'confusion_matrix': cm.tolist(),
        'n_train_samples': int(len(X_train)),
        'n_test_samples': int(len(X_test)),
        'cv': cv_results
    }

    # convenience flat fields for common UI access
    try:
        req = cv_results.get('requested_metric', {})
        result['cv_mean'] = float(req.get('mean')) if req.get('mean') is not None else None
        result['cv_std'] = float(req.get('std')) if req.get('std') is not None else None
    except Exception:
        result['cv_mean'] = None
        result['cv_std'] = None

    return result

# Le funzioni rimanenti restano invariate
def prepare_data(df, target_column):
    """Prepare features and target for ML"""
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    X = X.fillna(X.mean())

    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    return X, y

def assess_utility_quality(orig_result, aug_result, comb_result):
    """Assess the quality of augmentation based on model performance"""
    improvement = comb_result['accuracy'] - orig_result['accuracy']
    aug_quality = aug_result['accuracy']

    if improvement > 0.05 and aug_quality > 0.6:
        return 'high'
    elif improvement > 0.02 or aug_quality > 0.5:
        return 'medium'
    else:
        return 'low'

def calculate_overall_utility_score(model_results):
    """Calculate overall utility score (0-1)"""
    avg_improvement = np.mean([m['improvement']['accuracy'] for m in model_results])
    avg_combined_acc = np.mean([m['combined']['accuracy'] for m in model_results])

    improvement_score = min(1.0, max(0.0, (avg_improvement + 0.1) / 0.2))
    performance_score = avg_combined_acc

    overall = 0.6 * improvement_score + 0.4 * performance_score
    return float(overall)

def generate_utility_recommendation(model_results):
    """Generate recommendation based on utility analysis"""
    avg_improvement = np.mean([m['improvement']['accuracy'] for m in model_results])

    if avg_improvement > 0.05:
        return {
            'verdict': 'Excellent utility',
            'message': 'Augmented data significantly improves model performance. Recommended for production use.',
            'icon': '✅'
        }
    elif avg_improvement > 0.02:
        return {
            'verdict': 'Good utility',
            'message': 'Augmented data provides measurable improvement. Consider using for training.',
            'icon': '👍'
        }
    elif avg_improvement > -0.02:
        return {
            'verdict': 'Neutral utility',
            'message': 'Augmented data does not significantly change model performance. May still be useful for robustness.',
            'icon': '⚠️'
        }
    else:
        return {
            'verdict': 'Poor utility',
            'message': 'Augmented data degrades model performance. Review augmentation method.',
            'icon': '❌'
        }