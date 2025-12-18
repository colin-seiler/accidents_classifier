import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

SCORERS = {"f1": "f1_macro", "acc": "balanced_accuracy"}

def objective_scorer(pipeline, X_train, y_train):
    return cross_validate(pipeline, X_train, y_train, cv=3, scoring=SCORERS, n_jobs=-1, return_train_score=False)

def optional_use_pca(preprocessing, estimator, use_pca, pca_components):
    steps = [clone(preprocessing)]
    if use_pca:
        steps.append(PCA(n_components=pca_components, random_state=42))
    steps.append(estimator)
    return make_pipeline(*steps)

def objective_logistic(trial, preprocessing, X_train, y_train, use_pca):
    C = trial.suggest_float("logisticregression__C", 1e-3, 100.0, log=True)
    if use_pca:
        pca_components = trial.suggest_float("pca__n_components", 0.90, 0.99)
    else:
        pca_components = None

    estimator = LogisticRegression(
        C=C,
        max_iter=2000,
        random_state=42
    )

    pipeline = optional_use_pca(preprocessing, estimator, use_pca, pca_components)

    cv_results = objective_scorer(pipeline, X_train, y_train)

    return cv_results["test_f1"].mean()

def objective_ridge(trial, preprocessing, X_train, y_train, use_pca):
    alpha = trial.suggest_float("ridgeclassifier__alpha", 1e-3, 100.0, log=True)
    if use_pca:
        pca_components = trial.suggest_float("pca__n_components", 0.90, 0.99)
    else:
        pca_components = None

    estimator = RidgeClassifier(
            alpha=alpha,
            random_state=42
        )
    
    pipeline = optional_use_pca(preprocessing, estimator, use_pca, pca_components)

    cv_results = objective_scorer(pipeline, X_train, y_train)

    return cv_results["test_f1"].mean()

def objective_xgboost(trial, preprocessing, X_train, y_train, use_pca):
    learning_rate = trial.suggest_float("xgbclassifier__learning_rate", 0.05, 0.3)
    max_depth = trial.suggest_int("xgbclassifier__max_depth", 3, 8)
    n_estimators = trial.suggest_int("xgbclassifier__n_estimators", 100, 300, step=50)
    if use_pca:
        pca_components = trial.suggest_float("xgbclassifier__n_components", 0.90, 0.99)
    else:
        pca_components = None

    estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss"
        )
    
    pipeline = optional_use_pca(preprocessing, estimator, use_pca, pca_components)

    cv_results = objective_scorer(pipeline, X_train, y_train)

    return cv_results["test_f1"].mean()

def objective_lightgbm(trial, preprocessing, X_train, y_train, use_pca):
    learning_rate = trial.suggest_float("lgbmclassifier__learning_rate", 0.05, 0.3)
    num_leaves = trial.suggest_int("lgbmclassifier__num_leaves", 20, 80)
    n_estimators = trial.suggest_int("lgbmclassifier__n_estimators", 100, 300, step=50)
    if use_pca:
        pca_components = trial.suggest_float("pca__n_components", 0.90, 0.99)
    else:
        pca_components = None

    estimator = LGBMClassifier(
            objective="multiclass",
            num_class=len(np.unique(y_train)),
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            force_col_wise=True,
        )
    
    pipeline = optional_use_pca(preprocessing, estimator, use_pca, pca_components)

    cv_results = objective_scorer(pipeline, X_train, y_train)

    return cv_results["test_f1"].mean()

OBJ_FUNCTIONS = {
    "logistic": objective_logistic,
    "ridge": objective_ridge,
    "xgboost": objective_xgboost,
    "lightgbm": objective_lightgbm
}