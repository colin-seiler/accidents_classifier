import numpy as np
import pandas as pd
import optuna
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

import mlflow
from mlflow.models import infer_signature
from optuna.samplers import TPESampler

from src.data.build_database import PROJECT_ROOT
from src.utils.pipelines import build_preprocessing, make_estimator_for_name
from src.utils.mlflow import set_mlflow, log_mlflow_helper, MLFLOW_TRACKING_URI
from src.models.opt import OBJ_FUNCTIONS
from src.models.utils import train_eval

OUTPUT = 'severity'
MODELS = ["logistic", "ridge", "xgboost", "lightgbm"]
SCORERS = {"f1": "f1_macro", "acc": "balanced_accuracy"}

MODELS_ROOT = PROJECT_ROOT / "models"

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train(df, pca=False, tune=False):
    if mlflow.active_run() is not None:
        mlflow.end_run()

    preprocessing = build_preprocessing(50)
    set_mlflow(MLFLOW_TRACKING_URI, 'accident_prediction_model')

    X = df.drop(columns=[OUTPUT])
    y = df[OUTPUT].astype(int) - 1

    split = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    X_train = split[0]
    X_test = split[1]
    y_train = split[2]
    y_test = split[3]

    models = {}
    for name in MODELS:
        print(f'🏋️‍♂️ Training Model: {name} with pca={pca} and tune={tune}')
        if pca:
            est = make_estimator_for_name(name, 4)
            models[name] = make_pipeline(clone(preprocessing), PCA(n_components=0.95), est)

            if tune:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(seed=42),
                    study_name=f"{name}_pca_study"
                )

                study.optimize(
                    lambda trial: OBJ_FUNCTIONS[name](trial, preprocessing, X_train, y_train, True),
                    n_trials=10,
                    show_progress_bar=True
                )
                best_params = study.best_params

                models[name].set_params(**best_params)
        else:
            est = make_estimator_for_name(name, 4)
            models[name] = make_pipeline(clone(preprocessing), est)

            if tune:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(seed=42),
                    study_name=f"{name}_study"
                )

                study.optimize(
                    lambda trial: OBJ_FUNCTIONS[name](trial, preprocessing, X_train, y_train, False),
                    n_trials=10,
                    show_progress_bar=True
                )
                best_params = study.best_params

                models[name].set_params(**best_params)

    results = {}
    if pca:
        run = f'_pca_optuna'
        if tune:
            run = f'_pca_optuna'
    else:
        run = f'_baseline'
        if tune:
            run = f'_baseline_optuna'

    for name, pipeline in models.items():
        run_name = name+run
        if mlflow.active_run() is not None:
            mlflow.end_run()

        results[name] = train_eval(pipeline, *split)
        if pca:
            with mlflow.start_run(run_name=run_name, nested=True):
                signature = infer_signature(X_train, pipeline.predict(X_train))
                log_mlflow_helper(name, results[name], True, signature, X_train)
        else:
            with mlflow.start_run(run_name=run_name, nested=True):
                signature = infer_signature(X_train, pipeline.predict(X_train))
                log_mlflow_helper(name, results[name], False, signature, X_train)

    return results