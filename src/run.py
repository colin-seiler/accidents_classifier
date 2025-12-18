import argparse
import time
import warnings
import logging

from src.data.build_database import PROJECT_ROOT
from src.data.load_database import load_database
from src.models.train import train
from src.utils.helper import save_model

warnings.filterwarnings(
    "ignore",
    message=".*Inferred schema contains integer column.*"
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.tracking").setLevel(logging.ERROR)
logging.getLogger("mlflow.store").setLevel(logging.ERROR)
logging.getLogger("mlflow.store.model_registry.abstract_store").setLevel(logging.ERROR)

MODELS_ROOT = PROJECT_ROOT / 'models'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train accident severity classifiers"
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Enable PCA in the pipeline"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna hyperparameter tuning"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all combinations of PCA and tuning"
    )
    args = parser.parse_args()

    start_time = time.monotonic()

    df = load_database()
    runs = []
    if args.tune:
        runs.append((False, True))
        runs.append((True, True))
    elif args.pca:
        runs.append((False, False))
        runs.append((True, False))
    elif args.all:
        runs.append((False, False))
        runs.append((False, True))
        runs.append((True, False))
        runs.append((True, True))
    else:
        runs.append((False, False))

    all_results = {}
    for pca_flag, tune_flag in runs:
        print(
            f"\n{'='*80}\n"
            f"🏃 Starting up training...\n"
            f"{'='*80}"
        )
        
        all_results.update(train(
            df,
            pca=pca_flag,
            tune=tune_flag
        ))

    global_best_name = max(all_results, key=lambda k: all_results[k]["test_f1"])
    global_best_f1 = all_results[global_best_name]["test_f1"]
    global_best_cv_f1 = all_results[global_best_name]["cv_f1"]
    global_best_pipeline = all_results[global_best_name]["pipeline"]

    uses_pca = "_pca" in global_best_name

    print("\n" + "=" * 80)
    print("GLOBAL BEST MODEL")
    print("=" * 80)
    print(f"Global best model key: {global_best_name}")
    print(f"Global best CV F1:    {global_best_cv_f1:,.2f}")
    print(f"Global best Test F1:  {global_best_f1:,.2f}")
    print(f"Uses PCA:              {uses_pca}")

    if args.tune or args.all:
        save_model(global_best_pipeline, MODELS_ROOT / 'global_best_model_optuna.pkl')
    if args.pca:
        save_model(global_best_pipeline, MODELS_ROOT / 'global_best_model_pca.pkl')
    else:
        save_model(global_best_pipeline, MODELS_ROOT / 'global_best_model.pkl')

    end_time = time.monotonic()

    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"Elapsed time: {minutes} minutes and {seconds:.2f} seconds")