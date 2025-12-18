import argparse

from src.data.load_database import load_database
from src.models.train import train

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

    df = load_database()
    if args.all:
        runs = [(False, False),(False, True),(True, False),(True, True)]
    else:
        runs = [(args.pca, args.tune)]

    for pca_flag, tune_flag in runs:
        print(
            f"\n{'='*80}\n"
            f"Running training with pca={pca_flag}, tune={tune_flag}\n"
            f"{'='*80}"
        )
        
        train(
            df,
            pca=pca_flag,
            tune=tune_flag
        )