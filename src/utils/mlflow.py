import os
import mlflow
from dotenv import load_dotenv

load_dotenv(
    dotenv_path="/content/gdrive/MyDrive/Colab Notebooks/housing_fall2025/notebooks/.env",
    override=True
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

def set_mlflow(URI, experiment):
    if MLFLOW_TRACKING_USERNAME:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    if MLFLOW_TRACKING_PASSWORD:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
        
    mlflow.set_tracking_uri(URI)
    mlflow.set_experiment(experiment)

def log_mlflow_helper(name, name_dict, pca, signature, X_train):
    pipeline = name_dict['pipeline']

    mlflow.log_param("model_family", name)
    mlflow.log_param("uses_pca", pca)

    est_step_name = list(pipeline.named_steps.keys())[-1]
    est = pipeline.named_steps[est_step_name]
    est_params = {f"{est_step_name}__{k}": v for k, v in est.get_params().items()}
    mlflow.log_params(est_params)

    for key, val in name_dict.items():
        if key == 'pipeline':
            continue
        mlflow.log_metric(key, val)

    if pca:
        pca_step = pipeline.named_steps["pca"]
        mlflow.log_param("pca__n_components", pca_step.n_components)

        mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="accident_model_pca",
                signature=signature,
                input_example=X_train,
                registered_model_name=f"{name}_pipeline_pca",
            )
    else:    
        mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="accident_model",
                signature=signature,
                input_example=X_train,
                registered_model_name=f"{name}_pipeline",
            )