from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.transform(X)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

def build_preprocessing(k = 10):
    preprocessing = ColumnTransformer(
        [
            ("geo", ClusterSimilarity(n_clusters=k, gamma=1.0, random_state=42), ["latitude", "longitude"]),
            ("cat", cat_pipeline, ['state', 'weather_condition'])
        ],
        remainder=default_num_pipeline,
    )
    return preprocessing

def make_estimator_for_name(name: str, n_classes: int):
    """
    Factory for multiclass classifiers used in experiments.
    PCA is handled in the preprocessing pipeline, NOT here.
    """
    if name == "logistic":
        return LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
    elif name == "ridge":
        return RidgeClassifier(
            random_state=42
        )
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(
            random_state=42
        )
    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(
            random_state=42
        )
    elif name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )
    elif name == "lightgbm":
        return LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {name}")