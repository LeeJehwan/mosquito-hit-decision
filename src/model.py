from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.config import EnsembleConfig, LightGBMConfig, LogisticRegressionConfig
from src.data_io import ensure_parent_directory


class SoftVotingEnsemble:
    """확률 평균(soft voting) 앙상블. 1위 방법론.

    sklearn 호환 인터페이스(fit/predict_proba)를 직접 제공해 joblib 직렬화와
    기존 train/infer 파이프라인을 그대로 사용한다.
    """

    def __init__(self, estimators: list):
        self.estimators = estimators

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "SoftVotingEnsemble":
        for estimator in self.estimators:
            estimator.fit(features, labels)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.mean(
            [estimator.predict_proba(features)[:, 1] for estimator in self.estimators],
            axis=0,
        )
        return np.column_stack([1.0 - positive, positive])


Model = LGBMClassifier | Pipeline | SoftVotingEnsemble


def create_ensemble(config: EnsembleConfig, seed: int) -> SoftVotingEnsemble:
    lightgbm = LGBMClassifier(
        objective="binary",
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        random_state=seed,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    hist = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.hist_max_iter,
        max_leaf_nodes=config.hist_max_leaf_nodes,
        l2_regularization=config.hist_l2,
        early_stopping=False,
        random_state=seed,
    )
    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=tuple(config.mlp_hidden),
                    activation="relu",
                    alpha=config.mlp_alpha,
                    max_iter=config.mlp_max_iter,
                    early_stopping=config.mlp_early_stopping,
                    n_iter_no_change=15,
                    random_state=seed,
                ),
            ),
        ]
    )
    return SoftVotingEnsemble([lightgbm, hist, mlp])


def create_model(
    model_type: str,
    config: LightGBMConfig | LogisticRegressionConfig | EnsembleConfig,
    seed: int,
) -> Model:
    if model_type == "ensemble":
        if not isinstance(config, EnsembleConfig):
            raise TypeError("Ensemble requires EnsembleConfig")
        return create_ensemble(config, seed)
    if model_type == "lightgbm":
        if not isinstance(config, LightGBMConfig):
            raise TypeError("LightGBM requires LightGBMConfig")
        return LGBMClassifier(
            objective="binary",
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=seed,
            n_jobs=config.n_jobs,
            verbosity=-1,
        )
    if model_type == "logistic":
        if not isinstance(config, LogisticRegressionConfig):
            raise TypeError("Logistic regression requires LogisticRegressionConfig")
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config.C,
                        max_iter=config.max_iter,
                        solver=config.solver,
                        class_weight=None,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def create_tqdm_callback(description: str):
    progress = None

    def callback(environment) -> None:
        nonlocal progress
        if progress is None:
            progress = tqdm(
                total=environment.end_iteration - environment.begin_iteration,
                desc=description,
                unit="tree",
            )
        progress.update(1)
        if environment.iteration + 1 >= environment.end_iteration:
            progress.close()

    callback.order = 10
    callback.before_iteration = False
    return callback


def train_model(
    model: Model,
    features: pd.DataFrame,
    labels: pd.Series,
    show_progress: bool = False,
    description: str = "Training LightGBM",
) -> Model:
    if isinstance(model, LGBMClassifier):
        callbacks = [create_tqdm_callback(description)] if show_progress else None
        return model.fit(features, labels, callbacks=callbacks)
    return model.fit(features, labels)


def predict_hit_probabilities(model: Model, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if probabilities.shape[1] != 2:
        raise ValueError("Model must provide probabilities for two classes")
    return probabilities[:, 1]


def save_model(model: Model, path: Path) -> None:
    ensure_parent_directory(path)
    joblib.dump(model, path)


def load_model(path: Path) -> Model:
    return joblib.load(path)
