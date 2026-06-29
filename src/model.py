from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.config import LightGBMConfig, LogisticRegressionConfig
from src.data_io import ensure_parent_directory


Model = LGBMClassifier | Pipeline


def create_model(
    model_type: str,
    config: LightGBMConfig | LogisticRegressionConfig,
    seed: int,
) -> Model:
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
