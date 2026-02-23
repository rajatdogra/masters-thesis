"""
ML Models Module
XGBoost, Random Forest, LightGBM, Neural Network with Optuna tuning.
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# XGBoost
# ============================================================
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    random_state: int = 42,
):
    """Train XGBoost with Optuna hyperparameter tuning."""
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }

        model = xgb.XGBRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        return _rmse(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"XGBoost best RMSE: {study.best_value:.2f}")
    logger.info(f"XGBoost best params: {study.best_params}")

    # Retrain with best params
    best_model = xgb.XGBRegressor(
        **study.best_params,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return best_model, study


# ============================================================
# Random Forest
# ============================================================
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
):
    """Train Random Forest with Optuna hyperparameter tuning."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }

        model = RandomForestRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return _rmse(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"RF best RMSE: {study.best_value:.2f}")
    logger.info(f"RF best params: {study.best_params}")

    best_model = RandomForestRegressor(
        **study.best_params,
        random_state=random_state,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train)
    return best_model, study


# ============================================================
# LightGBM
# ============================================================
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
):
    """Train LightGBM with Optuna hyperparameter tuning."""
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        }

        model = lgb.LGBMRegressor(
            **params,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        return _rmse(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"LightGBM best RMSE: {study.best_value:.2f}")
    logger.info(f"LightGBM best params: {study.best_params}")

    best_model = lgb.LGBMRegressor(
        **study.best_params,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    return best_model, study


# ============================================================
# CatBoost
# ============================================================
def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
):
    """Train CatBoost with Optuna hyperparameter tuning."""
    from catboost import CatBoostRegressor

    def objective(trial):
        params = {
            "iterations":        trial.suggest_int("iterations", 200, 1500),
            "depth":             trial.suggest_int("depth", 4, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":   trial.suggest_float("random_strength", 1e-4, 10.0, log=True),
            "border_count":      trial.suggest_int("border_count", 32, 255),
        }
        model = CatBoostRegressor(
            **params,
            random_seed=random_state,
            eval_metric="RMSE",
            verbose=0,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        return _rmse(y_val, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"CatBoost best RMSE: {study.best_value:.2f}")
    logger.info(f"CatBoost best params: {study.best_params}")

    best_model = CatBoostRegressor(
        **study.best_params,
        random_seed=random_state,
        eval_metric="RMSE",
        verbose=0,
        early_stopping_rounds=50,
    )
    best_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return best_model, study


# ============================================================
# LightGBM Quantile Regression (prediction intervals)
# ============================================================
def train_lgbm_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: list = None,
    random_state: int = 42,
) -> dict:
    """
    Train LightGBM quantile regressors at each requested quantile.

    Returns dict: quantile (float) -> fitted LGBMRegressor.
    Hyperparameters are tuned once at the median (q=0.5) and reused.
    """
    import lightgbm as lgb

    if quantiles is None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Tune on median to get representative hyperparams
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        }
        model = lgb.LGBMRegressor(
            **params,
            objective="quantile",
            alpha=0.5,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        return _rmse(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params

    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            **best_params,
            objective="quantile",
            alpha=q,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        models[q] = m
        logger.info(f"  Quantile q={q:.2f} trained.")

    logger.info(f"LightGBM quantile models trained at {quantiles}")
    return models


# ============================================================
# Neural Network (Keras)
# ============================================================
def build_nn_model(input_dim: int, params: dict):
    """Build a Keras neural network model."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Dense(
        params.get("units_1", 128),
        activation="relu",
        input_shape=(input_dim,),
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(params.get("dropout_1", 0.3)))

    # Hidden layer 2
    model.add(keras.layers.Dense(
        params.get("units_2", 64),
        activation="relu",
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(params.get("dropout_2", 0.2)))

    # Hidden layer 3
    model.add(keras.layers.Dense(
        params.get("units_3", 32),
        activation="relu",
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(params.get("dropout_3", 0.1)))

    # Output layer
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001))
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    random_state: int = 42,
):
    """Train Neural Network with Optuna hyperparameter tuning."""
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler

    tf.random.set_seed(random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    def objective(trial):
        params = {
            "units_1": trial.suggest_categorical("units_1", [64, 128, 256]),
            "units_2": trial.suggest_categorical("units_2", [32, 64, 128]),
            "units_3": trial.suggest_categorical("units_3", [16, 32, 64]),
            "dropout_1": trial.suggest_float("dropout_1", 0.1, 0.5),
            "dropout_2": trial.suggest_float("dropout_2", 0.1, 0.4),
            "dropout_3": trial.suggest_float("dropout_3", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }

        model = build_nn_model(X_train_scaled.shape[1], params)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=params["batch_size"],
            callbacks=[early_stop],
            verbose=0,
        )

        preds = model.predict(X_val_scaled, verbose=0).flatten()
        return _rmse(y_val, preds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"NN best RMSE: {study.best_value:.2f}")
    logger.info(f"NN best params: {study.best_params}")

    # Retrain with best params
    best_params = study.best_params
    best_model = build_nn_model(X_train_scaled.shape[1], best_params)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = best_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=300,
        batch_size=best_params.get("batch_size", 64),
        callbacks=[early_stop],
        verbose=0,
    )

    return best_model, scaler, study, history


# ============================================================
# Save / Load
# ============================================================
def save_model(model, name: str, scaler=None, format_key: str = "mens_odi"):
    """Save a trained model to disk."""
    path = MODELS_DIR / f"{format_key}_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    logger.info(f"Saved {name} to {path}")


def load_model(name: str, format_key: str = "mens_odi"):
    """Load a trained model from disk."""
    path = MODELS_DIR / f"{format_key}_{name}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("scaler")


# ============================================================
# Unified Training Pipeline
# ============================================================
def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    format_key: str = "mens_odi",
    xgb_trials: int = 100,
    rf_trials: int = 50,
    lgb_trials: int = 50,
    nn_trials: int = 30,
):
    """
    Train all 4 ML models with Optuna tuning.
    Returns dict of model_name -> (model, optional_scaler, study)
    """
    results = {}

    # XGBoost
    logger.info("=" * 60)
    logger.info("Training XGBoost...")
    logger.info("=" * 60)
    xgb_model, xgb_study = train_xgboost(X_train, y_train, X_val, y_val, n_trials=xgb_trials)
    save_model(xgb_model, "xgboost", format_key=format_key)
    results["XGBoost"] = {"model": xgb_model, "scaler": None, "study": xgb_study}

    # Random Forest
    logger.info("=" * 60)
    logger.info("Training Random Forest...")
    logger.info("=" * 60)
    rf_model, rf_study = train_random_forest(X_train, y_train, X_val, y_val, n_trials=rf_trials)
    save_model(rf_model, "random_forest", format_key=format_key)
    results["RandomForest"] = {"model": rf_model, "scaler": None, "study": rf_study}

    # LightGBM
    logger.info("=" * 60)
    logger.info("Training LightGBM...")
    logger.info("=" * 60)
    lgb_model, lgb_study = train_lightgbm(X_train, y_train, X_val, y_val, n_trials=lgb_trials)
    save_model(lgb_model, "lightgbm", format_key=format_key)
    results["LightGBM"] = {"model": lgb_model, "scaler": None, "study": lgb_study}

    # Neural Network (optional - can be slow)
    if nn_trials > 0:
        logger.info("=" * 60)
        logger.info("Training Neural Network...")
        logger.info("=" * 60)
        nn_model, nn_scaler, nn_study, nn_history = train_neural_network(
            X_train, y_train, X_val, y_val, n_trials=nn_trials
        )
        save_model(nn_model, "neural_network", scaler=nn_scaler, format_key=format_key)
        results["NeuralNetwork"] = {
            "model": nn_model, "scaler": nn_scaler,
            "study": nn_study, "history": nn_history,
        }

    return results


def predict_with_model(model, X, scaler=None):
    """Make predictions, handling scaling for NN."""
    if scaler is not None:
        X_input = scaler.transform(X)
        preds = model.predict(X_input, verbose=0).flatten()
    else:
        preds = model.predict(X)

    return preds
