# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:37:52 2026

@author: Baize
"""

# -*- coding: utf-8 -*-
"""
Stock Price Prediction using LSTM & GRU

This script implements a research-style experiment for comparing LSTM and GRU
models on mid-to-high-frequency stock price prediction.

Main features:
- 5-minute stock data
- Sliding window sequence modeling
- LSTM vs GRU comparison
- MAE / RMSE / MAPE / R2 evaluation
- Directional accuracy
- Simple trading strategy simulation

Author: AsterZhou
"""

import os
import time
import random
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn import preprocessing
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


# ============================================================
# 1. Global Configuration
# ============================================================

SEED = 314
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Data configuration
TICKER = "601988.XSHG"
CSV_PATH = os.path.join("data", f"{TICKER}_5m_full.csv")

FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# Sliding window configuration
N_STEPS = 144
LOOKUP_STEP = 24

# Dataset split configuration
SCALE = True
SHUFFLE = False
SPLIT_BY_DATE = True
TEST_SIZE = 0.2

# Model configuration
N_LAYERS = 2
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

LOSS = "huber_loss"
OPTIMIZER = "adam"

# Training configuration
BATCH_SIZE = 128
EPOCHS = 15
USE_EARLY_STOPPING = True

# Output folders
RESULTS_DIR = "results"
LOGS_DIR = "logs"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# ============================================================
# 2. Utility Functions
# ============================================================

def create_folders():
    """Create output folders if they do not exist."""
    for folder in [RESULTS_DIR, LOGS_DIR, FIGURES_DIR, "data"]:
        if not os.path.isdir(folder):
            os.makedirs(folder)


def shuffle_in_unison(a, b):
    """Shuffle two arrays in the same way."""
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def load_csv_data(csv_path):
    """Load stock CSV data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Please place your data file under the data/ folder."
        )

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "close" in df.columns and "adjclose" not in df.columns:
        df.rename(columns={"close": "adjclose"}, inplace=True)

    required_columns = set(FEATURE_COLUMNS)
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.sort_index()
    return df


# ============================================================
# 3. Data Processing
# ============================================================

def load_data(
    df,
    n_steps=144,
    scale=True,
    shuffle=False,
    lookup_step=24,
    split_by_date=True,
    test_size=0.2,
    feature_columns=None,
):
    """
    Convert raw time-series data into supervised learning samples.

    Input:
        Historical stock dataframe.

    Output:
        Dictionary containing train/test data, scalers, and test dataframe.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    if not split_by_date and not shuffle:
        print("Warning: Current configuration may cause information leakage.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    result = {}
    df = df.copy()
    result["df"] = df.copy()

    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"'{col}' does not exist in the dataframe.")

    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}

        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1)
            )
            column_scaler[column] = scaler

        result["column_scaler"] = column_scaler

    df["future"] = df["adjclose"].shift(-lookup_step)

    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(
        df[feature_columns + ["date"]].values,
        df["future"].values,
    ):
        sequences.append(entry)

        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    result["last_sequence"] = last_sequence

    X, y = [], []

    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        train_samples = int((1 - test_size) * len(X))

        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]

        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]

        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])

    else:
        from sklearn.model_selection import train_test_split

        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
        )

    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep="first")]

    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


# ============================================================
# 4. Model Construction
# ============================================================

def create_recurrent_model(
    sequence_length,
    n_features,
    cell_type="LSTM",
    units=256,
    n_layers=2,
    dropout=0.4,
    loss="huber_loss",
    optimizer="adam",
    bidirectional=False,
):
    """
    Create an LSTM or GRU model.
    """
    if cell_type.upper() == "LSTM":
        cell = LSTM
    elif cell_type.upper() == "GRU":
        cell = GRU
    else:
        raise ValueError("cell_type must be either 'LSTM' or 'GRU'.")

    model = Sequential()

    for i in range(n_layers):
        return_sequences = i < n_layers - 1

        if i == 0:
            if bidirectional:
                model.add(
                    Bidirectional(
                        cell(units, return_sequences=return_sequences),
                        batch_input_shape=(None, sequence_length, n_features),
                    )
                )
            else:
                model.add(
                    cell(
                        units,
                        return_sequences=return_sequences,
                        batch_input_shape=(None, sequence_length, n_features),
                    )
                )
        else:
            if bidirectional:
                model.add(
                    Bidirectional(
                        cell(units, return_sequences=return_sequences)
                    )
                )
            else:
                model.add(
                    cell(units, return_sequences=return_sequences)
                )

        model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))

    model.compile(
        loss=loss,
        metrics=["mean_absolute_error"],
        optimizer=optimizer,
    )

    return model


# ============================================================
# 5. Training
# ============================================================

def train_model(model, data, model_name):
    """
    Train a model and save the best weights.
    """
    model_path = os.path.join(RESULTS_DIR, model_name + ".h5")

    callbacks = [
        ModelCheckpoint(
            model_path,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(LOGS_DIR, model_name)),
    ]

    if USE_EARLY_STOPPING:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data["X_test"], data["y_test"]),
        callbacks=callbacks,
        verbose=1,
    )

    print(f"Model saved to: {model_path}")

    return history, model_path


# ============================================================
# 6. Evaluation
# ============================================================

def get_final_df(model, data, lookup_step=24):
    """
    Generate final dataframe with predictions and trading simulation results.
    """
    buy_profit = lambda current, pred_future, true_future: (
        true_future - current if pred_future > current else 0
    )

    sell_profit = lambda current, pred_future, true_future: (
        current - true_future if pred_future < current else 0
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    y_pred = model.predict(X_test)

    if SCALE:
        y_test = data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(y_test, axis=0)
        ).flatten()

        y_pred = data["column_scaler"]["adjclose"].inverse_transform(
            y_pred
        ).flatten()

    df = data["test_df"].copy()

    min_len = min(len(df), len(y_test), len(y_pred))
    df = df.iloc[:min_len]
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]

    df[f"predicted_adjclose_{lookup_step}"] = y_pred
    df[f"true_adjclose_{lookup_step}"] = y_test

    df["buy_profit"] = list(
        map(
            buy_profit,
            df["adjclose"],
            df[f"predicted_adjclose_{lookup_step}"],
            df[f"true_adjclose_{lookup_step}"],
        )
    )

    df["sell_profit"] = list(
        map(
            sell_profit,
            df["adjclose"],
            df[f"predicted_adjclose_{lookup_step}"],
            df[f"true_adjclose_{lookup_step}"],
        )
    )

    df["predicted_direction"] = np.where(
        df[f"predicted_adjclose_{lookup_step}"] > df["adjclose"],
        1,
        -1,
    )

    df["true_direction"] = np.where(
        df[f"true_adjclose_{lookup_step}"] > df["adjclose"],
        1,
        -1,
    )

    return df


def evaluate_model(df, model_label, ticker):
    """
    Evaluate prediction accuracy and trading performance.
    """
    y_true = df[f"true_adjclose_{LOOKUP_STEP}"]
    y_pred = df[f"predicted_adjclose_{LOOKUP_STEP}"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    directional_accuracy = (
        df["predicted_direction"] == df["true_direction"]
    ).mean()

    total_profit = df["buy_profit"].sum() + df["sell_profit"].sum()
    profit_per_trade = total_profit / len(df)

    print("\n" + "=" * 60)
    print(f"{model_label} Evaluation on {ticker}")
    print("=" * 60)
    print(f"MAE                  : {mae:.6f}")
    print(f"RMSE                 : {rmse:.6f}")
    print(f"MAPE                 : {mape:.6f}")
    print(f"R2                   : {r2:.6f}")
    print(f"Directional Accuracy : {directional_accuracy:.6f}")
    print(f"Total Profit         : {total_profit:.6f}")
    print(f"Profit per Trade     : {profit_per_trade:.6f}")

    return {
        "model": model_label,
        "ticker": ticker,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "directional_accuracy": directional_accuracy,
        "total_profit": total_profit,
        "profit_per_trade": profit_per_trade,
    }


# ============================================================
# 7. Visualization
# ============================================================

def plot_prediction(df_lstm, df_gru, ticker):
    """
    Plot true price vs LSTM and GRU predictions.
    """
    plt.figure(figsize=(12, 5))

    plt.plot(
        df_lstm[f"true_adjclose_{LOOKUP_STEP}"].values,
        label="True Price",
    )

    plt.plot(
        df_lstm[f"predicted_adjclose_{LOOKUP_STEP}"].values,
        linestyle="--",
        label="LSTM Prediction",
    )

    plt.plot(
        df_gru[f"predicted_adjclose_{LOOKUP_STEP}"].values,
        linestyle="-.",
        label="GRU Prediction",
    )

    plt.title(f"LSTM vs GRU Stock Price Prediction ({ticker})")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"{ticker}_lstm_gru_prediction.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Prediction plot saved to: {save_path}")


def plot_training_history(history_lstm, history_gru, ticker):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(12, 5))

    plt.plot(history_lstm.history["loss"], label="LSTM Train Loss")
    plt.plot(history_lstm.history["val_loss"], label="LSTM Val Loss")

    plt.plot(history_gru.history["loss"], label="GRU Train Loss")
    plt.plot(history_gru.history["val_loss"], label="GRU Val Loss")

    plt.title(f"LSTM vs GRU Training Loss ({ticker})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"{ticker}_training_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Training loss plot saved to: {save_path}")


# ============================================================
# 8. Hyperparameter Search
# ============================================================

def run_grid_search(data):
    """
    Optional grid search for LSTM model.
    """
    from itertools import product

    param_grid = {
        "units": [64, 128],
        "dropout": [0.2, 0.3],
        "optimizer": ["adam", "rmsprop"],
    }

    param_combinations = list(
        product(
            param_grid["units"],
            param_grid["dropout"],
            param_grid["optimizer"],
        )
    )

    results = []

    for units, dropout, optimizer in param_combinations:
        print(
            f"Testing parameters: units={units}, "
            f"dropout={dropout}, optimizer={optimizer}"
        )

        model = create_recurrent_model(
            sequence_length=N_STEPS,
            n_features=len(FEATURE_COLUMNS),
            cell_type="LSTM",
            units=units,
            n_layers=N_LAYERS,
            dropout=dropout,
            loss=LOSS,
            optimizer=optimizer,
            bidirectional=BIDIRECTIONAL,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )

        history = model.fit(
            data["X_train"],
            data["y_train"],
            validation_data=(data["X_test"], data["y_test"]),
            epochs=16,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[early_stop],
        )

        results.append(
            {
                "units": units,
                "dropout": dropout,
                "optimizer": optimizer,
                "best_val_loss": min(history.history["val_loss"]),
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="best_val_loss")
    print("\nGrid Search Results:")
    print(results_df)

    results_path = os.path.join(RESULTS_DIR, "grid_search_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"Grid search results saved to: {results_path}")

    return results_df


# ============================================================
# 9. Main Experiment
# ============================================================

def main():
    create_folders()

    date_now = time.strftime("%Y-%m-%d")

    print("=" * 60)
    print("Stock Price Prediction using LSTM & GRU")
    print("=" * 60)
    print(f"Ticker       : {TICKER}")
    print(f"CSV Path     : {CSV_PATH}")
    print(f"N_STEPS      : {N_STEPS}")
    print(f"LOOKUP_STEP  : {LOOKUP_STEP}")
    print(f"Features     : {FEATURE_COLUMNS}")
    print("=" * 60)

    df = load_csv_data(CSV_PATH)

    data = load_data(
        df,
        n_steps=N_STEPS,
        scale=SCALE,
        shuffle=SHUFFLE,
        lookup_step=LOOKUP_STEP,
        split_by_date=SPLIT_BY_DATE,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
    )

    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape : {data['X_test'].shape}")

    base_model_name = (
        f"{date_now}_{TICKER}"
        f"-sh-{int(SHUFFLE)}"
        f"-sc-{int(SCALE)}"
        f"-sbd-{int(SPLIT_BY_DATE)}"
        f"-{LOSS}"
        f"-{OPTIMIZER}"
        f"-seq-{N_STEPS}"
        f"-step-{LOOKUP_STEP}"
        f"-layers-{N_LAYERS}"
        f"-units-{UNITS}"
    )

    if BIDIRECTIONAL:
        base_model_name += "-bidirectional"

    # ---------------- LSTM ----------------
    model_lstm = create_recurrent_model(
        sequence_length=N_STEPS,
        n_features=len(FEATURE_COLUMNS),
        cell_type="LSTM",
        units=UNITS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        loss=LOSS,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )

    model_name_lstm = "LSTM_" + base_model_name

    history_lstm, model_path_lstm = train_model(
        model_lstm,
        data,
        model_name_lstm,
    )

    # ---------------- GRU ----------------
    model_gru = create_recurrent_model(
        sequence_length=N_STEPS,
        n_features=len(FEATURE_COLUMNS),
        cell_type="GRU",
        units=UNITS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        loss=LOSS,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )

    model_name_gru = "GRU_" + base_model_name

    history_gru, model_path_gru = train_model(
        model_gru,
        data,
        model_name_gru,
    )

    # Load best weights
    model_lstm.load_weights(model_path_lstm)
    model_gru.load_weights(model_path_gru)

    # Evaluation
    df_lstm = get_final_df(model_lstm, data, lookup_step=LOOKUP_STEP)
    df_gru = get_final_df(model_gru, data, lookup_step=LOOKUP_STEP)

    metrics_lstm = evaluate_model(df_lstm, "LSTM", TICKER)
    metrics_gru = evaluate_model(df_gru, "GRU", TICKER)

    metrics_df = pd.DataFrame([metrics_lstm, metrics_gru])
    metrics_path = os.path.join(RESULTS_DIR, f"{TICKER}_evaluation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nEvaluation metrics saved to: {metrics_path}")

    # Save prediction results
    lstm_prediction_path = os.path.join(RESULTS_DIR, f"{TICKER}_lstm_predictions.csv")
    gru_prediction_path = os.path.join(RESULTS_DIR, f"{TICKER}_gru_predictions.csv")

    df_lstm.to_csv(lstm_prediction_path)
    df_gru.to_csv(gru_prediction_path)

    print(f"LSTM prediction results saved to: {lstm_prediction_path}")
    print(f"GRU prediction results saved to: {gru_prediction_path}")

    # Visualization
    plot_prediction(df_lstm, df_gru, TICKER)
    plot_training_history(history_lstm, history_gru, TICKER)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()