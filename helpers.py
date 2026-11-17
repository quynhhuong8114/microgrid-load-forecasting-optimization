"""Helper functions for all notebooks - microgrid load forecasting (v2 - BDG2)."""
import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def inverse_transform_target(y_scaled, scaler, target_idx, n_features):
    """Inverse transform target variable from scaled to real values."""
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, target_idx] = np.asarray(y_scaled).flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE (daily avg), R2 on real kW values."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    # MAPE on daily average
    n_days = len(y_true) // 24
    if n_days > 0:
        yt_daily = np.array([y_true[i * 24:(i + 1) * 24].mean() for i in range(n_days)])
        yp_daily = np.array([y_pred[i * 24:(i + 1) * 24].mean() for i in range(n_days)])
        mape = float(np.mean(np.abs((yt_daily - yp_daily) / yt_daily)) * 100)
    else:
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100)

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def plot_training_history(history, model_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name.lower().replace("-", "_")}_training_loss.png'), dpi=150)
    plt.show()


def plot_predictions(y_true, y_pred, model_name, n_points=168):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(n_points), y_true[:n_points], 'b-', label='Actual', linewidth=1.5)
    ax.plot(range(n_points), y_pred[:n_points], 'r--', label='Forecast', linewidth=1.5, alpha=0.8)
    metrics = calculate_metrics(y_true, y_pred)
    ax.set_title(f'{model_name} - MAPE={metrics["MAPE"]:.2f}%, R2={metrics["R2"]:.4f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Load (kW)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                f'{model_name.lower().replace("-", "_")}_predictions.png'), dpi=150)
    plt.show()
