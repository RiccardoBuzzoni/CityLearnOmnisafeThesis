import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────

DATA_PATH   = "results/advanced_rbc_results.csv"
OUTPUT_DIR  = "linear_regressor_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns (observations) and target columns (actions)
FEATURE_COLS = [
    "hour", "indoor_temp", "cooling_setpoint", "outdoor_temp",
    "outdoor_temp_predicted", "cooling_demand", "elec_price",
    "carbon_intensity", "solar_generation", "occupant_count",
    "electrical_storage_soc", "dhw_storage_soc", "dhw_demand"
]

TARGET_COLS = ["cooling_device", "dhw_storage", "electrical_storage"]

# Temporal split: first 504 timesteps (~3 weeks) for training,
# remaining ~215 timesteps (~1 week) for evaluation
TRAIN_STEPS = 504

EPOCHS      = 200
LR          = 1e-3
BATCH_SIZE  = 32
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────

class RBCDataset(Dataset):
    """
    PyTorch Dataset for RBC observations and a single action target.

    Parameters
    ----------
    X : np.ndarray
        Normalised feature matrix of shape (N, 13).
    y : np.ndarray
        Target action values of shape (N,).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────
# 3. MODEL
# ──────────────────────────────────────────────

class LinearRegressor(nn.Module):
    """
    Single-output linear regressor implemented as a one-layer neural network.
    Using a linear model keeps the explainability analysis with LIME straightforward.

    Parameters
    ----------
    input_dim : int
        Number of input features (13 in our case).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ──────────────────────────────────────────────
# 4. TRAINING & EVALUATION FUNCTIONS
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    mse : float
        Average MSE loss over the epoch.
    mae : float
        Average MAE over the epoch.
    """
    model.train()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds      = model(X_batch)
        loss       = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        batch_size  = X_batch.size(0)
        total_mse  += loss.item() * batch_size
        total_mae  += torch.mean(torch.abs(preds.detach() - y_batch)).item() * batch_size
        n_samples  += batch_size
    return total_mse / n_samples, total_mae / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module
) -> tuple[float, float]:
    """
    Evaluate the model on a DataLoader.

    Returns
    -------
    mse : float
        Average MSE loss over the dataset.
    mae : float
        Average MAE (mean absolute error) over the dataset.
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds      = model(X_batch)
            mse        = criterion(preds, y_batch)
            mae        = torch.mean(torch.abs(preds - y_batch))
            batch_size = X_batch.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            n_samples += batch_size
    return total_mse / n_samples, total_mae / n_samples


# ──────────────────────────────────────────────
# 5. PLOTTING
# ──────────────────────────────────────────────

def plot_metrics(
    train_losses: list,
    eval_losses: list,
    train_maes: list,
    eval_maes: list,
    target_name: str,
    save_dir: str
):
    """
    Plot and save two side-by-side subplots (MAE and MSE Loss) for a single target,
    mirroring the 'model accuracy / model loss' layout.

    Parameters
    ----------
    train_losses : list of float
        MSE loss recorded at each epoch during training.
    eval_losses : list of float
        MSE loss recorded at each epoch during evaluation.
    train_maes : list of float
        MAE recorded at each epoch during training.
    eval_maes : list of float
        MAE recorded at each epoch during evaluation.
    target_name : str
        Name of the action target (used in title and filename).
    save_dir : str
        Directory where the plot image is saved.
    """
    epochs_range = range(1, len(train_losses) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot: model accuracy (MAE)
    ax_acc.plot(epochs_range, train_maes, label="train", color="steelblue")
    ax_acc.plot(epochs_range, eval_maes,  label="test",  color="tomato")
    ax_acc.set_title("model accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (MAE)")
    ax_acc.legend()

    # Right subplot: model loss (MSE)
    ax_loss.plot(epochs_range, train_losses, label="train", color="steelblue")
    ax_loss.plot(epochs_range, eval_losses,  label="test",  color="tomato")
    ax_loss.set_title("model loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (MSE)")
    ax_loss.legend()

    fig.suptitle(f"Linear Regressor — {target_name}", fontsize=13)
    plt.tight_layout()
    filename = os.path.join(save_dir, f"metrics_{target_name}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Plot saved → {filename}")


# ──────────────────────────────────────────────
# 6. MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    # ── Load dataset ──────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} timesteps, {len(FEATURE_COLS)} features, {len(TARGET_COLS)} targets")

    X = df[FEATURE_COLS].values  # (719, 13)
    Y = df[TARGET_COLS].values   # (719, 3)

    # ── Temporal split ────────────────────────
    X_train, X_eval = X[:TRAIN_STEPS], X[TRAIN_STEPS:]
    Y_train, Y_eval = Y[:TRAIN_STEPS], Y[TRAIN_STEPS:]
    print(f"Train: {len(X_train)} steps | Eval: {len(X_eval)} steps")

    # ── Feature normalisation (fit on train only) ──
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant features

    X_train_norm = (X_train - mean) / std
    X_eval_norm  = (X_eval  - mean) / std

    # Store one model per target so LIME can query each independently later
    trained_models = {}

    # ── Train one model per target ─────────────
    for idx, target in enumerate(TARGET_COLS):
        print(f"\n{'─'*50}")
        print(f"Target: {target}")

        y_train = Y_train[:, idx]
        y_eval  = Y_eval[:, idx]

        train_dataset = RBCDataset(X_train_norm, y_train)
        eval_dataset  = RBCDataset(X_eval_norm,  y_eval)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        model     = LinearRegressor(input_dim=len(FEATURE_COLS))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train_losses, eval_losses = [], []
        train_maes,   eval_maes   = [], []

        for epoch in range(1, EPOCHS + 1):
            t_loss, t_mae = train_one_epoch(model, train_loader, criterion, optimizer)
            e_loss, e_mae = evaluate(model, eval_loader, criterion)
            train_losses.append(t_loss)
            eval_losses.append(e_loss)
            train_maes.append(t_mae)
            eval_maes.append(e_mae)

            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{EPOCHS} | Train MSE: {t_loss:.6f} MAE: {t_mae:.6f} | Eval MSE: {e_loss:.6f} MAE: {e_mae:.6f}")

        plot_metrics(train_losses, eval_losses, train_maes, eval_maes, target, OUTPUT_DIR)
        trained_models[target] = model

    print(f"\n{'─'*50}")
    print("Training complete.")

    # ── Save normalisation stats (useful for LIME step) ──
    np.save(os.path.join(OUTPUT_DIR, "feature_mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "feature_std.npy"),  std)

    return trained_models, mean, std


if __name__ == "__main__":
    main()