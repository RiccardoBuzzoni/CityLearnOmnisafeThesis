import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from lime.lime_tabular import LimeTabularExplainer

from rbc_linear_regressor import (
    LinearRegressor,
    FEATURE_COLS,
    TARGET_COLS,
    DOMAIN_FEATURES,
    TRAIN_STEPS,
    DATA_PATH,
    OUTPUT_DIR,
    main as train_models,
)

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────

# Targets to analyse — all three actions
LIME_TARGETS = TARGET_COLS  # ["cooling_device", "dhw_storage", "electrical_storage"]

# Output directory for LIME plots
LIME_OUTPUT_DIR = "lime_results"
os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)

# Number of samples LIME uses internally to fit its local linear approximation
LIME_N_SAMPLES = 1000

SEED = 42
np.random.seed(SEED)


# ──────────────────────────────────────────────
# 2. PREDICT HELPER
# ──────────────────────────────────────────────

def make_predict_fn(model: torch.nn.Module):
    """
    Wrap a LinearRegressor so that LIME can call it.
    LIME passes a 2-D numpy array of shape (n_samples, n_features)
    and expects a 1-D numpy array of predictions back.

    Parameters
    ----------
    model : LinearRegressor
        Trained PyTorch model.

    Returns
    -------
    predict_fn : callable
        Function that takes a numpy array and returns predictions.
    """
    def predict_fn(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            preds  = model(tensor).squeeze(1).numpy()
        return preds
    return predict_fn


# ──────────────────────────────────────────────
# 3. REPRESENTATIVE TIMESTEP SELECTION
# ──────────────────────────────────────────────

def select_representative_timesteps(
    X_eval_norm: np.ndarray,
    y_eval: np.ndarray,
    model: torch.nn.Module,
) -> dict[str, int]:
    """
    Select 5 representative timesteps from the evaluation set:
      - max_action   : timestep where the RBC assigns the highest action value
      - min_action   : timestep where the RBC assigns the lowest action value
      - mean_action  : timestep whose action is closest to the mean action
      - worst_pred   : timestep where the regressor makes its largest error
      - best_pred    : timestep where the regressor makes its smallest error

    Parameters
    ----------
    X_eval_norm : np.ndarray
        Normalised evaluation features filtered to domain columns, shape (N_eval, n_feats).
    y_eval : np.ndarray
        Ground-truth action values for the eval set, shape (N_eval,).
    model : LinearRegressor
        Trained PyTorch model for this target.

    Returns
    -------
    dict mapping label -> local index within the eval set.
    """
    predict_fn = make_predict_fn(model)
    preds = predict_fn(X_eval_norm)
    errors = np.abs(preds - y_eval)

    mean_val = y_eval.mean()
    mean_idx = int(np.argmin(np.abs(y_eval - mean_val)))

    return {
        "max_action" : int(np.argmax(y_eval)),
        "min_action" : int(np.argmin(y_eval)),
        "mean_action": mean_idx,
        "worst_pred" : int(np.argmax(errors)),
        "best_pred"  : int(np.argmin(errors)),
    }


# ──────────────────────────────────────────────
# 4. LIME ANALYSIS
# ──────────────────────────────────────────────

def run_lime(
    explainer: LimeTabularExplainer,
    predict_fn,
    instance: np.ndarray,
    label: str,
    target_name: str,
    true_val: float,
    pred_val: float,
    save_dir: str,
) -> list:
    """
    Run LIME on a single instance, then plot and save the feature importance bar chart.

    Parameters
    ----------
    explainer : LimeTabularExplainer
        Fitted LIME explainer (trained on the training set).
    predict_fn : callable
        Model prediction function.
    instance : np.ndarray
        Single normalised feature vector, shape (n_domain_feats,).
    label : str
        Descriptive label for the timestep (e.g. 'max_action').
    target_name : str
        Name of the action target.
    true_val : float
        Ground-truth action value at this timestep.
    pred_val : float
        Model predicted action value at this timestep.
    save_dir : str
        Directory where the plot is saved.

    Returns
    -------
    list of (feature_description, weight) tuples sorted by abs weight descending.
    """
    explanation = explainer.explain_instance(
        data_row     = instance,
        predict_fn   = predict_fn,
        num_features = len(DOMAIN_FEATURES[target_name]),
        num_samples  = LIME_N_SAMPLES,
    )

    # Extract (feature_name, weight) pairs sorted by absolute weight descending
    feature_weights = explanation.as_list()
    feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    names  = [fw[0] for fw in feature_weights]
    weights = [fw[1] for fw in feature_weights]
    colors  = ["steelblue" if w >= 0 else "tomato" for w in weights]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names[::-1], weights[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight (contribution to prediction)")
    ax.set_title(
        f"LIME — {target_name} | {label}\n"
        f"True: {true_val:.3f}  |  Predicted: {pred_val:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    filename = os.path.join(save_dir, f"lime_{target_name}_{label}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  LIME plot saved → {filename}")

    return feature_weights


def plot_mean_importance(
    all_weights: dict[str, list],
    target_name: str,
    save_dir: str,
) -> None:
    """
    Compute and plot the mean absolute LIME feature importance across all
    analysed timesteps.

    Parameters
    ----------
    all_weights : dict mapping label -> list of (feature_description, weight) pairs.
    target_name : str
        Name of the action target.
    save_dir : str
        Directory where the plot is saved.
    """
    domain_feats = DOMAIN_FEATURES[target_name]

    # Accumulate absolute weights per feature across all instances
    importance: dict[str, list] = {feat: [] for feat in domain_feats}

    for fw_list in all_weights.values():
        for feat_desc, weight in fw_list:
            # LIME descriptions include threshold conditions (e.g. 'solar_generation > 0.90')
            # so we match by substring against the original feature names
            for feat in domain_feats:
                if feat in feat_desc:
                    importance[feat].append(abs(weight))
                    break

    mean_importance = {feat: float(np.mean(vals)) if vals else 0.0
                       for feat, vals in importance.items()}

    sorted_feats = sorted(mean_importance, key=lambda f: mean_importance[f], reverse=True)
    sorted_vals  = [mean_importance[f] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sorted_feats[::-1], list(sorted_vals[::-1]), color="steelblue")
    ax.set_xlabel("Mean absolute LIME weight")
    ax.set_title(
        f"LIME — Mean Feature Importance across representative timesteps\n{target_name}",
        fontsize=11
    )
    plt.tight_layout()
    filename = os.path.join(save_dir, f"lime_{target_name}_mean_importance.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Mean importance plot saved → {filename}")


# ──────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    # ── Train models (one per target, each on its domain features) ────
    print("Training linear regressors...")
    trained_models, trained_features, mean, std = train_models()

    # ── Reload raw data and rebuild eval split ────────────────────────
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURE_COLS].values
    Y  = df[TARGET_COLS].values

    X_train_raw = X[:TRAIN_STEPS]
    X_eval_raw  = X[TRAIN_STEPS:]
    Y_eval      = Y[TRAIN_STEPS:]

    # Re-apply normalisation using the same train statistics
    X_train_norm_all = (X_train_raw - mean) / std
    X_eval_norm_all  = (X_eval_raw  - mean) / std

    # ── Iterate over all three targets ───────────────────────────────
    for target in LIME_TARGETS:
        print(f"\n{'═'*50}")
        print(f"LIME analysis — {target}")

        target_idx   = TARGET_COLS.index(target)
        y_eval       = Y_eval[:, target_idx]
        model        = trained_models[target]
        domain_feats = trained_features[target]
        col_idx      = [FEATURE_COLS.index(f) for f in domain_feats]

        # Slice domain-relevant columns
        X_train_lime = X_train_norm_all[:, col_idx]
        X_eval_lime  = X_eval_norm_all[:,  col_idx]

        predict_fn = make_predict_fn(model)

        # Build LIME explainer on the domain-filtered training set
        explainer = LimeTabularExplainer(
            training_data = X_train_lime,
            feature_names = domain_feats,
            mode          = "regression",
            random_state  = SEED,
        )

        # Select representative timesteps
        rep_timesteps = select_representative_timesteps(X_eval_lime, y_eval, model)

        print(f"Representative timesteps:")
        for label, local_idx in rep_timesteps.items():
            print(f"  {label:<15} → eval index {local_idx:>3}  "
                  f"| true action: {y_eval[local_idx]:.3f}")

        # Run LIME on each representative timestep to accumulate weights
        all_weights: dict[str, list] = {}

        print(f"Running LIME ({len(rep_timesteps)} timesteps)...")
        for label, local_idx in rep_timesteps.items():
            instance = X_eval_lime[local_idx]
            true_val = y_eval[local_idx]
            pred_val = float(predict_fn(instance.reshape(1, -1))[0])

            explanation = explainer.explain_instance(
                data_row     = instance,
                predict_fn   = predict_fn,
                num_features = len(domain_feats),
                num_samples  = LIME_N_SAMPLES,
            )
            fw = explanation.as_list()
            fw.sort(key=lambda x: abs(x[1]), reverse=True)
            all_weights[label] = fw
            print(f"  {label:<15} | true: {true_val:.3f} | pred: {pred_val:.3f}")

        # Plot mean feature importance for this target
        plot_mean_importance(all_weights, target, LIME_OUTPUT_DIR)

    print("\nLIME analysis complete.")


if __name__ == "__main__":
    main()