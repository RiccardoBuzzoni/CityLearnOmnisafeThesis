# Imports
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from utils import Config, default_env_config, select_env_config

# Import of AdvancedRBC agent
from main_rbc import AdvancedRBC

# Import of LinearRegressor and domain feature configuration
from rbc_linear_regressor import (
    LinearRegressor,
    FEATURE_COLS,
    TARGET_COLS,
    DOMAIN_FEATURES,
    main as train_linear_models,
)


# ──────────────────────────────────────────────
# LINEAR REGRESSOR AGENT
# ──────────────────────────────────────────────

class LinearRegressorAgent:
    """
    CityLearn-compatible agent that uses the three trained LinearRegressor
    models to predict actions from observations.

    Each model was trained only on the domain-relevant features for its
    target action, as defined in DOMAIN_FEATURES. Observations are
    normalised using the training-set statistics (mean, std) before
    being passed to the models.

    Parameters
    ----------
    env : CityLearnEnv
        CityLearn environment (used to read observation/action names).
    trained_models : dict[str, LinearRegressor]
        Trained PyTorch models keyed by target name.
    trained_features : dict[str, list[str]]
        Domain feature lists keyed by target name.
    mean : np.ndarray
        Per-feature training mean, shape (len(FEATURE_COLS),).
    std : np.ndarray
        Per-feature training std, shape (len(FEATURE_COLS),).
    """

    def __init__(
        self,
        env: CityLearnEnv,
        trained_models: dict,
        trained_features: dict,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.env             = env
        self.trained_models  = trained_models
        self.trained_features = trained_features
        self.mean            = mean
        self.std             = std

        # Pre-compute column indices for each target to avoid repeated lookups
        self._col_idx: dict[str, list[int]] = {
            target: [FEATURE_COLS.index(f) for f in feats]
            for target, feats in trained_features.items()
        }

    def predict(self, observations: list) -> list:
        """
        Predict actions from raw observations.

        Parameters
        ----------
        observations : list of list of float
            Raw observation vectors from the environment, one per building.

        Returns
        -------
        list of list of float
            Predicted action vectors, one per building.
        """
        actions = []
        for i, obs_vec in enumerate(observations):
            obs_names = self.env.observation_names[i]
            act_names = self.env.action_names[i]

            # Build the full feature vector in FEATURE_COLS order
            full_obs = np.array([
                obs_vec[obs_names.index(col)] if col in obs_names else 0.0
                for col in FEATURE_COLS
            ], dtype=np.float32)

            # Normalise using training statistics
            full_obs_norm = (full_obs - self.mean) / self.std

            # Predict each action with its dedicated model
            action = [0.0] * len(act_names)
            for target in TARGET_COLS:
                if target not in act_names:
                    continue
                model    = self.trained_models[target]
                col_idx  = self._col_idx[target]
                x        = torch.tensor(
                    full_obs_norm[col_idx], dtype=torch.float32
                ).unsqueeze(0)   # shape (1, n_feats)

                model.eval()
                with torch.no_grad():
                    pred = model(x).item()

                action[act_names.index(target)] = pred

            # Clip actions to their physical bounds:
            # cooling_device ∈ [0, 1]  — cannot be negative (no reverse cooling)
            # dhw_storage    ∈ [-1, 1]
            # electrical_storage ∈ [-1, 1]
            CLIP_BOUNDS: dict[str, tuple[float, float]] = {
                "cooling_device":      (0.0,  1.0),
                "dhw_storage":         (-1.0, 1.0),
                "electrical_storage":  (-1.0, 1.0),
            }
            for target, (lo, hi) in CLIP_BOUNDS.items():
                if target in act_names:
                    idx = act_names.index(target)
                    action[idx] = float(np.clip(action[idx], lo, hi))

            actions.append(action)

        return actions


def plot_cooling_device(res_rbc, res_reg=None, save_path='cooling_device.png'):
    '''
    Plot the cooling_device action value, indoor temperature, setpoint and comfort band
    for both AdvancedRBC and the linear regressor (if provided).

    Left axis  : temperature (°C) and comfort band
    Right axis : cooling_device action value [0, 1]

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        res_reg: Optional dictionary containing the results from the LinearRegressorAgent.
        save_path: Path where the plot will be saved.
    '''
    time_steps   = range(res_rbc['env_h']['time_steps'])
    temp_data    = res_rbc['env_h']['temperature']
    indoor_temp  = np.array(temp_data['indoor_dry_bulb_temperature'])
    set_point    = np.array(temp_data['indoor_dry_bulb_temperature_set_point'])
    comfort_band = np.array(temp_data['comfort_band'])
    action_rbc   = np.array(res_rbc['env_h']['actions']['cooling_device'])

    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.set_title('Cooling Device — Action Value and Indoor Temperature')

    # Comfort band shading
    ax1.fill_between(time_steps, set_point + comfort_band, set_point - comfort_band,
                     color='green', alpha=0.2, label='Comfort band')

    # Temperature curves
    ax1.plot(time_steps, indoor_temp,
             label='Indoor Temp (RBC)', linewidth=1.5, color='tab:blue')
    ax1.plot(time_steps, set_point,
             label='Setpoint', linestyle='--', linewidth=0.8, color='gray')

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    # Action axis
    ax2 = ax1.twinx()
    ax2.plot(time_steps, action_rbc,
             label='cooling_device (RBC)', linewidth=1.0, color='tab:red')

    if res_reg is not None:
        indoor_temp_reg = np.array(res_reg['env_h']['temperature']['indoor_dry_bulb_temperature'])
        action_reg      = np.array(res_reg['env_h']['actions']['cooling_device'])

        ax1.plot(time_steps, indoor_temp_reg,
                 label='Indoor Temp (Regressor)', linewidth=1.5,
                 color='tab:blue', linestyle='--', alpha=0.7)
        ax2.plot(time_steps, action_reg,
                 label='cooling_device (Regressor)', linewidth=1.0,
                 color='tab:red', linestyle='--', alpha=0.7)

    ax2.set_ylabel('cooling_device action [0, 1]')
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")


def plot_dhw_storage(res_rbc, res_reg=None, save_path='dhw_storage.png'):
    '''
    Plot the dhw_storage action value and DHW storage SoC for both agents.

    Left axis  : dhw_storage SoC [0, 1]
    Right axis : dhw_storage action value [-1, 1]

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        res_reg: Optional dictionary containing the results from the LinearRegressorAgent.
        save_path: Path where the plot will be saved.
    '''
    time_steps  = range(res_rbc['env_h']['time_steps'])
    soc_rbc     = np.array(res_rbc['env_h']['dhw']['dhw_storage_soc'])
    action_rbc  = np.array(res_rbc['env_h']['actions']['dhw_storage'])

    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.set_title('DHW Storage — Action Value and State of Charge')

    ax1.plot(time_steps, soc_rbc,
             label='DHW SoC (RBC)', linewidth=1.5, color='tab:blue')
    ax1.set_ylabel('State of Charge (kWh/kWh_capacity)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(time_steps, action_rbc,
             label='dhw_storage action (RBC)', linewidth=1.0, color='tab:red')

    if res_reg is not None:
        soc_reg    = np.array(res_reg['env_h']['dhw']['dhw_storage_soc'])
        action_reg = np.array(res_reg['env_h']['actions']['dhw_storage'])

        ax1.plot(time_steps, soc_reg,
                 label='DHW SoC (Regressor)', linewidth=1.5,
                 color='tab:blue', linestyle='--', alpha=0.7)
        ax2.plot(time_steps, action_reg,
                 label='dhw_storage action (Regressor)', linewidth=1.0,
                 color='tab:red', linestyle='--', alpha=0.7)

    ax2.set_ylabel('dhw_storage action [-1, 1]')
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='gray', linewidth=0.6, linestyle=':')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")


def plot_electrical_storage(res_rbc, res_reg=None, save_path='electrical_storage.png'):
    '''
    Plot the electrical_storage action value and electrical storage SoC for both agents.

    Left axis  : electrical_storage SoC [0, 1]
    Right axis : electrical_storage action value [-1, 1]

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        res_reg: Optional dictionary containing the results from the LinearRegressorAgent.
        save_path: Path where the plot will be saved.
    '''
    time_steps  = range(res_rbc['env_h']['time_steps'])
    soc_rbc     = np.array(res_rbc['env_h']['electrical_storage_soc'])
    action_rbc  = np.array(res_rbc['env_h']['actions']['electrical_storage'])

    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax1.set_title('Electrical Storage — Action Value and State of Charge')

    ax1.plot(time_steps, soc_rbc,
             label='Electrical SoC (RBC)', linewidth=1.5, color='tab:orange')
    ax1.set_ylabel('State of Charge (kWh/kWh_capacity)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(time_steps, action_rbc,
             label='electrical_storage action (RBC)', linewidth=1.0, color='tab:red')

    if res_reg is not None:
        soc_reg    = np.array(res_reg['env_h']['electrical_storage_soc'])
        action_reg = np.array(res_reg['env_h']['actions']['electrical_storage'])

        ax1.plot(time_steps, soc_reg,
                 label='Electrical SoC (Regressor)', linewidth=1.5,
                 color='tab:orange', linestyle='--', alpha=0.7)
        ax2.plot(time_steps, action_reg,
                 label='electrical_storage action (Regressor)', linewidth=1.0,
                 color='tab:red', linestyle='--', alpha=0.7)

    ax2.set_ylabel('electrical_storage action [-1, 1]')
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='gray', linewidth=0.6, linestyle=':')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")



def evaluate_agent(agent_class, env):
    """
    Simulates episode and records results for the given agent in the provided environment.
    Accepts either an agent class (instantiated with just env) or an already-instantiated
    agent object (used for LinearRegressorAgent which needs extra constructor arguments).
    """
    if isinstance(agent_class, type):
        agent = agent_class(env)
    else:
        agent = agent_class

    obs, _ = env.reset()

    # Track action values at each timestep for plotting
    act_names = env.action_names[0]
    action_history: dict[str, list] = {name: [] for name in act_names}

    done = False
    while not done:
        actions = agent.predict(obs)
        # Record actions before stepping
        for j, name in enumerate(act_names):
            action_history[name].append(actions[0][j])
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

    b = env.buildings[0]

    results = {
        'env_h': {
            'time_steps': env.time_steps - 1,

            'actions': action_history,

            'temperature': {
                'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature[:-1],
                'indoor_dry_bulb_temperature_set_point': b.indoor_dry_bulb_temperature_cooling_set_point[:-1],
                'comfort_band': b.comfort_band,
                'cooling_demand': b.cooling_demand[:-1]
            },

            'electrical_storage_soc': b.electrical_storage.soc[:-1],

            'dhw': {
                'dhw_storage_soc': b.dhw_storage.soc[:-1],
                'dhw_demand': b.dhw_demand[:-1]
            },

            'grid_and_solar': {
                'elec_price': b.pricing.electricity_pricing[:-1],
                'solar_generation': b.solar_generation[:-1]
            }
        }
    }

    cb = b.comfort_band
    if not hasattr(cb, '__len__'):
        results['env_h']['temperature']['comfort_band'] = [cb] * results['env_h']['time_steps']
    elif len(cb) > results['env_h']['time_steps']:
        results['env_h']['temperature']['comfort_band'] = cb[:-1]

    return results


def collect_data(agent_class, env):
    agent = agent_class(env)
    obs, _ = env.reset()
    done = False
    rows = []

    obs_names = env.observation_names[0]     
    act_names = env.action_names[0]           

    while not done:
        actions = agent.predict(obs)
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        def get_obs(name):
            return obs[0][obs_names.index(name)] if name in obs_names else None

        def get_act(name):
            return actions[0][act_names.index(name)] if name in act_names else None

        row = {
            # Observations
            "hour":                   get_obs("hour"),
            "indoor_temp":            get_obs("indoor_dry_bulb_temperature"),
            "cooling_setpoint":       get_obs("indoor_dry_bulb_temperature_cooling_set_point"),
            "outdoor_temp":           get_obs("outdoor_dry_bulb_temperature"),
            "outdoor_temp_predicted": get_obs("outdoor_dry_bulb_temperature_predicted_1"),
            "cooling_demand":         get_obs("cooling_demand"),
            "elec_price":             get_obs("electricity_pricing"),
            "carbon_intensity":       get_obs("carbon_intensity"),
            "solar_generation":       get_obs("solar_generation"),
            "occupant_count":         get_obs("occupant_count"),
            "electrical_storage_soc": get_obs("electrical_storage_soc"),
            "dhw_storage_soc":        get_obs("dhw_storage_soc"),
            "dhw_demand":             get_obs("dhw_demand"),
            # Actions
            "cooling_device":  get_act("cooling_device"),
            "dhw_storage":      get_act("dhw_storage"),
            "electrical_storage":  get_act("electrical_storage"),
            # Reward
            "reward": rewards[0],
        }

        rows.append(row)
        obs = next_obs
        done = terminated or truncated

    df = pd.DataFrame(rows)
    return df

def main():
    # Same configuration as main_rbc.py for consistency
    conf = Config()
    args = conf.args
    
    if args.data is None:
        args.data = 'citylearn_challenge_2023_phase_1'

    # Load CityLearn schema
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data) # type: ignore

    print(f"Start evaluation with dataset: {args.data}")

    # ── Train linear regressors ───────────────────────────────────────
    # Each model is trained on the domain-relevant features for its target.
    # Two separate environments are needed because each simulation resets
    # the environment state — running both agents on the same env instance
    # would produce inconsistent results.
    print("Training linear regressors...")
    trained_models, trained_features, mean, std = train_linear_models()

    # ── RBC evaluation ────────────────────────────────────────────────
    env_rbc = CityLearnEnv(schema=schema, central_agent=True)
    print("Running AdvancedRBC...")
    results_rbc = evaluate_agent(AdvancedRBC, env_rbc)

    # ── Linear Regressor evaluation ───────────────────────────────────
    env_reg = CityLearnEnv(schema=schema, central_agent=True)
    reg_agent = LinearRegressorAgent(
        env          = env_reg,
        trained_models  = trained_models,
        trained_features = trained_features,
        mean         = mean,
        std          = std,
    )
    print("Running LinearRegressorAgent...")
    results_reg = evaluate_agent(reg_agent, env_reg)

    # ── Plotting — overlaid curves ────────────────────────────────────
    plot_cooling_device(
        results_rbc, results_reg,
        save_path='evaluation_plots/cooling_device.png'
    )
    plot_dhw_storage(
        results_rbc, results_reg,
        save_path='evaluation_plots/dhw_storage.png'
    )
    plot_electrical_storage(
        results_rbc, results_reg,
        save_path='evaluation_plots/electrical_storage.png'
    )

    # ── CSV export (RBC only, as before) ──────────────────────────────
    df = collect_data(AdvancedRBC, env_rbc)
    df.to_csv('results/advanced_rbc_results.csv', index=False)
    print(df.shape)
    print("CSV file was successfully saved")

if __name__ == '__main__':
    main()