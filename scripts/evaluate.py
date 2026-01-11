# Imports
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from utils import Config, default_env_config, select_env_config

# Import of AdvancedRBC agent
from main_rbc import AdvancedRBC

def plot_temperature(res_rbc, save_path='temperature_rbc.png'):
    '''
    Function to plot indoor temperature, setpoint, comfort band and cooling demand over time.

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        save_path: Path where the plot will be saved.
    '''

    # Data extraction
    temp_data = res_rbc['env_h']['temperature']
    time_steps = range(res_rbc['env_h']['time_steps'])
    
    indoor_temp = temp_data['indoor_dry_bulb_temperature']
    set_point = np.array(temp_data['indoor_dry_bulb_temperature_set_point'])
    comfort_band = np.array(temp_data['comfort_band'])
    cooling_demand = np.array(temp_data['cooling_demand'])
    
    # Plot creation
    fig, ax1 = plt.subplots(figsize=(20, 6))
    
    # Title
    ax1.set_title('Temperature Management with Cooling Demand')
    
    # comfort_band shading
    upper_band = set_point + comfort_band
    lower_band = set_point - comfort_band
    
    ax1.fill_between(
        time_steps,
        upper_band,
        lower_band,
        color='green',
        alpha=0.2,
        label='Comfort band'
    )
    
    # Internal temperature line
    ax1.plot(time_steps, indoor_temp, label='Indoor Temperature', linewidth=1.5, color='tab:blue')
    
    # Setpoint line
    ax1.plot(time_steps, set_point, label='Setpoint', linestyle='--', linewidth=0.8, color='gray')

    # Labels for termperature
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    # Cooling demand line
    ax2.plot(time_steps, cooling_demand, label='Cooling Demand', linewidth=1.0, color='tab:red')

    # Labels for cooling demand
    ax2.set_ylabel('Cooling Demand (kWh)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")
    #plt.show()

def plot_electrical_storage_soc(res_rbc, save_path='electrical_storage_soc.png'):
    '''
    Function to plot the State of Charge (SoC) of electrical storage over time.

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        save_path: Path where the plot will be saved.
    '''

    # Data extraction
    soc_data = res_rbc['env_h']['electrical_storage_soc']
    time_steps = range(res_rbc['env_h']['time_steps'])
    
    soc = np.array(soc_data)
    
    # Plot creation
    plt.figure(figsize=(20, 6))
    plt.plot(time_steps, soc, label='State of Charge (SoC)', linewidth=1.5, color='tab:orange')
    
    # Labels and title
    plt.title('Electrical Storage SOC')
    plt.xlabel('Time Step (Hours)')
    plt.ylabel('State of Charge (kWh/hWh_capacity)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")
    #plt.show()

def plot_dhw_storage(res_rbc, save_path='dhw_storage_soc.png'):
    '''
    Function to plot the State of Charge (SoC) and Demand of domestic hot water (DHW) storage over time.

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        save_path: Path where the plot will be saved.
    '''

    # Data extraction
    dhw_data = res_rbc['env_h']['dhw']
    time_steps = range(res_rbc['env_h']['time_steps'])
    
    dhw_soc = np.array(dhw_data['dhw_storage_soc'])
    dhw_demand = np.array(dhw_data['dhw_demand'])
    
    # Plot creation
    fig, ax1 = plt.subplots(figsize=(20, 6))

    # Title
    ax1.set_title('SoC and Demand of DHW Storage')

    # SoC line
    ax1.plot(time_steps, dhw_soc, label='DHW Storage SoC', linewidth=1.5, color='tab:blue')

    # Labels for SoC
    ax1.set_ylabel('State of Charge (kWh/kWh_capacity)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    # DHW Demand line
    ax2.plot(time_steps, dhw_demand, label='DHW Demand', linewidth=1.0, color='tab:red')

    # Labels for DHW Demand
    ax2.set_ylabel('DHW Demand (kWh)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")
    #plt.show()

def plot_grid_and_solar(res_rbc, save_path='grid_and_solar.png'):
    '''
    Function to plot electricity price and solar generation over time.

    Args:
        res_rbc: Dictionary containing the results from the RBC agent.
        save_path: Path where the plot will be saved.
    '''

    # Data extraction
    grid_solar_data = res_rbc['env_h']['grid_and_solar']
    time_steps = range(res_rbc['env_h']['time_steps'])
    
    elec_price = np.array(grid_solar_data['elec_price'])
    solar_generation = np.array(grid_solar_data['solar_generation'])
    
    # Plot creation
    fig, ax1 = plt.subplots(figsize=(20, 6))

    # Title
    ax1.set_title('Electricity Price and Solar Generation')

    # Electricity price line
    ax1.plot(time_steps, elec_price, label='Electricity Price', linewidth=1.5, color='tab:green')

    # Labels for electricity price
    ax1.set_ylabel('Electricity Price ($/kWh)')
    ax1.set_xlabel('Time Step (Hours)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    # Solar generation line
    ax2.plot(time_steps, solar_generation, label='Solar Generation', linewidth=1.0, color='tab:orange')

    # Labels for solar generation
    ax2.set_ylabel('Solar Generation (kWh)')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Plot saved in: {save_path}")
    #plt.show()

def evaluate_agent(agent_class, env):
    """
    Simulates episode and records results for the given agent in the provided environment.
    """
    agent = agent_class(env)
    
    # Environment reset
    obs, _ = env.reset()
    
    # Add variables to track history ???
    
    done = False
    while not done:
        actions = agent.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

    # The extracted results will be retrieved from the environment's building with index 0 (building used to simulate AdvancedRBC's actions in main_rbc.py)
    b = env.buildings[0]
    
    results = {
        'env_h': {
            'time_steps': env.time_steps - 1,

            'temperature': {
                # Temperature related data
                'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature[:-1],
                'indoor_dry_bulb_temperature_set_point': b.indoor_dry_bulb_temperature_cooling_set_point[:-1],
                'comfort_band': b.comfort_band,

                # kWh related data
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
    
    # Adjust comfort_band length if necessary
    cb = b.comfort_band
    if not hasattr(cb, '__len__'): # It's a scalar
         results['env_h']['temperature']['comfort_band'] = [cb] * results['env_h']['time_steps']
    elif len(cb) > results['env_h']['time_steps']:
         results['env_h']['temperature']['comfort_band'] = cb[:-1]
         
    return results

def main():
    # Same configuration as main_rbc.py for consistency
    conf = Config()
    args = conf.args
    
    if args.data is None:
        args.data = 'citylearn_challenge_2023_phase_1' # O quello che usi di solito

    # Load CityLearn schema
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data) # type: ignore

    # Creation of the CityLearn environment
    env = CityLearnEnv(schema=schema, central_agent=True)
    
    print(f"Start evaluation with dataset: {args.data}")
    
    # Evaluation of AdvancedRBC
    print("Running AdvancedRBC...")
    results = evaluate_agent(AdvancedRBC, env)
    
    # Plotting results
    plot_temperature(results, save_path='imgs/temperature_and_cooling_demand.png')
    plot_electrical_storage_soc(results, save_path='imgs/electrical_storage_soc.png')
    plot_dhw_storage(results, save_path='imgs/dhw_storage_soc_and_demand.png')
    plot_grid_and_solar(results, save_path='imgs/grid_and_solar.png')

if __name__ == '__main__':
    main()
