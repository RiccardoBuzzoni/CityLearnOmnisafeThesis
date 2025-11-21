import sys, os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CityLearn.citylearn.agents.base import Agent
from CityLearn.citylearn.agents.rbc import BasicRBC, HourRBC, OptimizedRBC
from CityLearn.citylearn.citylearn import CityLearnEnv, Building

# Core
from typing import Mapping, Union

import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from utils import *

class CustomRBC(BasicRBC):

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    @HourRBC.action_map.setter
    def action_map(self, action_map: Union[List[Mapping[str, Mapping[int, float]]], Mapping[str, Mapping[int, float]], Mapping[int, float]]):
        if action_map is None:
            action_map = {}
            action_names = [a_ for a in self.action_names for a_ in a]
            action_names = list(set(action_names))


            for n in action_names:
                action_map[n] = {}

                if 'electrical_storage' in n:
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        # TODO: Implement RBC policy

                        action_map[n][hour] = value
                
                elif n == 'dhw_storage':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        # TODO: Implement RBC policy

                        action_map[n][hour] = value

                elif n == 'cooling_device':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        # TODO: Implement RBC policy

                        action_map[n][hour] = value
                
                else:
                    raise ValueError(f'Unknown action name: {n}')
                
        # Imposta la mappa nella superclasse
        HourRBC.action_map.fset(self, action_map)

class AdvancedRBC(Agent):
    """
    Advanced Rule-Based Controller (RBC) Agent with comfort band consideration.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment to perform control on.
    band: float
        Comfort band to try to satisfy. 

    """
    def __init__(self, env: CityLearnEnv, band: float=2.0, **kwargs):

        # Init OptimizedRBC
        super().__init__(env, **kwargs)

        # Comfort band (+/-) to satisfy
        self.comfort_band = band 

    def predict(self, observations: List[List[float]], deterministic: bool = True) -> List[List[float]]:        
            
        actions = []
        for i, o in enumerate(observations):

            # Available spaces
            available_obs = self.observation_names[i]
            available_act = self.action_names[i]
            action = [0.0 for _ in range(len(available_act))]

            # Indoor temperature and setpoints
            indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]

            # Outdoor temperature
            outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature')]
            predicted_outdoor_temperature = o[available_obs.index('outdoor_dry_bulb_temperature_predicted_1')]

            # Electricity pricing
            electricity_pricing = o[available_obs.index('electricity_pricing')]

            # Emission
            carbon_intensity = o[available_obs.index('carbon_intensity')]

            # Solar generation
            solar_generation = o[available_obs.index('solar_generation')]

            # Occupatns presence
            occupants_present = o[available_obs.index('occupant_count')]    

            # Hours of the day
            hour = o[available_obs.index('hour')]

            # Electrical storage state of charge
            electrical_soc = o[available_obs.index('electrical_storage_soc')]

            # DHW storage state of charge
            dhw_storage_soc = o[available_obs.index('dhw_storage_soc')]

            # DHW demand
            dhw_demand = o[available_obs.index('dhw_demand')]

            if 'cooling_device' in available_act:
                # Peak hours
                if 12 <= hour <= 16:
                    # Indoor temperature avove setpoint + comfort band
                    if indoor_temp > cooling_setpoint + self.comfort_band:
                        if occupants_present == 0:
                            action[available_act.index('cooling_device')] = 0.0
                        else:
                            # Carbon emission evaluation
                            if carbon_intensity < 0.40:
                                # Low electricity price
                                if electricity_pricing <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.8
                                # High electricity price
                                elif electricity_pricing > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # Low electricity price + high solar generation
                                if electricity_pricing <= 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    else:
                                        action[available_act.index('cooling_device')] = 0.9
                                # High electricity price + high solar generation
                                if electricity_pricing > 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    else:
                                        action[available_act.index('cooling_device')] = 0.7
                                # Default action
                                    action[available_act.index('cooling_device')] = 0.66
                            # High carbon emission
                            else:
                                # Low electricity price
                                if electricity_pricing <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    else:
                                        action[available_act.index('cooling_device')] = 0.7
                                # High electricity price
                                elif electricity_pricing > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    else:
                                        action[available_act.index('cooling_device')] = 0.5
                                # Low electricity price + high solar generation
                                if electricity_pricing <= 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.8
                                # High electricity price + high solar generation
                                if electricity_pricing > 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # Default action
                                action[available_act.index('cooling_device')] = 0.55
                    
                    else:
                        action[available_act.index('cooling_device')] = 0.0
                
                # Off-peak hours
                else:
                    # Indoor temperature above setpoint
                    if indoor_temp > cooling_setpoint:
                        if occupants_present == 0:
                            action[available_act.index('cooling_device')] = 0.0
                        else:
                            # Carbon emission evaluation
                            if carbon_intensity < 0.40:
                                # Low electricity price
                                if electricity_pricing <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    else:
                                        action[available_act.index('cooling_device')] = 0.5
                                # High electricity price
                                elif electricity_pricing > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    else:
                                        action[available_act.index('cooling_device')] = 0.3
                                # Low electricity price + high solar generation
                                if electricity_pricing <= 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # High electricity price + high solar generation
                                if electricity_pricing > 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    else:
                                        action[available_act.index('cooling_device')] = 0.4
                                # Default action
                                action[available_act.index('cooling_device')] = 0.33
                            # High carbon emission
                            else:
                                # Low electricity price
                                if electricity_pricing <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    else:
                                        action[available_act.index('cooling_device')] = 0.4
                                # High electricity price
                                elif electricity_pricing > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.001
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    else:
                                        action[available_act.index('cooling_device')] = 0.2
                                # Low electricity price + high solar generation
                                if electricity_pricing <= 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    else:
                                        action[available_act.index('cooling_device')] = 0.5
                                # High electricity price + high solar generation
                                if electricity_pricing > 0.03 and solar_generation > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.025
                                    elif predicted_outdoor_temperature < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    else:
                                        action[available_act.index('cooling_device')] = 0.3
                                # Default action
                                action[available_act.index('cooling_device')] = 0.22
                    
                    # Indoor temperature avove setpoint + comfort band
                    elif indoor_temp > cooling_setpoint + self.comfort_band:
                        # Carbon emission evaluation
                        if carbon_intensity < 0.40:
                            # Low electricity price
                            if electricity_pricing <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # High electricity price
                            elif electricity_pricing > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.05
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                else:
                                    action[available_act.index('cooling_device')] = 0.4
                            # Low electricity price + high solar generation
                            if electricity_pricing <= 0.03 and solar_generation > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                else:
                                    action[available_act.index('cooling_device')] = 0.7
                            # High electricity price + high solar generation
                            if electricity_pricing > 0.03 and solar_generation > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.1
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                else:
                                    action[available_act.index('cooling_device')] = 0.5
                            # Default action
                            action[available_act.index('cooling_device')] = 0.44
                        # High carbon emission
                        else:
                            # Low electricity price
                            if electricity_pricing <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.1
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                else:
                                    action[available_act.index('cooling_device')] = 0.5
                            # High electricity price
                            elif electricity_pricing > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.025
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.1
                                else:
                                    action[available_act.index('cooling_device')] = 0.3
                            # Low electricity price + high solar generation
                            if electricity_pricing <= 0.03 and solar_generation > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # High electricity price + high solar generation
                            if electricity_pricing > 0.03 and solar_generation > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.05
                                elif predicted_outdoor_temperature < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                else:
                                    action[available_act.index('cooling_device')] = 0.4
                            # Default action
                            action[available_act.index('cooling_device')] = 0.22
                    
                    else:
                        action[available_act.index('cooling_device')] = 0.0

            if 'electrical_storage' in available_act:
                if electrical_soc == 1.0:
                    action[available_act.index('electrical_storage')] = -0.7  # Discharge
                elif electrical_soc == 0.0:
                    action[available_act.index('electrical_storage')] = 0.7   # Charge
                
                else:
                    # Peak hours
                    if 8 <= hour <= 18:
                        # High electricity price -> high discharge
                        if electricity_pricing > 0.03:
                            action[available_act.index('electrical_storage')] = -0.6
                        # Low electricity price -> medium discharge
                        else:
                            action[available_act.index('electrical_storage')] = -0.4
                    
                    # High solar generation during peak hours -> charge
                    elif 10 <= hour <= 15 and solar_generation > 0.2:
                        action[available_act.index('electrical_storage')] = 0.5
                    
                    # Off-peak hours -> charge  
                    elif 0 <= hour <= 5:
                        # Low electricity price -> high charge
                        if electricity_pricing <= 0.03:
                            action[available_act.index('electrical_storage')] = 0.5
                        # High electricity price -> medium charge
                        elif electricity_pricing > 0.03:
                            action[available_act.index('electrical_storage')] = 0.3
                        # Low electricity price + high solar generation
                        elif electricity_pricing <= 0.03 and solar_generation > 0.2:
                            action[available_act.index('electrical_storage')] = 0.7

                    # Default action
                    else:
                        if electrical_soc >= 0.5:
                            action[available_act.index('electrical_storage')] = -0.3
                        else:
                            action[available_act.index('electrical_storage')] = 0.3


            if 'dhw_storage' in available_act:
                if dhw_storage_soc == 1.0:
                    action[available_act.index('dhw_storage')] = -0.5
                elif dhw_storage_soc == 0.0:
                    action[available_act.index('dhw_storage')] = 0.5

                else:
                    # Peak hours or high DHW demand
                    if (8 <= hour <= 10) or (18 <= hour <= 21) or (dhw_demand > 0.6):
                        # High electricity price -> discharge
                        if electricity_pricing > 0.03:
                            action[available_act.index('dhw_storage')] = -0.4
                        # Low electricity price -> lower discharge
                        else:
                            action[available_act.index('dhw_storage')] = -0.2
                    # High solar generation during peak hours -> charge
                    elif 10 <= hour <= 15 and solar_generation > 0.2:
                        action[available_act.index('dhw_storage')] = 0.4
                    
                    # Off-peak hours -> charge
                    elif 0 <= hour <= 5:
                        # Low electricity price -> high charge
                        if electricity_pricing <= 0.03:
                            action[available_act.index('dhw_storage')] = 0.4
                        # High electricity price -> medium charge
                        elif electricity_pricing > 0.03:
                            action[available_act.index('dhw_storage')] = 0.2
                        # Low electricity price + high solar generation
                        elif electricity_pricing <= 0.03 and solar_generation > 0.2:
                            action[available_act.index('dhw_storage')] = 0.5
                
                    # Default action
                    else:
                        if dhw_storage_soc >= 0.5:
                            action[available_act.index('dhw_storage')] = -0.33
                        else:
                            action[available_act.index('dhw_storage')] = 0.33

            # Actions value per hour
            debug_action_dict = {}
            if 'cooling_device' in available_act:
                debug_action_dict['cooling_device'] = action[available_act.index('cooling_device')]
            if 'dhw_storage' in available_act:
                debug_action_dict['dhw_storage'] = action[available_act.index('dhw_storage')]
            if 'electrical_storage' in available_act:
                debug_action_dict['electrical_storage'] = action[available_act.index('electrical_storage')]

            print(f"[DEBUG] Hour (from obs): {hour:.0f}, Actions: {debug_action_dict}")

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions


def run_simulation(agent, env):
    print("Starting simulation...")
    print(f"Agent: {agent.__class__.__name__}")
    observations, _ = env.reset()
    max_steps = env.time_steps - 1

    # KPI logging: weekly and daily. Each time step is 1 hour.
    weekly_interval_steps = 7 * 24  # 168 steps
    daily_interval_steps = 1 * 24   # 24 steps
    kpi_dir = 'kpi_logs'
    weekly_kpi_file = os.path.join(kpi_dir, f'{agent.__class__.__name__}_weekly_kpis_log.txt')
    daily_kpi_file = os.path.join(kpi_dir, f'{agent.__class__.__name__}_daily_kpis_log.txt')
    os.makedirs(kpi_dir, exist_ok=True)

    def _append_kpis(step_index: int, filepath: str):
        try:
            kpis = get_kpis(env)
            # Filter district-level KPIs
            kpis = kpis[kpis['level'] == 'district']

            day = step_index // 24
            with open(filepath, 'a') as fh:
                fh.write(f'--- STEP: {step_index} (day {day}) ---\n')
                for _, row in kpis.iterrows():
                    fh.write(f"{row['kpi']}: {row['value']}\n")
                fh.write('\n')
        except Exception as e:
            # Debugging log
            with open(filepath, 'a') as fh:
                fh.write(f'Failed to write KPIs at step {step_index}: {e}\n\n')

    # Initial KPI snapshots at start (step 0)
    _append_kpis(0, daily_kpi_file)
    _append_kpis(0, weekly_kpi_file)

    step_index = 0
    for _ in range(max_steps):
        actions = agent.predict(observations)
        observations, reward, terminated, truncated, info = env.step(actions)
        step_index += 1

        # Write daily KPIs every daily_interval_steps
        if step_index % daily_interval_steps == 0:
            _append_kpis(step_index, daily_kpi_file)

        # Write weekly KPIs every weekly_interval_steps
        if step_index % weekly_interval_steps == 0:
            _append_kpis(step_index, weekly_kpi_file)

    print("Simulation completed.\n")

def main(args):

    # Get schema from CityLearn dataset
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)

    # Create CityLearn environment
    env_1 = CityLearnEnv(schema=schema, central_agent=True)
    env_2 = CityLearnEnv(schema=schema, central_agent=True)

    agent = AdvancedRBC(env_1)
    baseline_agent = OptimizedRBC(env_2)

    run_simulation(agent, env_1)
    run_simulation(baseline_agent, env_2)

    # Compare results
    plot_district_kpis(
        {'AdvancedRBC': env_1, 'OptimizedRBC': env_2},
        base_path='imgs'
    )

if __name__ == '__main__':
    # Configurations
    conf = Config()
    args = conf.args

    main(args)