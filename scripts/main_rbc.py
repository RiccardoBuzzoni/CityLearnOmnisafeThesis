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

            # Cooling demand
            cooling_demand = o[available_obs.index('cooling_demand')]

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
            electrical_storage_soc = o[available_obs.index('electrical_storage_soc')]

            # DHW storage state of charge
            dhw_storage_soc = o[available_obs.index('dhw_storage_soc')]

            # DHW demand
            dhw_demand = o[available_obs.index('dhw_demand')]

            if 'cooling_device' in available_act:
                # Peak hours
                if 12 <= hour <= 17:
                    if indoor_temp > cooling_setpoint + self.comfort_band:
                        if occupant_present == 0:
                            action[available_act.index('cooling_device')] = 0.0
                        else:
                            # Carbon emission evaluation
                            if carbon_int < 0.40:
                                # Low electricity price
                                if elec_price <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.8
                                    else:
                                        action[available_act.index('cooling_device')] = 1.0
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.8
                                # Low electricity price + high solar generation
                                if elec_price <= 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.85
                                    else:
                                        action[available_act.index('cooling_device')] = 0.1
                                # High electricity price + high solar generation
                                if elec_price > 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    else:
                                        action[available_act.index('cooling_device')] = 0.9
                                # Default action
                                    action[available_act.index('cooling_device')] = 0.66
                            # High carbon emission
                            else:
                                # Low electricity price
                                if elec_price <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.7
                                    else:
                                        action[available_act.index('cooling_device')] = 0.9
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.5
                                    else:
                                        action[available_act.index('cooling_device')] = 0.7
                                # Low electricity price + high solar generation
                                if elec_price <= 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.8
                                    else:
                                        action[available_act.index('cooling_device')] = 1.0
                                # High electricity price + high solar generatio
                                if elec_price > 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # Default action
                                action[available_act.index('cooling_device')] = 0.55
                    
                    # Cooling demand evaluation
                    else:
                        if 0.7 <= cooling_demand <= 1.0:
                            action[available_act.index('cooling_device')] = 0.66
                        elif 0.3 <= cooling_demand < 0.7:
                            action[available_act.index('cooling_device')] = 0.4
                        else:
                            action[available_act.index('cooling_device')] = 0.2

                # Off-peak hours
                else:
                    # Indoor temperature avove setpoint + comfort band
                    if indoor_temp > cooling_setpoint + self.comfort_band:
                        # Carbon emission evaluation
                        if carbon_int < 0.40:
                            # Low electricity price
                            if elec_price <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.6
                                else:
                                    action[available_act.index('cooling_device')] = 0.8
                            # High electricity price
                            elif elec_price > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.25
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # Low electricity price + high solar generation
                            if elec_price <= 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.7
                                else:
                                    action[available_act.index('cooling_device')] = 0.9
                            # High electricity price + high solar generation
                            if elec_price > 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                else:
                                    action[available_act.index('cooling_device')] = 0.7
                            # Default action
                            action[available_act.index('cooling_device')] = 0.5
                        # High carbon emission
                        else:
                            # Low electricity price
                            if elec_price <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                else:
                                    action[available_act.index('cooling_device')] = 0.7
                            # High electricity price
                            elif elec_price > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                else:
                                    action[available_act.index('cooling_device')] = 0.5
                            # Low electricity price + high solar generation
                            if elec_price <= 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.6
                                else:
                                    action[available_act.index('cooling_device')] = 0.8
                            # High electricity price + high solar generation
                            if elec_price > 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.25
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # Default action
                            action[available_act.index('cooling_device')] = 0.44
                    
                    # Cooling demand evaluation
                    else:
                        if 0.7 <= cooling_demand <= 1.0:
                            action[available_act.index('cooling_device')] = 0.4
                        elif 0.3 <= cooling_demand < 0.7:
                            action[available_act.index('cooling_device')] = 0.2
                        else:
                            action[available_act.index('cooling_device')] = 0.1

            if 'electrical_storage' in available_act:
                if electrical_storage_soc == 1.0:
                    # Peak hours -> discharge
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        # High solar generation
                        if solar_generation > 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = -0.66
                            else:
                                action[available_act.index('electrical_storage')] = -0.85
                        elif 0.3 <= solar_generation <= 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = -0.45
                            else:
                                action[available_act.index('electrical_storage')] = -0.6
                        else:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = -0.3
                            else:
                                action[available_act.index('electrical_storage')] = -0.4
                    # Off-peak hours -> battery at max capacity so no action
                    else:
                        action[available_act.index('electrical_storage')] = 0.0

                elif 0.5 <= electrical_storage_soc < 1.0:
                    # Peak hours -> discharge
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        # High solar generation
                        if solar_generation > 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = -0.5
                            else:
                                action[available_act.index('electrical_storage')] = -0.6
                        elif 0.3 <= solar_generation <= 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = -0.35
                            else:
                                action[available_act.index('electrical_storage')] = -0.45
                        else:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = -0.15
                            else:
                                action[available_act.index('electrical_storage')] = -0.3
                    # Off-peak hours
                    else:
                        # High solar generation
                        if solar_generation > 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.5
                            else:
                                action[available_act.index('electrical_storage')] = 0.35
                        elif 0.3 <= solar_generation <= 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.4
                            else:
                                action[available_act.index('electrical_storage')] = 0.25
                        else:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.3
                            else:
                                action[available_act.index('electrical_storage')] = 0.15
                
                elif 0.2 <= electrical_storage_soc < 0.5:
                    # Peak hours -> low battery capacity so minimal discharge
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        action[available_act.index('electrical_storage')] = -0.1
                    # Off-peak hours -> charge
                    else:
                        # High solar generation
                        if solar_generation > 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.7
                            else:
                                action[available_act.index('electrical_storage')] = 0.55
                        elif 0.3 <= solar_generation <= 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.65
                            else:
                                action[available_act.index('electrical_storage')] = 0.4
                        else:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.5
                            else:
                                action[available_act.index('electrical_storage')] = 0.3
                
                # Very low state of charge -> charge
                else:
                    # Peak hours -> no action
                    if 8 <= hour <= 10 or 18 <= hour <= 21:
                        action[available_act.index('electrical_storage')] = 0.0
                    # Off-peak hours -> charge
                    else:
                        # High solar generation
                        if solar_generation > 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.9
                            else:
                                action[available_act.index('electrical_storage')] = 0.6
                        elif 0.3 <= solar_generation <= 0.6:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.7
                            else:
                                action[available_act.index('electrical_storage')] = 0.45
                        else:
                            if electricity_pricing < 0.03:
                                action[available_act.index('electrical_storage')] = 0.6
                            else:
                                action[available_act.index('electrical_storage')] = 0.4

            if 'dhw_storage' in available_act:
                if 0.7 <= dhw_storage_soc <= 1.0:
                    # Peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        # High demand or high electricity price
                        if dhw_demand >= 0.6 or electricity_pricing > 0.03:
                            action[available_act.index('dhw_storage')] = -0.6
                        # Low demand
                        elif dhw_demand < 0.3:
                            action[available_act.index('dhw_storage')] = -0.2
                        # Default action
                        else:
                            action[available_act.index('dhw_storage')] = -0.35
                    # Off-peak hours with possible high solar generation
                    elif 10 <= hour <= 16:
                        # High solar generation and low electricity price
                        if solar_generation > 0.6 and electricity_pricing < 0.03:
                            action[available_act.index('dhw_storage')] = 0.4
                        # Medium solar generation
                        elif 0.2 <= solar_generation <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.25
                        # Low solar generation -> default action
                        else:
                            action[available_act.index('dhw_storage')] = 0.0
                    # Night hours -> high dhw storage state of charge so no action
                    else:
                        action[available_act.index('dhw_storage')] = 0.0
                
                elif 0.4 <= dhw_storage_soc < 0.7:
                    # Peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        # High demand or high electricity price
                        if dhw_demand >= 0.6 or electricity_pricing > 0.03:
                            action[available_act.index('dhw_storage')] = -0.4
                        # Low demand
                        elif dhw_demand < 0.3:
                            action[available_act.index('dhw_storage')] = -0.15
                        # Default action
                        else:
                            action[available_act.index('dhw_storage')] = -0.25
                    # Off-peak hours with possible high solar generation
                    elif 10 <= hour <= 16:
                        # High solar generation and low electricity price
                        if solar_generation > 0.6 and electricity_pricing < 0.03:
                            action[available_act.index('dhw_storage')] = 0.65
                        # Medium solar generation
                        elif 0.2 <= solar_generation <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.35
                        # Low solar generation -> default action
                        else:
                            action[available_act.index('dhw_storage')] = 0.15
                    # Night hours -> high dhw storage state of charge so no action
                    else:
                        action[available_act.index('dhw_storage')] = 0.5

                # Very low state of charge
                else:
                    # Peak hours -> minimal discharge
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        action[available_act.index('dhw_storage')] = -0.01
                    # Off-peak hours with possible high solar generation
                    elif 10 <= hour <= 16:
                        # High solar generation and low electricity price
                        if solar_generation > 0.6 and electricity_pricing < 0.03:
                            action[available_act.index('dhw_storage')] = 0.8
                        # Medium solar generation
                        elif 0.2 <= solar_generation <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.6
                        # Low solar generation -> default action
                        else:
                            action[available_act.index('dhw_storage')] = 0.4
                    # Night hours -> low dhw storage state of charge so charge
                    else:
                        action[available_act.index('dhw_storage')] = 0.7

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

    # KPI logging: weekly and daily.
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