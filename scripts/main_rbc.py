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

            # TODO add other observations if needed
            # TODO implement more advanced RBC logic for each device

            indoor_temp = o[available_obs.index('indoor_dry_bulb_temperature')]
            outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature')]
            predicted_outdoor_temp = o[available_obs.index('outdoor_dry_bulb_temperature_predicted_1')]
            cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]
            solar_gen = o[available_obs.index('solar_generation')]
            dhw_demand = o[available_obs.index('dhw_demand')]
            occupant_present = o[available_obs.index('occupant_count')]
            carbon_int = o[available_obs.index('carbon_intensity')]
            elec_price = o[available_obs.index('electricity_pricing')]
            hour = o[available_obs.index('hour')]
            electrical_storage_soc = o[available_obs.index('electrical_storage_soc')] 
            dhw_storage_soc = o[available_obs.index('dhw_storage_soc')] 
            

            if 'cooling_device' in available_act:
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
                                        action[available_act.index('cooling_device')] = 0.4
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.6
                                    else:
                                        action[available_act.index('cooling_device')] = 0.8
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
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
                                    action[available_act.index('cooling_device')] = 0.66
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
                                        action[available_act.index('cooling_device')] = 0.1
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
                                        action[available_act.index('cooling_device')] = 0.2
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # Default action
                                action[available_act.index('cooling_device')] = 0.55
                    
                    else:
                        action[available_act.index('cooling_device')] = 0.0
                
                # Off-peak hours
                else:
                    action[available_act.index('cooling_device')] = 0.0  # Turn off cooling
                    # Indoor temperature above setpoint
                    if indoor_temp > cooling_setpoint:
                        if occupant_present == 0:
                            action[available_act.index('cooling_device')] = 0.0
                        else:
                            # Carbon emission evaluation
                            if carbon_int < 0.40:
                                # Low electricity price
                                if elec_price <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    else:
                                        action[available_act.index('cooling_device')] = 0.5
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    else:
                                        action[available_act.index('cooling_device')] = 0.3
                                # Low electricity price + high solar generation
                                if elec_price <= 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.4
                                    else:
                                        action[available_act.index('cooling_device')] = 0.6
                                # High electricity price + high solar generation
                                if elec_price > 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    else:
                                        action[available_act.index('cooling_device')] = 0.4
                                # Default action
                                action[available_act.index('cooling_device')] = 0.33
                            # High carbon emission
                            else:
                                # Low electricity price
                                if elec_price <= 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.2
                                    else:
                                        action[available_act.index('cooling_device')] = 0.4
                                # High electricity price
                                elif elec_price > 0.03:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.001
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.05
                                    else:
                                        action[available_act.index('cooling_device')] = 0.2
                                # Low electricity price + high solar generation
                                if elec_price <= 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.3
                                    else:
                                        action[available_act.index('cooling_device')] = 0.5
                                # High electricity price + high solar generation
                                if elec_price > 0.03 and solar_gen > 0.2:
                                    if outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.025
                                    elif predicted_outdoor_temp < cooling_setpoint:
                                        action[available_act.index('cooling_device')] = 0.1
                                    else:
                                        action[available_act.index('cooling_device')] = 0.3
                                # Default action
                                action[available_act.index('cooling_device')] = 0.22
                    
                    # Indoor temperature avove setpoint + comfort band
                    elif indoor_temp > cooling_setpoint + self.comfort_band:
                        # Carbon emission evaluation
                        if carbon_int < 0.40:
                            # Low electricity price
                            if elec_price <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # High electricity price
                            elif elec_price > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.05
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                else:
                                    action[available_act.index('cooling_device')] = 0.4
                            # Low electricity price + high solar generation
                            if elec_price <= 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.5
                                else:
                                    action[available_act.index('cooling_device')] = 0.7
                            # High electricity price + high solar generation
                            if elec_price > 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.1
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                else:
                                    action[available_act.index('cooling_device')] = 0.5
                            # Default action
                            action[available_act.index('cooling_device')] = 0.44
                        # High carbon emission
                        else:
                            # Low electricity price
                            if elec_price <= 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.1
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.3
                                else:
                                    action[available_act.index('cooling_device')] = 0.5
                            # High electricity price
                            elif elec_price > 0.03:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.025
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.1
                                else:
                                    action[available_act.index('cooling_device')] = 0.3
                            # Low electricity price + high solar generation
                            if elec_price <= 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.4
                                else:
                                    action[available_act.index('cooling_device')] = 0.6
                            # High electricity price + high solar generation
                            if elec_price > 0.03 and solar_gen > 0.2:
                                if outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.05
                                elif predicted_outdoor_temp < cooling_setpoint:
                                    action[available_act.index('cooling_device')] = 0.2
                                else:
                                    action[available_act.index('cooling_device')] = 0.4
                            # Default action
                            action[available_act.index('cooling_device')] = 0.22
                    
                    else:
                        action[available_act.index('cooling_device')] = 0.0

            if 'electrical_storage' in available_act:
                if electrical_storage_soc == 1.0:
                    # Peak hours
                    if 8 <= hour <= 18:
                        if solar_gen > 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = -0.66  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = -0.85
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = -0.45  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = -0.6
                        else:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = -0.3  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = -0.5
                    # Not peak hours
                    else:
                        action[available_act.index('electrical_storage')] = 0.0

                elif 0.5 <= electrical_storage_soc < 1.0:
                     # Peak hours
                    if 8 <= hour <= 18:
                        if solar_gen > 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = -0.5  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = -0.6
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = -0.35  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = -0.45
                        else:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = -0.15  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = -0.3
                    # Not peak hours
                    else:
                        if solar_gen > 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = 0.5  # charge
                            else:
                                action[available_act.index('electrical_storage')] = 0.35
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = 0.4  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = 0.25
                        else:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = 0.3  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = 0.15
                
                elif 0.2 <= electrical_storage_soc < 0.5:
                    # Peak hours
                    if 8 <= hour <= 18:
                        action[available_act.index('electrical_storage')] = -0.1
                    # Not peak hours
                    else:
                        if solar_gen > 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = 0.8  # charge
                            else:
                                action[available_act.index('electrical_storage')] = 0.55
                        elif 0.3 <= solar_gen <= 0.6:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = 0.65  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = 0.4
                        else:
                            if elec_price < 0.3:
                                action[available_act.index('electrical_storage')] = 0.5  # discharge
                            else:
                                action[available_act.index('electrical_storage')] = 0.3

                # Battery almost empty
                elif electrical_storage_soc < 0.2:
                    action[available_act.index('electrical_storage')] = 0.9
            



            if 'dhw_storage' in available_act:
                if 0.7 <= dhw_storage_soc <= 1.0:
                    #peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        if dhw_demand > 0.6 and elec_price < 0.03: #high demand but low price - high discharge
                            action[available_act.index('dhw_storage')] = -0.6
                        elif dhw_demand > 0.3 and elec_price > 0.03: #high demand but high price - moderate discharge
                            action[available_act.index('dhw_storage')] = -0.3
                        else: 
                            action[available_act.index('dhw_storage')] = -0.2
                    #non peak hours
                    elif 10 <= hour <= 17 or solar_gen > 0.4:
                        action[available_act.index('dhw_storage')] = 0.7  
                    elif 10 <= hour <= 17 or solar_gen <= 0.4:
                        action[available_act.index('dhw_storage')] = 0.45  
                    #non peak hours - night
                    elif 0 <= hour < 6 or hour == 23:
                        if elec_price <= 0.03:
                            action[available_act.index('dhw_storage')] = 0.7 #charge during night with low electricity price
                        elif elec_price > 0.03:
                            action[available_act.index('dhw_storage')] = 0.3 #charge during night with high electricity price
                
                elif 0.5 <= dhw_storage_soc < 0.7:
                    
                    #peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        if dhw_demand > 0.6 and elec_price < 0.03: #high demand but low price - high discharge
                            action[available_act.index('dhw_storage')] = -0.5
                        elif dhw_demand > 0.3 and elec_price > 0.03: #high demand but high price - moderate discharge
                            action[available_act.index('dhw_storage')] = -0.25
                        else: 
                            action[available_act.index('dhw_storage')] = -0.15
                    #non peak hours
                    elif 10 <= hour <= 17 or solar_gen > 0.4:
                        action[available_act.index('dhw_storage')] = 0.66  
                    elif 10 <= hour <= 17 or solar_gen <= 0.4:
                        action[available_act.index('dhw_storage')] = 0.4  
                    #non peak hours - night
                    elif 0 <= hour < 6 or hour == 23:
                        if elec_price <= 0.03:
                            action[available_act.index('dhw_storage')] = 0.8 #charge during night with low electricity price
                        elif elec_price > 0.03:
                            action[available_act.index('dhw_storage')] = 0.5 #charge during night with high electricity price
                
                elif 0.2 <= dhw_storage_soc <= 0.4:
                    #peak hours
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        if dhw_demand > 0.6 and elec_price < 0.03: #high demand but low price - high discharge
                            action[available_act.index('dhw_storage')] = -0.7
                        elif dhw_demand > 0.3 and elec_price > 0.03: #high demand but high price - moderate discharge
                            action[available_act.index('dhw_storage')] = -0.35
                        else: 
                            action[available_act.index('dhw_storage')] = -0.25
                    #non peak hours
                    elif 10 <= hour <= 17 or solar_gen > 0.4:
                        action[available_act.index('dhw_storage')] = 0.8  
                    elif 10 <= hour <= 17 or solar_gen <= 0.4:
                        action[available_act.index('dhw_storage')] = 0.5  
                    #non peak hours - night
                    elif 0 <= hour < 6 or hour == 23:
                        if elec_price <= 0.03:
                            action[available_act.index('dhw_storage')] = 0.9 #charge during night with low electricity price
                        elif elec_price > 0.03:
                            action[available_act.index('dhw_storage')] = 0.6 #charge during night with high electricity price
                else: # dhw_storage_soc < 0.2
                    action[available_act.index('dhw_storage')] = 0.9 #charge    

            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions

def print_observations_per_step(observations, observation_names, step):
    print(f"\n==================== TIMESTEP {step} ====================")
    for b_idx, obs in enumerate(observations):
        print(f"\n--- Building {b_idx} ---")
        for name, value in zip(observation_names[b_idx], obs):
            print(f"{name:40s} : {value}")

def run_simulation(agent, env):
    print("Starting simulation...")
    print(f"Agent: {agent.__class__.__name__}")
    observations, _ = env.reset()
    max_steps = env.time_steps - 1

    # print initial observations
    print_observations_per_step(
        observations,
        env.observation_names,
        step=0
    )

    for step in range(1, max_steps):
        actions = agent.predict(observations)
        observations, reward, terminated, truncated, info = env.step(actions)

        # print observations every step
        print_observations_per_step(
            observations,
            env.observation_names,
            step=step
        )

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

    # Debug: Print available observations for the first building
    print("Available observations:", env_1.observation_names[0])

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