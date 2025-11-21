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
            predicted_outdoor_temp_6h = o[available_obs.index('outdoor_dry_bulb_temperature_predicted_1')]
            cooling_setpoint = o[available_obs.index('indoor_dry_bulb_temperature_cooling_set_point')]
            solar_gen = o[available_obs.index('solar_generation')]
            dhw_demand = o[available_obs.index('dhw_demand')]
            occupant_present = o[available_obs.index('occupant_count')]
            carbon_int = o[available_obs.index('carbon_intensity')]
            elec_price = o[available_obs.index('electricity_pricing')]
            elec_price_pred = o[available_obs.index('electricity_pricing_predicted_1')]
            hour = o[available_obs.index('hour')]
            electrical_storage_soc = o[available_obs.index('electrical_storage_soc')] #O to 1
            dhw_storage_soc = o[available_obs.index('dhw_storage_soc')] # 0 to 1
            

            if 'cooling_device' in available_act:
                if 'cooling_device' in available_act:

                    idx = available_act.index('cooling_device')

                    # 0) Se non ci sono persone --> niente raffreddamento per risparmiare energia
                    if occupant_present == 0:
                        action[idx] = 0.0

                    else:
                        # 1) Logica temperatura interna
                        too_hot = indoor_temp > cooling_setpoint + self.comfort_band
                        slightly_hot = indoor_temp > cooling_setpoint

                        # 2) Condizioni "ambientali" sfavorevoli (prezzo e carbon intensity)
                        high_price = elec_price > 0.25      # prezzo alto
                        high_carbon = carbon_int > 0.45     # CO2 alta
                        very_hot_outdoor = outdoor_temp > 30

                        # 3) Fascia "calda" della giornata
                        peak_cooling_hours = 12 <= hour <= 17

                        # ---- DECISIONE RBC ----

                        if too_hot:
                            # Fa caldo e ci sono persone
                            if high_price or high_carbon:
                                # Evita consumi alti in condizioni sfavorevoli
                                action[idx] = 0.4
                            else:
                                # Raffredda forte solo quando conviene
                                action[idx] = 0.9

                        elif slightly_hot:
                            # Temp leggermente sopra il setpoint
                            if peak_cooling_hours or very_hot_outdoor:
                                action[idx] = 0.5
                            else:
                                action[idx] = 0.3

                        else:
                            # Temp confortevole
                            if peak_cooling_hours and solar_gen > 0.3:
                                # Usiamo surplus solare per precool
                                action[idx] = 0.25
                            else:
                                action[idx] = 0.0


            if 'electrical_storage' in available_act:
                if electrical_storage_soc == 1.0:
                    
                    #peak hours
                    if 17 <= hour <= 22:        
                        if elec_price > 0.03 or solar_gen > 0.3:
                            action[available_act.index('electrical_storage')] = -0.7 #discharge during peak hours with high electricity price
                        elif elec_price <= 0.03 or solar_gen <= 0.3:
                            action[available_act.index('electrical_storage')] = -0.35 #discharge during peak hours with low electricity price
                    #non peak
                    elif 8 <= hour <= 16:
                        if elec_price > 0.03 or solar_gen > 0.3:
                            action[available_act.index('electrical_storage')] = 0.35 #low charge during non-peak hours with high electricity price
                        elif elec_price <= 0.03 or solar_gen >= 0.3:
                            action[available_act.index('electrical_storage')] = 0.75 #high charge during non-peak hours with low electricity price
                        elif elec_price <= 0.03 or solar_gen < 0.3:
                            action[available_act.index('electrical_storage')] = 0.55 #charge during non-peak hours with low electricity price and high solar generation
                    
                    #non peak hours - night
                    if 0 <= hour <= 7 or hour == 23:
                        if elec_price <= 0.03:
                            action[available_act.index('electrical_storage')] = 0.75 #charge during night with low electricity price
                        elif elec_price_pred > 0.03:
                            action[available_act.index('electrical_storage')] = 0.35 #discharge during night with high electricity price
                
                elif 0.5 <= electrical_storage_soc < 1.0:
                    #peak hours
                    if 17 <= hour <= 22:        
                        if elec_price > 0.03 or solar_gen > 0.3:
                            action[available_act.index('electrical_storage')] = -0.5 #discharge during peak hours with high electricity price
                        elif elec_price <= 0.03 or solar_gen <= 0.3:
                            action[available_act.index('electrical_storage')] = -0.25 #discharge during peak hours with low electricity price
                    
                    #non peak
                    elif 8 <= hour <= 16:
                        if elec_price > 0.03 or solar_gen > 0.3:
                            action[available_act.index('electrical_storage')] = 0.45 #low charge during non-peak hours with high electricity price
                        elif elec_price <= 0.03 or solar_gen >= 0.3:
                            action[available_act.index('electrical_storage')] = 0.8 #high charge during non-peak hours with low electricity price
                        elif elec_price <= 0.03 or solar_gen < 0.3:
                            action[available_act.index('electrical_storage')] = 0.35 #charge during non-peak hours with low electricity price and high solar generation
                    
                    #non peak hours - night
                    if 0 <= hour <= 7 or hour == 23:
                        if elec_price <= 0.03:
                            action[available_act.index('electrical_storage')] = 0.9 #charge during night with low electricity price
                        elif elec_price_pred > 0.03:
                            action[available_act.index('electrical_storage')] = 0.45 #discharge during night with high electricity price

                else: # electrical_storage_soc < 0.5
                    action[available_act.index('electrical_storage')] = 0.75 #charge
            
            
            
            if 'dhw_storage' in available_act:
                if dhw_storage_soc == 1.0:
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
                        action[available_act.index('dhw_storage')] = 0.4  
                    #non peak hours - night
                    elif 0 <= hour < 6 or hour == 23:
                        if elec_price_pred <= 0.03:
                            action[available_act.index('dhw_storage')] = 0.7 #charge during night with low electricity price
                        elif elec_price_pred > 0.03:
                            action[available_act.index('dhw_storage')] = 0.3 #charge during night with high electricity price
                
                elif 0.5 <= dhw_storage_soc < 1.0:
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
                        action[available_act.index('dhw_storage')] = 0.8  
                    elif 10 <= hour <= 17 or solar_gen <= 0.4:
                        action[available_act.index('dhw_storage')] = 0.5  
                    #non peak hours - night
                    elif 0 <= hour < 6 or hour == 23:
                        if elec_price_pred <= 0.03:
                            action[available_act.index('dhw_storage')] = 0.9 #charge during night with low electricity price
                        elif elec_price_pred > 0.03:
                            action[available_act.index('dhw_storage')] = 0.45 #charge during night with high electricity price
                else: # dhw_storage_soc < 0.5
                    action[available_act.index('dhw_storage')] = 0.75 #charge


            actions.append(action)

        # Return overwritten actions
        self.actions = actions
        return actions


def run_simulation(agent, env):
    print("Starting simulation...")
    print(f"Agent: {agent.__class__.__name__}")
    observations, _ = env.reset()
    max_steps = env.time_steps - 1
    for _ in range(max_steps):
        actions = agent.predict(observations)
        observations, reward, terminated, truncated, info = env.step(actions)

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