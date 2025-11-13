import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CityLearn.citylearn.agents.rbc import BasicRBC, HourRBC, OptimizedRBC
from CityLearn.citylearn.citylearn import CityLearnEnv, Building
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                        '''
                        Goal:
                            Ottimizzare l'utilizzo delle batterie con lo scopo di ridurre
                            i costi e massimizzare l'efficenza.
                        '''

                        if 8 <= hour <= 18: # Scarica ad alta intensità nelle ore di punta.
                            value = -0.6
                        elif 22 <= hour <= 5:
                            value = 0.4 # Ricarica ad intensità media.
                        else:
                            value = 0.0

                        action_map[n][hour] = value
                
                elif n == 'dhw_storage':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        '''
                        Goal:
                            Bilanciare comfort ed efficienza fornendo piu' calore nelle ore
                            di punta.
                        '''

                        if 6 <= hour <= 9 or 18 <= hour <= 22: # Efficienza moderata nelle ore di picco.
                            value = 0.5
                        elif 12 <= hour <= 15: # Efficienza medio/bassa nell'ora di pranzo
                            value = 0.3
                        else:
                            value = 0.0 #Stato conservativo.

                        action_map[n][hour] = value

                elif n == 'cooling_device':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        '''
                        Goal:
                            Ottimizzare il comfort termico riducendo il consumo energetico.
                        '''

                        if 10 <= hour <= 16: # Intensità elevata nelle ore più calde.
                            value = 0.7
                        elif 6 <= hour <= 9 or 17 <= hour <= 19: # Intensità intermedia nelle ore in cui potrebbe ancora esserci caldo moderato.
                            value = 0.4
                        else:
                            value = 0.0 # Stato conservativo.

                        action_map[n][hour] = value
                
                else:
                    raise ValueError(f'Unknown action name: {n}')
                
        # Imposta la mappa nella superclasse
        HourRBC.action_map.fset(self, action_map)

def run_simulation(agent, env):
    print("Starting simulation...")
    print(f"Agent: {agent.__class__.__name__}")
    observations, _ = env.reset()
    max_steps = env.time_steps - 1
    for _ in range(max_steps):
        actions = agent.predict(observations)
        observations, reward, terminated, truncated, info = env.step(actions)

    print("Simulation completed.")

def main(args):

    # Get schema from CityLearn dataset
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)

    # Create CityLearn environment
    env_1 = CityLearnEnv(schema=schema, central_agent=True)
    env_2 = CityLearnEnv(schema=schema, central_agent=True)
    agent = CustomRBC(env_1)
    baseline_agent = OptimizedRBC(env_2)

    run_simulation(agent, env_1)
    run_simulation(baseline_agent, env_2)

    # Compare results
    plot_district_kpis(
        {'CustomRBC': env_1, 'OptimizedRBC': env_2},
        base_path='imgs'
    )

if __name__ == '__main__':
    # Configurations
    conf = Config()
    args = conf.args

    main(args)