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
                        Logica:
                            Dalle 8 alle 18 (fascia oraria nella quale la domanda di energia
                            è più alta) vengono scaricate le batterie.
                            Al di fuori di questa fascia oraria le batterie vengono caricate.
                        '''

                        if 8 <= hour <= 18:
                            value = 0.9 # Stato di massima efficienza.
                        else:
                            value = 0.1 # Stato di minimo sforzo.

                        action_map[n][hour] = value
                
                elif n == 'dhw_storage':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        '''
                        Goal:
                            Fornire acqua calda nei momenti in cui la richiesta è più alta.
                        Logica:
                            Vengono definite due fasce orarie di riferimento. Per la mattina
                            viene fissata la fascia oraria che va dalle 6 alle 9, mentre per
                            la sera la fascia oraria che va dalle 18 alle 22. In questi archi
                            di tempo viene aumentata la produzione di calore.
                            Al di fuori di queste fasce orarie si mantiene una temperatura di
                            base per risparmiare energia.
                        '''

                        if 6 <= hour >= 9 or 18 <= hour >= 22:
                            value = 0.8 # Stato di alta efficienza.
                        else:
                            value = 0.2 # Stato conservativo.

                        action_map[n][hour] = value

                elif n == 'cooling_device':
                    for hour in Building.get_periodic_observation_metadata()['hour']:
                        '''
                        Goal:
                            Ottimizzare il comfort termico riducendo il consumo energetico.
                        Logica:
                            La fascia oraria che va dalle 10 alle 16 rappresenta il periodo nel
                            quale la temperatura esterna è più elevata, quindi in questo lasso
                            di tempo il raffreddamento viene attivato in modi più intenso.
                            Al di fuori di queste ore viene ridotto l'uso del sistema.
                        '''

                        if 10 <= hour >= 16:
                            value = 0.7 # Stato di alto sforzo.
                        else:
                            value = 0.3 # Stato di sforzo minore.

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