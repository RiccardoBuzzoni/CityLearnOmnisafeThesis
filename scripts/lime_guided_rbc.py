import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CityLearn.citylearn.agents.base import Agent
from CityLearn.citylearn.agents.rbc import BasicRBC, HourRBC, OptimizedRBC
from CityLearn.citylearn.citylearn import CityLearnEnv, Building

# Core
from typing import Mapping, Union

import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from utils import *
from main_rbc import AdvancedRBC as MainAdvancedRBC

'''
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
'''

class LimeGuidedRBC(Agent):
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

            # Electricity pricing
            elec_price = o[available_obs.index('electricity_pricing')]

            # Solar generation
            solar_gen = o[available_obs.index('solar_generation')]

            # Hours of the day
            hour = o[available_obs.index('hour')]

            # DHW storage state of charge
            dhw_storage_soc = o[available_obs.index('dhw_storage_soc')]

            # DHW demand
            dhw_demand = o[available_obs.index('dhw_demand')]

            # ------------------------------------------------------------------
            # COOLING DEVICE
            # Features usate (LIME top-4): indoor_temp, cooling_setpoint,
            #                              outdoor_temp, solar_generation
            # Logica: verifica comfort band -> se temp fuori range usa solar_gen
            # e outdoor_temp per scalare l'azione; altrimenti segue la domanda.
            # ------------------------------------------------------------------
            if 'cooling_device' in available_act:
                # Temperatura interna sopra la soglia di comfort
                if indoor_temp > cooling_setpoint + self.comfort_band:
                    # Alta generazione solare -> energia quasi gratuita, raffredda di piu'
                    if solar_gen > 0.6:
                        if outdoor_temp < cooling_setpoint:
                            action[available_act.index('cooling_device')] = 0.6
                        else:
                            action[available_act.index('cooling_device')] = 0.9
                    # Media generazione solare
                    elif 0.2 <= solar_gen <= 0.6:
                        if outdoor_temp < cooling_setpoint:
                            action[available_act.index('cooling_device')] = 0.4
                        else:
                            action[available_act.index('cooling_device')] = 0.7
                    # Bassa generazione solare -> risparmia energia
                    else:
                        if outdoor_temp < cooling_setpoint:
                            action[available_act.index('cooling_device')] = 0.25
                        else:
                            action[available_act.index('cooling_device')] = 0.55

                # Temperatura interna nella zona di comfort
                elif indoor_temp >= cooling_setpoint - self.comfort_band:
                    # Outdoor alta + solare disponibile -> azione preventiva moderata
                    # per evitare che la temp esca dal comfort nelle prossime ore
                    if outdoor_temp > cooling_setpoint + 5:
                        if solar_gen > 0.4:
                            action[available_act.index('cooling_device')] = 0.3
                        else:
                            action[available_act.index('cooling_device')] = 0.15
                    # Outdoor nella fascia media -> leggero mantenimento solo se c'e' solare
                    elif outdoor_temp > cooling_setpoint:
                        if solar_gen > 0.4:
                            action[available_act.index('cooling_device')] = 0.15
                        else:
                            action[available_act.index('cooling_device')] = 0.0
                    # Outdoor fresca -> nessun rischio termico imminente, spegni
                    else:
                        action[available_act.index('cooling_device')] = 0.0
                # Temperatura interna sotto il setpoint -> spegni
                else:
                    action[available_act.index('cooling_device')] = 0.0

            # ------------------------------------------------------------------
            # ELECTRICAL STORAGE
            # Features usate (LIME top-2): solar_generation, elec_price
            # Logica: carica quando c'e' solare (energia "gratis") o quando il
            # prezzo e' basso; scarica durante le ore di punta o quando il
            # prezzo e' alto e non c'e' solare disponibile.
            # ------------------------------------------------------------------
            if 'electrical_storage' in available_act:
                # Ore di punta -> scarica per ridurre il prelievo dalla rete
                if 8 <= hour <= 10 or 18 <= hour <= 21:
                    if solar_gen > 0.6:
                        # Solare alto: il fotovoltaico copre parte del carico, scarica meno
                        if elec_price > 0.03:
                            action[available_act.index('electrical_storage')] = -0.5
                        else:
                            action[available_act.index('electrical_storage')] = -0.35
                    elif 0.3 <= solar_gen <= 0.6:
                        if elec_price > 0.03:
                            action[available_act.index('electrical_storage')] = -0.65
                        else:
                            action[available_act.index('electrical_storage')] = -0.5
                    else:
                        # Solare basso: scarica di piu' per coprire il carico dalla batteria
                        if elec_price > 0.03:
                            action[available_act.index('electrical_storage')] = -0.8
                        else:
                            action[available_act.index('electrical_storage')] = -0.6
                # Fuori ore di punta -> carica sfruttando solare o prezzi bassi
                else:
                    if solar_gen > 0.6:
                        if elec_price <= 0.03:
                            action[available_act.index('electrical_storage')] = 0.9
                        else:
                            action[available_act.index('electrical_storage')] = 0.7
                    elif 0.3 <= solar_gen <= 0.6:
                        if elec_price <= 0.03:
                            action[available_act.index('electrical_storage')] = 0.6
                        else:
                            action[available_act.index('electrical_storage')] = 0.4
                    else:
                        # Solare basso: carica solo se il prezzo e' conveniente
                        if elec_price <= 0.03:
                            action[available_act.index('electrical_storage')] = 0.35
                        else:
                            action[available_act.index('electrical_storage')] = 0.1

            # ------------------------------------------------------------------
            # DHW STORAGE
            # Features usate (LIME top-2): solar_generation, dhw_demand
            # Logica: carica il boiler quando c'e' solare disponibile (costo
            # energetico basso); scarica / non caricare quando c'e' alta domanda
            # di acqua calda (il boiler serve il carico direttamente).
            # La fascia oraria viene mantenuta come struttura di base per
            # decidere quando il consumo e' probabile.
            # ------------------------------------------------------------------
            if 'dhw_storage' in available_act:
                # Serbatoio quasi pieno (soc >= 0.7)
                if dhw_storage_soc >= 0.7:
                    # Ore di utilizzo tipico acqua calda (mattina/sera)
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        # Alta domanda -> scarica per soddisfare il carico
                        if dhw_demand >= 0.6:
                            action[available_act.index('dhw_storage')] = -0.6
                        # Bassa domanda -> scarica poco
                        elif dhw_demand < 0.3:
                            action[available_act.index('dhw_storage')] = -0.2
                        else:
                            action[available_act.index('dhw_storage')] = -0.35
                    # Ore solari (10-16) -> mantieni o ricarica se disponibile
                    elif 10 <= hour <= 16:
                        if solar_gen > 0.6:
                            # Serbatoio gia' pieno + solare alto -> mantieni
                            action[available_act.index('dhw_storage')] = 0.15
                        elif 0.2 <= solar_gen <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.0
                        else:
                            action[available_act.index('dhw_storage')] = 0.0
                    # Ore notturne -> serbatoio pieno, nessuna azione
                    else:
                        action[available_act.index('dhw_storage')] = 0.0

                # Serbatoio a meta' (0.3 <= soc < 0.7)
                elif 0.3 <= dhw_storage_soc < 0.7:
                    # Ore di utilizzo tipico acqua calda
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        # Alta domanda -> scarica moderatamente
                        if dhw_demand >= 0.6:
                            action[available_act.index('dhw_storage')] = -0.35
                        # Bassa domanda -> scarica poco
                        elif dhw_demand < 0.3:
                            action[available_act.index('dhw_storage')] = -0.1
                        else:
                            action[available_act.index('dhw_storage')] = -0.2
                    # Ore solari -> carica sfruttando il solare
                    elif 10 <= hour <= 16:
                        if solar_gen > 0.6:
                            action[available_act.index('dhw_storage')] = 0.65
                        elif 0.2 <= solar_gen <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.4
                        else:
                            # Solare basso ma serbatoio a meta' -> carica moderatamente
                            action[available_act.index('dhw_storage')] = 0.2
                    # Ore notturne -> carica un po' per prepararsi alla mattina
                    else:
                        action[available_act.index('dhw_storage')] = 0.3

                # Serbatoio quasi vuoto (soc < 0.3) -> priorita' alla ricarica
                else:
                    # Ore di utilizzo tipico acqua calda -> scarica minima obbligatoria
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        if dhw_demand >= 0.6:
                            action[available_act.index('dhw_storage')] = -0.05
                        else:
                            action[available_act.index('dhw_storage')] = 0.0
                    # Ore solari -> carica aggressivamente con il solare
                    elif 10 <= hour <= 16:
                        if solar_gen > 0.6:
                            action[available_act.index('dhw_storage')] = 0.85
                        elif 0.2 <= solar_gen <= 0.6:
                            action[available_act.index('dhw_storage')] = 0.65
                        else:
                            action[available_act.index('dhw_storage')] = 0.45
                    # Ore notturne -> carica per evitare di rimanere senza
                    else:
                        action[available_act.index('dhw_storage')] = 0.6

            # Debugging log for actions values and observations values -> Can be modified as needed
            # Actions value logging should be implemented in a more structured way for UserRBC
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

    # KPI logging: weekly and daily -> To implement in UserRBC
    weekly_interval_steps = 7 * 24  # 168 steps
    daily_interval_steps = 1 * 24   # 24 steps
    kpi_dir = 'kpi_logs'
    # Log files for weekly and daily KPIs are saved as .txt files
    # Change from .csv to .txt for better readability has been taken into consideration for UserRBC
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
    env_1 = CityLearnEnv(schema=schema, central_agent=True)  # LimeGuidedRBC (new_rbc)
    env_2 = CityLearnEnv(schema=schema, central_agent=True)  # AdvancedRBC (main_rbc)
    env_3 = CityLearnEnv(schema=schema, central_agent=True)  # OptimizedRBC (baseline)

    # LimeGuidedRBC: controller di questo file
    lime_agent = LimeGuidedRBC(env_1)

    # AdvancedRBC: controller importato da main_rbc
    main_agent = MainAdvancedRBC(env_2)

    # OptimizedRBC: baseline CityLearn
    baseline_agent = OptimizedRBC(env_3)

    run_simulation(lime_agent, env_1)
    run_simulation(main_agent, env_2)
    run_simulation(baseline_agent, env_3)

    # Compare results: LimeGuidedRBC vs AdvancedRBC (main_rbc) vs OptimizedRBC
    plot_district_kpis(
        {
            'LimeGuidedRBC': env_1,
            'AdvancedRBC':   env_2,
            'OptimizedRBC':  env_3,
        },
        base_path='imgs'
    )

if __name__ == '__main__':
    # Configurations
    conf = Config()
    args = conf.args

    main(args)