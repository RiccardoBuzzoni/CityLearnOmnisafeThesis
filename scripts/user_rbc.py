import sys
import os
import math
import textwrap
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Tuple
from CityLearn.citylearn.citylearn import CityLearnEnv
from CityLearn.citylearn.agents.base import Agent
from scripts.main_rbc import AdvancedRBC
from utils import Config, default_env_config, select_env_config, get_kpis, plot_district_kpis


class UserRBC(Agent):
    """Interactive user rule-based controller.

    schedules: Dict[int, Dict[str, Union[float, Dict[int, float]]]]
        Mapping building_index -> action_base -> scalar or hour->value map.
    """

    def __init__(self, env: CityLearnEnv, schedules: Dict[int, Dict[str, Any]]):
        super().__init__(env)
        self.schedules = schedules

        # Build a flat mapping from action index -> (building_index, action_key)
        # For central agents action_names is a single flattened list.
        self.action_index_map: List[Tuple[int, str]] = []
        if self.env.unwrapped.central_agent:
            # action_names is a list with one element that is the flattened action list
            flat = self.env.unwrapped.action_names[0]
            # We reconstruct by taking each building's active_actions in order
            idx = 0
            for b_idx, b in enumerate(self.env.unwrapped.buildings):
                for act in b.active_actions:
                    # action name in flat will appear in same order; assume identical string
                    self.action_index_map.append((b_idx, act))
                    idx += 1
        else:
            # Non-central: per-building sublists
            for b_idx, acts in enumerate(self.env.unwrapped.action_names):
                for act in acts:
                    self.action_index_map.append((b_idx, act))

    def _get_value(self, building_idx: int, action_key: str, hour: int) -> float:
        # Determine base action type
        for base in ('electrical_storage', 'dhw_storage', 'cooling_device'):
            if base in action_key:
                bmap = self.schedules.get(building_idx, {})
                if base in bmap:
                    v = bmap[base]
                    if isinstance(v, dict):
                        return float(v.get(hour, 0.0))
                    else:
                        return float(v)
                else:
                    return 0.0
        # If action is not one of the supported, return 0.0
        return 0.0

    def predict(self, observations: List[List[float]], deterministic: bool = True) -> List[List[float]]:
        actions: List[List[float]] = []

        for agent_idx, (act_names, obs) in enumerate(zip(self.action_names, observations)):
            hour = int(round(obs[self.observation_names[agent_idx].index('hour')]))
            # Normalize hour into 1-24 candidates like HourRBC does
            hour_candidates = [hour, hour % 24, ((hour - 1) % 24) + 1]
            action_vals: List[float] = []

            # Determine starting offset in action_index_map for this agent
            # For central agents we assume a single agent and direct mapping
            if self.env.unwrapped.central_agent:
                # agent 0 maps to full action_index_map, which aligns to act_names order
                for pos, act_key in enumerate(act_names):
                    building_idx, full_key = self.action_index_map[pos]
                    # pick the first valid hour candidate
                    chosen_hour = None
                    for cand in hour_candidates:
                        chosen_hour = cand
                        break
                    val = self._get_value(building_idx, full_key, chosen_hour)
                    action_vals.append(val)

            else:
                # decentralized: act_names corresponds to consecutive entries for building
                # find a slice in action_index_map matching this building (match by first occurrence)
                # fallback: use mapping by order
                base_pos = 0
                # find a starting pos where building indices match
                for i, (bidx, _) in enumerate(self.action_index_map):
                    if bidx == agent_idx:
                        base_pos = i
                        break
                for j, act_key in enumerate(act_names):
                    building_idx, full_key = self.action_index_map[base_pos + j]
                    val = self._get_value(building_idx, full_key, hour_candidates[0])
                    action_vals.append(val)

            actions.append(action_vals)

        self.actions = actions
        self.next_time_step()
        return actions


def _prompt_choice(prompt: str, choices: List[str], allow_multi: bool = False) -> List[int]:
    print(prompt)
    for i, c in enumerate(choices):
        print(f'  {i+1}. {c}')
    resp = input('Enter selection (numbers, comma separated): ').strip()
    picks: List[int] = []
    for token in resp.split(','):
        token = token.strip()
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(choices):
                picks.append(idx)
    return picks


def interactive_menu(schema: Dict[str, Any]):
    # Create environment to inspect buildings/actions
    env = CityLearnEnv(schema=schema, central_agent=True)
    buildings = env.unwrapped.buildings

    print('\nAvailable buildings:')
    for i, b in enumerate(buildings):
        print(f'  {i+1}. {b.name}  (actions: {b.active_actions})')

    sel = input('\nSelect buildings to configure (comma list, or "all"): ').strip()
    if sel.lower() == 'all' or sel == '':
        selected = list(range(len(buildings)))
    else:
        selected = []
        for tok in sel.split(','):
            tok = tok.strip()
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(buildings):
                    selected.append(idx)

    # Build schedules per building
    schedules: Dict[int, Dict[str, Any]] = {}

    for bidx in selected:
        print(f'\nConfiguring Building {bidx+1}: {buildings[bidx].name}')
        schedules[bidx] = {}
        for base in ('electrical_storage', 'dhw_storage', 'cooling_device'):
            use = input(f'  Configure "{base}" for this building? (y/N): ').strip().lower()
            if use != 'y':
                continue
            mode = input('    Set (1) constant OR (2) per-hour map? [1/2]: ').strip()
            if mode == '2':
                print('    Enter 24 comma-separated numeric values for hours 1..24')
                vals_raw = input('    values: ').strip()
                parts = [p.strip() for p in vals_raw.split(',') if p.strip()]
                if len(parts) != 24:
                    print('    Invalid count — expected 24 values. Skipping this action.')
                    continue
                hour_map = {i+1: float(parts[i]) for i in range(24)}
                schedules[bidx][base] = hour_map
            else:
                v = input('    Enter constant value (e.g., 0.5 or -0.2): ').strip()
                try:
                    schedules[bidx][base] = float(v)
                except Exception:
                    print('    Invalid number — skipping')

    print('\nConfiguration complete. Summary:')
    for bidx in selected:
        print(f'  Building {bidx+1} ({buildings[bidx].name}): {schedules.get(bidx, {})}')

    return schema, schedules


def run_simulation(agent: Agent, env: CityLearnEnv, agent_name: str):
    print(f'Running simulation for {agent_name}...')
    observations, _ = env.reset()
    max_steps = env.time_steps - 1

    weekly_interval_steps = 7 * 24
    daily_interval_steps = 1 * 24
    os.makedirs('kpi_logs', exist_ok=True)
    weekly_file = os.path.join('kpi_logs', f'{agent_name}_weekly_kpis_log.txt')
    daily_file = os.path.join('kpi_logs', f'{agent_name}_daily_kpis_log.txt')

    def _write_kpis(step_idx: int, filepath: str):
        try:
            kpis = get_kpis(env)
            kpis = kpis[kpis['level'] == 'district']
            day = step_idx // 24
            with open(filepath, 'a') as fh:
                fh.write(f'--- STEP: {step_idx} (day {day}) ---\n')
                for _, row in kpis.iterrows():
                    fh.write(f"{row['kpi']}: {row['value']}\n")
                fh.write('\n')
        except Exception as e:
            with open(filepath, 'a') as fh:
                fh.write(f'Failed to write KPIs at step {step_idx}: {e}\n\n')

    _write_kpis(0, daily_file)
    _write_kpis(0, weekly_file)

    step = 0
    for _ in range(max_steps):
        actions = agent.predict(observations)
        observations, reward, terminated, truncated, info = env.step(actions)
        step += 1
        if step % daily_interval_steps == 0:
            _write_kpis(step, daily_file)
        if step % weekly_interval_steps == 0:
            _write_kpis(step, weekly_file)

    print(f'Finished {agent_name} simulation.')


def main():
    conf = Config()
    args = conf.args

    # Get schema
    if args.custom:
        schema = select_env_config(args.data)
    else:
        schema = default_env_config(args.data)

    schema, schedules = interactive_menu(schema)

    # Create envs
    env_user = CityLearnEnv(schema=schema, central_agent=True)
    env_adv = CityLearnEnv(schema=schema, central_agent=True)

    # Instantiate agents
    user_agent = UserRBC(env_user, schedules=schedules)

    # Use the same AdvancedRBC implementation from scripts/main_rbc.py
    adv_agent = AdvancedRBC(env_adv)

    # Run both simulations
    run_simulation(user_agent, env_user, 'UserRBC')
    run_simulation(adv_agent, env_adv, 'AdvancedRBC')

    # Plot district KPIs (saves to imgs/) and show
    plot_district_kpis({'UserRBC': env_user, 'AdvancedRBC': env_adv}, base_path='imgs')
    img_path = os.path.join('imgs', 'district_kpis.png')
    # Show the generated plot
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
