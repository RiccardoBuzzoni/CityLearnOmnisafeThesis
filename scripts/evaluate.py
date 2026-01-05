'''
This is a function that evaluates the performance of a custom rule-based controller
(AdvancedRBC) in a CityLearn environment. It runs a simulation episode, collects
key performance indicators (KPIs), and plots the results for analysis.
''' 

# Imports

import os
import json
import argparse
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# CityLearn imports
from citylearn.citylearn import CityLearnEnv
from main_rbc import AdvancedRBC # RBC agent to be evaluated
from utils import *

# Plotting battery history
def plot_battery_h(history: Dict[str, List[float]], ax1: Axes):
    assert 'soc' in history.keys(), 'Missing state of charge information in battery history.'
    assert 'discharge' in history.keys(), 'Missing charge rate information in battery history.'
    ax1.set_axisbelow(True)
    ax1.grid(visible=True, linestyle='dashed')

    # Charge rate
    ax1.bar(range(len(history['discharge'])), history['discharge'], color = 'xkcd:soft blue')
    ax1.set_ylabel('(Dis)Charge (kW/h)')
    ax1.yaxis.label.set_color('xkcd:soft blue')

    ax2 = ax1.twinx()

    # State of charge
    ax2.plot(history['soc'], c='xkcd:orange')
    ax2.set_ylabel('SoC (%)')
    ax2.set_ylim(ymin=-0.05, ymax=1.05)
    ax2.yaxis.label.set_color('xkcd:orange')

# Evaluation function
def evaluate_rbc(args, schema:dict, building_idx: int = 0):
    env = CityLearnEnv(schema = schema, cental_agent = True)
    agent = AdvancedRBC(env)

    results = {}

    # Simulate episode
    observations, _ = env.reset()
    while not env.terminated:
        actions = agent.predict(observations)
        observations, _, _, _, _ = env.step(actions)

    # Extract KPIs for building of interest
    kpis = get_kpis(env)