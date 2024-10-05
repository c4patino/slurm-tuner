from __future__ import annotations

import os
import time
import subprocess
import pandas as pd

import optuna
import neptune
import neptune.integrations.optuna as neptune_optuna

API_KEY = os.environ.get('NEPTUNE_API_KEY')
PROJECT = 'c4thebomb/testing'

run = neptune.init_run(project=PROJECT, api_token=API_KEY)

params = {'direction': 'maximize', 'n_trials': 15}
run['parameters'] = params

results_path = 'results.csv'


def objective(trial) -> float:
    trial_id = trial.number
    trial_params = {
        'trajectories': trial.suggest_int('trajectories', 1, 5000),
    }

    command = f'sbatch test.submit {trial_id}'
    for _, value in trial_params.items():
        command += f' {value}'

    subprocess.run(command, shell=True, check=True)

    while True:
        df = pd.read_csv(results_path)

        row = matching_row = df.loc[df['trial'] == trial_id]
        if matching_row.empty:
            time.sleep(1)
            continue

        return row['cumulative_rewards'].values[0]


neptune_callback = neptune_optuna.NeptuneCallback(run)

study = optuna.create_study(direction=params["direction"])
study.optimize(objective, n_trials=params["n_trials"], callbacks=[neptune_callback])

run.stop()
