from __future__ import annotations
from typing import Dict, Tuple, Callable
from optuna.trial import Trial
import logging

import time
import os
import subprocess
import pandas as pd

from hyperparameter_tuner.loss import Loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_objective(
    slurm_script: str,
    results_path: str,
    loss: Loss,
    trial_param_types: Dict[str, Tuple[str, Tuple, Dict]],
) -> Callable[[Trial], float]:
    """
    Create an objective function for Optuna to optimize, which submits SLURM jobs and waits for results.

    Args:
        slurm_script: str - Path to the SLURM script to execute.
        results_path: str - Path to the CSV file where results will be logged.
        loss: Loss - An instance of a Loss class that implements a `calculate` method.
        trial_param_types: Dict[str, Tuple[str, Tuple, Dict]]: Dictionary mapping parameter names to their types
            and arguments. Each entry is structured as:
            - key str: The name of the parameter.
            - value: Tuple[str, Tuple, Dict]
                - str - The parameter type ('int', 'float', or 'categorical').
                - Tuple - Positional arguments for the parameter's sampling method.
                - Dict - Keyword arguments for the parameter's sampling method.
    """

    def objective(trial: Trial) -> float:
        """
        Objective function to be passed to Optuna for trial evaluation.

        Args:
            trial (Trial): A trial object from Optuna, used to suggest parameter values.

        Returns:
            float: The calculated loss based on the results from the CSV file.

        Raises:
            ValueError: If an invalid parameter type is provided.
            subprocess.CalledProcessError: If the SLURM job submission fails.
        """
        param_methods = {'int': trial.suggest_int, 'float': trial.suggest_float, 'categorial': trial.suggest_categorical}

        trial_id = trial.number
        trial_params = {}
        for param_name, (arg_type, args, kwargs) in trial_param_types.items():
            if arg_type in param_methods:
                trial_params[param_name] = param_methods[arg_type](param_name, *args, **kwargs)
            else:
                logger.error(f'Invalid parameter type: {arg_type}')
                raise

        command = f'sbatch {slurm_script} {results_path} {trial_id} {" ".join(str(v) for v in trial_params.values())}'

        try:
            subprocess.run(command, shell=True, check=True)
            logger.info('SLURM job submitted with trial ID: {trial_id}')
            logger.info('Paramters: {trial_params}')
        except subprocess.CalledProcessError:
            logger.error('Error submitting SLURM job')
            raise

        while not os.path.isfile(results_path):
            time.sleep(5)

        while True:
            df = pd.read_csv(results_path)

            matching_row = df.loc[df['trial'] == trial_id]
            if matching_row.empty:
                time.sleep(5)
                continue

            row_data = matching_row.iloc[0]
            return loss.calculate(row_data)

    return objective
