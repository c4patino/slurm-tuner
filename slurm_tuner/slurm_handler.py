"""SLURM job submission and result handling for Optuna optimization."""
from __future__ import annotations
from typing import Dict, Tuple, Callable
from optuna.trial import Trial
import logging

import time
import os
import re
import subprocess
import pandas as pd
import optuna

from slurm_tuner.loss import Loss

slurm_logger = logging.getLogger('slurm_tuner')


def create_objective(
    slurm_script: str,
    results_path: str,
    loss: Loss,
    trial_param_types: Dict[str, Tuple[str, Tuple, Dict]],
    return_average_on_prune: bool = False,
    log_trial_id_with_intermediate: bool = False,
) -> Callable[[Trial], float]:
    """
    Create an objective function for Optuna to optimize, which submits SLURM jobs and waits for results.

    Args:
        slurm_script: str - Path to the SLURM script to execute.
        results_path: str - Path to the CSV file where results will be logged.
        loss: Loss - An instance of a Loss class that implements a `calculate` method.
        trial_param_types: Dict[str, Tuple[str, Tuple, Dict]]: Dictionary mapping parameter names to their types and arguments.
            Each entry is structured as:
                - key str: The name of the parameter.
                - value: Tuple[str, Tuple, Dict[str, Any]]
                    - str - The parameter type ('int', 'float', or 'categorical').
                    - Tuple - Positional arguments for the parameter's sampling method.
                    - Dict[str, Any] - Keyword arguments for the parameter's sampling method.
        return_average_on_prune: bool - Whether to return the average loss of the all intermediate steps when a trial is pruned.
        log_trial_id_with_intermediate: bool - Whether to return the trial ID with the intermediate value instead of current step.
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
                slurm_logger.error(f'Invalid parameter type: {arg_type}')
                raise

        command = f'sbatch {slurm_script} {results_path} {trial_id} {" ".join(str(v) for v in trial_params.values())}'
        regex = r'Submitted batch job (\d+)\n'

        try:
            output = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            match = re.search(regex, output.stdout)
            job_id = int(match.group(1))

            slurm_logger.info(f'SLURM job {job_id} submitted with trial ID: {trial_id}')
            slurm_logger.info(f'Paramters: {trial_params}')
        except subprocess.CalledProcessError:
            slurm_logger.error('Error submitting SLURM job')
            raise

        while not os.path.isfile(results_path):
            time.sleep(5)

        current_step = 0
        while True:
            df = pd.read_csv(results_path)
            trial_data = df[df['trial'] == trial_id]

            step_data = trial_data[trial_data['step'] == current_step]
            termination_data = trial_data[trial_data['step'] == -1]

            # The run has neither terminated nor reached the next step, so continue waiting
            if step_data.empty and termination_data.empty:
                time.sleep(5)
                continue

            # We have reached the last step and should return the objective value
            if not termination_data.empty:
                row_data = termination_data.iloc[0]
                return loss(row_data, trial.params)

            # The next step has returned, determine if we should prune
            if not step_data.empty:
                row_data = step_data.iloc[0]
                intermediate_value = loss(row_data, trial.params)

                # Determine whether the designated pruners say that we should prune based off of intermediate values
                trial.report(intermediate_value, current_step if not log_trial_id_with_intermediate else trial_id)
                if trial.should_prune():
                    cancel_command = f'scancel {job_id}'
                    try:
                        # Cancel the slurm job associated with that task
                        subprocess.run(cancel_command, shell=True, check=True)
                        slurm_logger.info(f'SLURM job {job_id} cancelled due to pruning')
                    except subprocess.CalledProcessError:
                        slurm_logger.error('Error cancelling SLURM job, resources have to be release manually')
                    finally:
                        if return_average_on_prune:
                            # Return average intermediate values (useful for Wilcoxon signed rank test)
                            losses = step_data.apply(lambda row: loss(row, trial.params), axis=1)
                            return losses.mean()
                        else:
                            # Completely abort the trial
                            raise optuna.TrialPruned()

                current_step += 1

    return objective
