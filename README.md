# Hyperparameter Tuning with SLURM and Optuna

This package integrates Optuna with SLURM to efficiently run hyperparameter optimization jobs on a cluster. It supports customizable trial parameters and allows you to track the results via a CSV file. The SLURM jobs process the parameter configurations and log their results, which are then read by Optuna to calculate the objective loss.
## Features

- Submit SLURM jobs with trial parameters generated by Optuna.
- Customizable loss functions that process results stored in a CSV file.
- Support for multiple parameter types (integers, floats, and categorical values).
- Flexible interface for handling different experiment configurations.
- Logs all SLURM submissions and errors using Python's logging package.


## Installation

Install with `poetry` or `pip`.

```bash
poetry add git+https://github.com/C4theBomb/hyperparameter-tuner.git
pip install git+https://github.com/C4theBomb/hyperparameter-tuner.git
```

## Usage

### 1. Define your loss function

```python
from hyperparameter_tuner import Loss

class CustomLossFunction(Loss):
  def calculate(self, row: pd.Series) -> float:
    return row[0]
```

### 2. Create your objective function

```python
from hyperparameter_tuner import create_objective

# Define your parameter types { name: (type, args, kwargs) }
trial_param_types = {
    'trajectories': ('int', (1, 5000), {}),
    'learning_rate': ('float', (1e-5, 1e-2), {'log': True}),
    'optimizer': ('categorical', (['adam', 'sgd'],), {})
}

loss = MyLoss()
slurm_script = 'path/to/slurm_script.sh'
results_path = 'path/to/results.csv'

objective = create_objective(slurm_script, results_path, loss, trial_param_types)
```

### 3. Write your SLURM script

```sh
#!/bin/bash
RESULTS_PATH=$1 # This is always required to be the first parameter
TRIAL_ID=$2 # This is required to be the second parameter

# Extract your trial parameters
TRAJECTORIES=$3
LEARNING_RATE=$4
OPTIMIZER=$5

REWARDS=$((RANDOM % 100))  # Replace with your experiment logic

# Write results to CSV (first column must be TRIAL_ID so that Optuna can identify the run).
echo "$TRIAL_ID,$REWARDS,$TRAJECTORIES,$LEARNING_RATE,$OPTIMIZER" >> $RESULTS_PATH
```

### 4. Run your script
```
python main.py
```
## License

[AGPL-3.0](https://choosealicense.com/licenses/agpl-3.0/)


## Authors

- [Ceferino Patino](https://www.github.com/C4theBomb)

