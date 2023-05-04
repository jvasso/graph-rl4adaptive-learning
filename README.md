## Graph RL for Adaptive Learning

This is the official implementation of the model in the paper: "Towards scalable adaptive learning with graph neural networks and reinforcement learning".


### INSTALLATION

The requirements can be installed with the following steps (using Anaconda):
```
conda create --name gnn_rl_env
conda activate gnn_rl_env
bash install.sh
```

### USAGE

#### Run experiment

To run the experiments of the paper, go to the "/code" directory and type the following command in command line:
```
python main_experiment.py
```
This will train the model (from scratch) on each corpus with multiple seeds and save the results in multiple log files.
There are as many trainings as there are configurations.
The number of configs is always equal to: number of hyperparameter configurations $\times$ number of corpora $\times$ number of seeds
If you do not change anything in the "/config" directory, this number should be equal to 300 (2*6*25)
You can also set `seed: "1seed_mode1"` in "config/exp_params.yaml" to reduce the number of seeds and get quicker results.
During RL training the logs are saved in the "/results" folder.

#### Visualize the results

To visualize the results, please type the two following commands:
```
python main_gridsearch.py
python main_visualize.py
```
The first one will generate a report that contains the results for each corpus.
The second one will use this report to plot the results.
It will plot the averaged learning curves and the dictionary of hyperparameters.

### HYPERPARAMETER TUNING

There are 4 types of parameters in the "/config" directory:
- "hyperparams" which determines the model
- "experiment_params" which determines the modalities of the experiment: 2 experiments with different experiment_params are not comparable.
- the "extra_params" which does not determine the model nor the results of the experiment (example: save frequency for the logs)
- the "visualization_params" (useless)

All these parameters are grouped in the "/config" folder.
A config is the combination of: a set of hyperparams, experiment_params and extra_params (and visualization_params, but this is not important).

To test multiple parameter values (e.g. hyperparameter values) just put a list of these values in the corresponding entry of the associated yaml file. This will automatically create all possible combinations of configs.
Each time a new set of hyperparams or experiment_params is created, it is saved in the "/ids" folder (the code checks beforehand that it does not already exist, so that the same experiment is never conducted twice).
