import GLOBAL_SETTING
from configManager import ConfigManager

import time
from datetime import timedelta
import functools

def run_experiment(config, global_var_dict):
    import GLOBAL_VAR
    GLOBAL_VAR.CONFIGURE_ALL(global_var_dict)
    config.display_config_infos(extra_space=False, max_infos=True)
    from experiment import Experiment
    experiment = Experiment(config)
    exp_infos = experiment.start_experiment()
    return exp_infos

# mandatory (to be able to visualize results)
save = True
load = True

global_var_dict = GLOBAL_SETTING.BUILD_GLOBAL_VAR_DICT(save, load)

configManager = ConfigManager()
configManager.print_report()
configs_to_do = configManager.get_config_to_do()

run_experiment_partial = functools.partial(run_experiment, global_var_dict=global_var_dict)

infos = {"running_times":[]}
total_start = time.time()
for idx in range(len(configs_to_do)) :
    config = configs_to_do[idx]
    start_time = time.time()
    print("\nCONFIG "+str(idx+1)+"/"+str(len(configs_to_do))+"\n")
    exp_infos = run_experiment_partial(config)
    infos["running_times"].append(exp_infos["running_time"])
    end_time = time.time()
    print("\nTook "+str(end_time-start_time)+" seconds.")
total_end = time.time()
total_time = total_end - total_start
total_time_converted = timedelta(seconds=total_time)
print(len(configs_to_do), "configurations trained in", total_time_converted)
