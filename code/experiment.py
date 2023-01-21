from config import Config
import RL_functions

import time

class Experiment:
    """
    1 hyperparams set
    1 corpus
    1 seed
    """
    def __init__(self, config:Config):
        self.config = config
    
    def start_experiment(self):
        start_time = time.time()
        
        RL_functions.run_train_test(self.config)
        
        end_time = time.time()
        running_time = end_time-start_time
        infos = {"running_time":running_time}
        return infos