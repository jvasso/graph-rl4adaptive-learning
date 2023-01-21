import PATHS
import utils_files

from hyperparams import Hyperparams
from expParams import ExpParams
from extraParams import ExtraParams
from config import Config

import os
from itertools import product
import copy

measure_time = 0

class ConfigManager:

    def __init__(self, hyperparams_ids_list:list=None, exp_params_ids_list:list=None, easy_init=False, super_easy_init=False):
        
        if super_easy_init:
            assert (hyperparams_ids_list is not None) and (exp_params_ids_list is not None)
            self.super_easy_init(hyperparams_ids_list, exp_params_ids_list)
        elif easy_init:
            assert (hyperparams_ids_list is not None)
            self.easy_init(hyperparams_ids_list)
        else:
            self.hyperparams_ids_list = None
            raw_hyperparams, raw_exp_params, raw_visualization_params, raw_extra_params = ConfigManager.load_config()
            
            self.exp_params_list, self.seeds_list, self.corpus_list = ConfigManager.preprocess_exp_params(raw_exp_params)

            # search previous hyperparams
            dumped_dicts = utils_files.retrieve_json_dicts_in_dir(PATHS.IDS_HYPERPARAMS)

            self.hyperparams_list     = ConfigManager.preprocess_hyperparams(raw_hyperparams, dumped_dicts)
            self.extra_params         = ConfigManager.preprocess_extra_params(raw_extra_params)
            self.visualization_params = ConfigManager.preprocess_visualization_params(raw_visualization_params)
            self.config_done, self.config_to_do = self.classify_configurations()
            
            self.hyperparams_ids_list = [hyperparams.get_id() for hyperparams in self.hyperparams_list]
    

    def super_easy_init(self, hyperparams_ids_list, exp_params_ids_list):
        self.exp_params_ids_list  = exp_params_ids_list
        self.hyperparams_ids_list = hyperparams_ids_list

        _, __, raw_visualization_params, raw_extra_params = ConfigManager.load_config(no_hyperparams=True)

        self.exp_params_list, seeds_list, corpus_list = ConfigManager.load_exp_params(self.exp_params_ids_list)
        self.seeds_list  = sorted(seeds_list)
        self.corpus_list = sorted(corpus_list)
        self.hyperparams_list     = ConfigManager.load_hyperparams(self.hyperparams_ids_list)
        self.extra_params         = ConfigManager.preprocess_extra_params(raw_extra_params)
        self.visualization_params = ConfigManager.preprocess_visualization_params(raw_visualization_params)

    
    def easy_init(self, hyperparams_ids_list):
        self.hyperparams_ids_list = hyperparams_ids_list
        _, raw_exp_params, raw_visualization_params, raw_extra_params = ConfigManager.load_config()
        
        self.exp_params_list, self.seeds_list, self.corpus_list = ConfigManager.preprocess_exp_params(raw_exp_params)
        self.hyperparams_list     = ConfigManager.load_hyperparams(self.hyperparams_ids_list)
        self.extra_params         = ConfigManager.preprocess_extra_params(raw_extra_params)
        self.visualization_params = ConfigManager.preprocess_visualization_params(raw_visualization_params)
    

    @staticmethod
    def load_exp_params(exp_params_ids_list):
        exp_params_list, seeds_list, corpus_list = [], [], []
        exp_params_ids_dict = ConfigManager.build_exp_params_ids_dict()
        for env_name, env_dict in exp_params_ids_dict.items():
            if env_name != ".DS_Store":
                for trainer_name, trainer_dict in env_dict.items():
                    if trainer_name != ".DS_Store":
                        for corpus_name, corpus_dict in trainer_dict.items():
                            if corpus_name != ".DS_Store":
                                for seed, seed_dict in corpus_dict.items():
                                    if seed != ".DS_Store":
                                        for exp_param_id in seed_dict:
                                            exp_param_id_no_ext = exp_param_id.split(".json")[0]
                                            if exp_param_id_no_ext in exp_params_ids_list :
                                                structure_alone = env_name+"/"+trainer_name+"/"+corpus_name+"/"+seed+"/"+exp_param_id
                                                exp_params_folder_path = PATHS.IDS_EXP_PARAMS+"/"+structure_alone
                                                exp_params_dict = utils_files.load_json_file(exp_params_folder_path)
                                                exp_params_object = ExpParams(copy.deepcopy(exp_params_dict), id=exp_param_id_no_ext, structure_alone=structure_alone, easy_init=True)
                                                exp_params_list.append(exp_params_object)
                                                seed_int = int(seed.split("seed")[1])
                                                if seed_int not in seeds_list: seeds_list.append(seed_int)
                                                if corpus_name not in corpus_list: corpus_list.append(corpus_name)
        return exp_params_list, seeds_list, corpus_list

    
    @staticmethod
    def build_exp_params_ids_dict():
        exp_params_ids_path = PATHS.IDS_EXP_PARAMS
        exp_params_ids_dict = utils_files.get_tree_structure(exp_params_ids_path)
        return exp_params_ids_dict


    @staticmethod
    def load_hyperparams(hyperparams_ids_list):
        hyperparams_list = []
        hyperparams_ids_path = PATHS.IDS_HYPERPARAMS
        for hyperparams_id in hyperparams_ids_list:
            hyperparams_file_path = hyperparams_ids_path+"/"+hyperparams_id +".json"
            hyperparams_dict = utils_files.load_json_file(hyperparams_file_path)
            hyperparams = Hyperparams(hyperparams_dict, id=hyperparams_id, easy=True)
            hyperparams_list.append(hyperparams)
        return hyperparams_list


    @staticmethod
    def preprocess_exp_params(raw_exp_params):
        seeds_list  = ConfigManager.generate_seeds(mode=raw_exp_params["seed"])
        corpus_list = raw_exp_params["corpus"]
        raw_exp_params["seed"] = seeds_list
        exp_params_listified_leaves = ConfigManager.listify_leaves(raw_exp_params)
        exp_params_list_of_dicts    = ConfigManager.DL2LD(exp_params_listified_leaves)
        exp_params_list_of_objects  = [ ExpParams(copy.deepcopy(exp_params_dict)) for exp_params_dict in exp_params_list_of_dicts ]
        return exp_params_list_of_objects, seeds_list, corpus_list
    

    @staticmethod
    def preprocess_hyperparams(raw_hyperparams, dumped_dicts):
        hyperparams_listified_leaves = ConfigManager.listify_leaves(raw_hyperparams)
        hyperparams_list_of_dicts   = ConfigManager.DL2LD(hyperparams_listified_leaves)

        hyperparams_list_of_objects = []
        hyperparams_ids_list        = []
        for hyperparams_dict in hyperparams_list_of_dicts:
            hyperparam_obj = Hyperparams(copy.deepcopy(hyperparams_dict), dumped_dicts)
            hyperparam_id = hyperparam_obj.get_id()
            if hyperparam_id not in hyperparams_ids_list:
                hyperparams_ids_list.append(hyperparam_id)
                hyperparams_list_of_objects.append(hyperparam_obj)
            if hyperparam_obj.is_new:
                dumped_dicts[hyperparam_id+".json"] = utils_files.dump_dict(hyperparam_obj.hyperparams_dict)
        
        return hyperparams_list_of_objects
    
    @staticmethod
    def remove_duplicates(list_of_objects):
        ids_list = []
        duplicates = []
        final_list_of_objects = []
        for obj in list_of_objects:
            obj_id = obj.get_id()
            if obj_id not in ids_list:
                ids_list.append(obj_id)
                final_list_of_objects.append(obj)
            else:
                duplicates.append(obj_id)
        return final_list_of_objects, duplicates
    

    @staticmethod
    def preprocess_extra_params(raw_extra_params):
        extraParams = ExtraParams(copy.deepcopy(raw_extra_params))
        return extraParams
    

    @staticmethod
    def preprocess_visualization_params(raw_visualization_params):
        return raw_visualization_params

    
    def classify_configurations(self):
        counter = 0
        config_done  = []
        config_to_do = []
        for exp_params in self.exp_params_list:
            for hyperparams in self.hyperparams_list:
                counter+=1
                dynamic_params = self.build_dynamic_params(exp_params, hyperparams)
                config_dict = {
                    "hyperparams":hyperparams,
                    "exp_params":exp_params,
                    "dynamic_params":dynamic_params,
                    "extra_params":self.extra_params,
                    "visualization_params":self.visualization_params
                }
                config = Config(config_dict)                
                if config.get_config_results() is None:
                    config_to_do.append(config)
                else:
                    config_done.append(config)
        return config_done, config_to_do
    

    def build_dynamic_params(self, exp_params:ExpParams, hyperparams:Hyperparams):
        dynamic_params = {}

        num_students = exp_params.get_nb_student()
        corpus_size  = exp_params.get_corpus_size()
        eval_frequency = self.extra_params.get_eval_frequency()
        trainer_params = hyperparams.get_trainer_params()
        
        # max epoch
        #assert num_students%eval_frequency==0
        dynamic_params["max_epoch"] = num_students//eval_frequency

        # step per episode
        chosen_option = exp_params.get_max_steps_per_episode()
        if chosen_option == "standard":
            dynamic_params["step_per_episode"] = corpus_size
        elif type(chosen_option)==int:
            dynamic_params["step_per_episode"] = chosen_option
        else:
            raise Exception("Option '"+str(chosen_option)+"' for max_steps_per_episode is not supported.")
        
        # step per epoch (DO NOT CHANGE ORDER: KEEP THIS AFTER step_per_episode AND BEFORE episode_per_collect)
        dynamic_params["step_per_epoch"] = eval_frequency*dynamic_params["step_per_episode"]

        # step per collect
        if "episode_per_collect" not in trainer_params.keys():
            step_per_collect_ratio = hyperparams.get_step_per_collect()
            step_per_collect = int(step_per_collect_ratio*dynamic_params["step_per_epoch"])
            dynamic_params["step_per_collect"] = step_per_collect
        
        return dynamic_params


    @staticmethod
    def generate_seeds(mode):
        if mode   == "1seed_mode1":
            return [0]
        elif mode == "3seeds_mode1":
            return list(range(3))
        elif mode == "5seeds_mode1":
            return list(range(5))
        elif mode == "6seeds_mode1":
            return list(range(6))
        elif mode == "10seeds_mode1": # ± 1.5 reward error
            return list(range(10))
        elif mode == "25seeds_mode1": # ± 1 reward error
            return list(range(25))
        elif mode == "50seeds_mode1": # ± 0.5 reward error
            return list(range(50))
        else:
            raise Exception("Mode "+mode+" not implemented.")

    @staticmethod
    def load_config(no_hyperparams=False):
        hyperparams_path          = PATHS.CONFIG_HYPERPARAMS
        exp_params_path           = PATHS.CONFIG_EXP_PARAMS
        visualization_params_path = PATHS.CONFIG_VISUALIZATION_PARAMS
        extra_params_path         = PATHS.CONFIG_EXTRA_PARAMS
        hyperparams          = None if no_hyperparams else ConfigManager.get_hyperparams(hyperparams_path)
        exp_params           = utils_files.load_yaml_file(exp_params_path)
        visualization_params = utils_files.load_yaml_file(visualization_params_path)
        extra_params         = utils_files.load_yaml_file(extra_params_path)
        return hyperparams, exp_params, visualization_params, extra_params

    @staticmethod
    def get_hyperparams(dir):
        hyperparams = {}
        _, sub_dirs, files = next(os.walk(dir))
        for file in files:
            file_no_ext, ext = file.split(".")
            if ext == "yaml":
                hyperparams[file_no_ext] = utils_files.load_yaml_file(dir+"/"+file)
        for sub_dir in sub_dirs:
            hyperparams[sub_dir] = ConfigManager.get_hyperparams(dir+"/"+sub_dir)
        return hyperparams

    @staticmethod
    def listify_leaves(tree: dict) -> dict:
        for key, value in tree.items():
            if isinstance(value, dict):
                # Recursively listify the leaves of the subtree
                tree[key] = ConfigManager.listify_leaves(value)
            elif not isinstance(value, list):
                # Convert the leave into a list
                tree[key] = [value]
        return tree
    
    
    @staticmethod
    def DL2LD(tree: dict):
        for key, value in tree.items():
            if type(value) == dict:
                tree[key] = ConfigManager.DL2LD(tree[key])
        keys = tree.keys()
        values = tree.values()
        combinations = list(product(*values))
        return [dict(zip(keys, combination)) for combination in combinations]
    

    @staticmethod
    def LD2DL(dicts):
        merged = {}
        for d in dicts:
            for key, value in d.items():
                if key in merged and isinstance(value, dict):
                    merged[key] = ConfigManager.LD2DL([merged[key], value])
                elif key in merged and isinstance(value, (list, tuple, set)):
                    if isinstance(merged[key], (list, tuple, set)):
                        merged[key] = list(set(merged[key] + value))
                    else:
                        merged[key] = [merged[key]] + value
                elif key in merged and not isinstance(value, (list, tuple, set)):
                    if isinstance(merged[key], (list, tuple, set)):
                        merged[key].append(value)
                    else:
                        merged[key] = [merged[key], value]
                else:
                    merged[key] = value
        return merged

    

    def print_report(self):
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Config Manager report:\n")
        print("• "+str(len(self.hyperparams_list)) + " hyperparameters")
        print("• "+str(len(self.corpus_list))+" corpus")
        print("• "+str(len(self.seeds_list))+" seeds")
        print("• "+str(len(self.config_to_do))+"/"+str(len(self.config_to_do)+len(self.config_done))+ " configs to test")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    

    def find_num_student(self):
        num_students_list = [ exp_params.get_nb_student() for exp_params in self.exp_params_list ]
        assert len(set(num_students_list)) == 1
        return num_students_list[0]
    
    def find_corpus_sizes(self):
        corpus_sizes = {corpus:[] for corpus in self.corpus_list}
        for exp_params in self.exp_params_list:
            corpus      = exp_params.get_corpus_name()
            corpus_size = exp_params.get_corpus_size()
            corpus_sizes[corpus].append(corpus_size)
        for corpus, sizes_list in corpus_sizes.items():
            assert len(set(sizes_list)) == 1
            corpus_sizes[corpus] = sizes_list[0]
        return corpus_sizes
    

    def get_config_lists(self):
        return self.config_done, self.config_to_do
    
    def get_config_to_do(self):
        return self.config_to_do
    
    def get_exp_params_list(self):
        return self.exp_params_list
    
    def get_nb_seeds(self):
        return len(self.seeds_list)
    
    def get_seeds_list(self):
        return self.seeds_list
    
    def get_corpus_list(self):
        return self.corpus_list

    def get_hyperparams_ids_list(self):
        assert self.hyperparams_ids_list is not None
        return self.hyperparams_ids_list
    
    def get_extra_params(self):
        return self.extra_params