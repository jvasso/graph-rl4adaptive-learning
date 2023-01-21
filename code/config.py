import os
import copy

import GLOBAL_VAR
import PATHS

import utils_dict
import utils_files

from hyperparams import Hyperparams
from expParams import ExpParams
from extraParams import ExtraParams


class Config:

    def __init__(self, config_dict):
        """
        config_dict = {
                        "hyperparams":copy.deepcopy(hyperparams),
                        "exp_params":copy.deepcopy(exp_params),
                        "dynamic_params":dynamic_params,
                        "extra_params":copy.deepcopy(self.extra_params),
                        "visualization_params":copy.deepcopy(self.visualization_params)
                    }
        """
        # do not change order
        self.config_dict = config_dict

        self.hyperparams: Hyperparams = config_dict["hyperparams"]
        self.expParams  : ExpParams   = config_dict["exp_params"]
        self.extraParams: ExtraParams = config_dict["extra_params"]
        self.dynamic_params: dict     = config_dict["dynamic_params"]

        self.env_params     = self.build_env_params()
        self.trainer_params = self.build_trainer_params()
        self.results_folder_path = self.build_results_folder_path()
        self.config_results = self.search_existing_results()
        
        if self.config_results is None:
            self.create_results_folder()
        self.unique_results_folder_path = None
    
    
    def create_results_folder(self):
        if GLOBAL_VAR.SAVE:
            utils_files.create_folder(self.results_folder_path)


    def create_log_file_path(self):
        if GLOBAL_VAR.SAVE:
            self.unique_date_id = utils_files.create_unique_date_id()
            log_id = "log" + self.unique_date_id
            self.log_file_path = self.results_folder_path + "/" + log_id + ".pkl"
            return self.log_file_path

    def build_env_params(self):
        env_params = self.expParams.get_env_params()
        env_params["step_per_episode"] = self.dynamic_params["step_per_episode"]
        env_params["feedback_mode"]    = self.hyperparams.get_feedback_mode()
        env_params["time_mode"]        = self.hyperparams.get_time_mode()
        return env_params
    
    def build_trainer_params(self):
        trainer_params_hyperparams  = copy.deepcopy(self.hyperparams.get_trainer_params())
        trainer_params_exp_params   = self.expParams.get_trainer_params()
        trainer_params_extra_params = self.extraParams.get_trainer_params()
        merged_dicts = utils_dict.merge_disjointed_dicts(trainer_params_hyperparams, trainer_params_exp_params, trainer_params_extra_params)
        
        merged_dicts["max_epoch"]      = self.dynamic_params["max_epoch"]
        merged_dicts["step_per_epoch"] = self.dynamic_params["step_per_epoch"]
        return merged_dicts

    def search_existing_results(self, verbose=False):
        if not GLOBAL_VAR.LOAD: return None
        if verbose: self.display_config_infos()
        if not os.path.isdir(self.results_folder_path):
            if verbose: self.display_did_not_found_results_folder_path(error_msg="doesn't exist.")
            return None
        multiple_results_dict = Config.build_results_dict(self.results_folder_path)
        if len(multiple_results_dict)==0:
            if verbose: self.display_did_not_found_results_folder_path(error_msg="is empty.")
        train_interval_ep = self.extraParams.get_logger_train_interval_ep()
        test_interval_ep  = self.extraParams.get_logger_test_interval_ep()
        true_nb_eps       = self.expParams.get_nb_student()
        eval_frequency    = self.extraParams.get_eval_frequency()
        for date, results_dict in multiple_results_dict.items():
            ignore_test=True
            complete, infos = Config.check_complete_results(results_dict, train_interval_ep, test_interval_ep, true_nb_eps, eval_frequency, ignore_test=ignore_test)
            if complete:
                self.unique_results_folder_path = self.results_folder_path+"/"+date
                if verbose: self.display_found_existing_results()
                return results_dict
            else:
                if verbose: self.display_not_matching(self.results_folder_path+"/"+date, infos)
        return None


    def display_config_infos(self, extra_space=True, max_infos=False):
        if extra_space: print("")
        print("* Configuration: "+str(self.results_folder_path))
        print("      • exp_params : "+self.expParams.get_id())
        print("            - corpus : "+self.expParams.get_corpus_name()+" (size: "+str(self.expParams.get_corpus_size())+")") if max_infos else 0
        print("            - seed   : "+str(self.expParams.get_seed())) if max_infos else 0
        print("      • hyperparams: "+self.hyperparams.get_id())
        print("            - policy             : "+str(self.hyperparams.get_policy_name())) if max_infos else 0
        print("            - lr                 : "+str(self.hyperparams.get_learning_rate())) if max_infos else 0
        print("            - middle_dim         : "+str(self.hyperparams.get_middle_dim())) if max_infos else 0
        print("            - doc_encoding       : "+str(self.hyperparams.get_doc_encoding())) if max_infos else 0
        print("            - batch_size         : "+str(self.hyperparams.get_batch_size())) if max_infos else 0
        print("            - repeat_per_collect : "+str(self.hyperparams.get_repeat_per_collect())) if max_infos else 0
        print("            - time_mode          : "+str(self.hyperparams.get_time_mode())) if max_infos else 0
        print("            - feedback_mode      : "+str(self.hyperparams.get_feedback_mode())) if max_infos else 0
        print("            - gnn_arch           : "+str(self.hyperparams.get_gnn_arch_name())) if max_infos else 0
        print("            - feedback_arch      : "+str(self.hyperparams.get_feedback_arch_name())) if max_infos else 0
        print("            - time_arch          : "+str(self.hyperparams.get_time_arch_name())) if max_infos else 0
        print("            - features params    : "+str(self.hyperparams.get_features_params())) if max_infos else 0

    def display_found_existing_results(self):
        print("  ==> Found existing results at:"+self.unique_results_folder_path)
    def display_did_not_found_results_folder_path(self, error_msg):
        print("  --> No results were found for this config: path '"+self.results_folder_path+"' "+error_msg)
    def display_not_matching(self, date_folder_path, error_msg):
        print("    • Results folder: "+date_folder_path)
        print("      -> "+error_msg)

    @staticmethod
    def check_complete_results(results_dict:dict, train_interval_ep:int, test_interval_ep:int, true_nb_eps:int, eval_frequency:int, ignore_test=False):
        if not( type(results_dict)==dict and set(results_dict.keys())=={"train","test"} ):
            return False, "the folder does not contain train and test."
        if not Config.check_train_or_test_results(results_dict["train"], train_interval_ep, true_nb_eps, eval_frequency=1, is_train=True):
            return False, "train results don't match logger requirements."
        if not ignore_test:
            if not Config.check_train_or_test_results(results_dict["test"], test_interval_ep, true_nb_eps, eval_frequency=eval_frequency, is_train=False):
                return False, "test results don't match logger requirements."
        return True, "ok"
    

    @staticmethod
    def check_train_or_test_results(results_dict:dict, interval_ep, true_nb_eps, eval_frequency, is_train):
        expected_eps = Config.build_expected_eps(interval_ep, true_nb_eps, is_train=is_train)
        if not Config.check_include_expected_eps(results_dict, expected_eps):
            return False
        return True
    
    
    @staticmethod
    def build_expected_eps(interval_ep, true_nb_eps, is_train=True):
        start = interval_ep
        end   = true_nb_eps+interval_ep
        freq  = interval_ep
        expected_eps = {str(i) for i in range(start, end, freq)}
        return expected_eps


    @staticmethod
    def check_include_expected_eps(results_dict, expected_eps):
        return set(results_dict.keys()) == expected_eps


    def build_results_folder_path(self):
        if not (GLOBAL_VAR.LOAD or GLOBAL_VAR.SAVE):
            return None
        results_path = PATHS.RESULTS
        path_structure_list = PATHS.RESULTS_STRUCTURE
        path_list = [ self.config_dict[key].get_id() for key in path_structure_list ]
        structure_path = "/".join(path_list)
        results_folder_path = results_path +"/"+ structure_path
        return results_folder_path


    @staticmethod
    def build_results_dict(directory, with_extra=False):
        directory = PATHS.EXTRA_PATH+directory if with_extra else directory
        results = {}
        for filename in os.listdir(directory):
            if filename.endswith(GLOBAL_VAR.RESULTS_FILES_EXT):
                if GLOBAL_VAR.RESULTS_FILES_EXT==".pkl":
                    results[filename] = utils_files.load_pickle_file(os.path.join(directory, filename), with_extra=False)
        return results
    
    def get_config_results(self):
        return self.config_results
    
    def get_seed(self):
        return self.expParams.get_seed()
    
    def get_corpus_name(self):
        return self.expParams.get_corpus_name()
    
    def get_corpus_size(self):
        return self.expParams.get_corpus_size()
    
    def get_nb_students(self):
        return self.expParams.get_nb_student()
    
    def get_env_params(self):
        return self.env_params
    
    def get_node_features(self):
        return self.hyperparams.get_node_features()
    
    def get_gnn_agent_params(self):
        return self.hyperparams.get_gnn_agent_params()
    
    def get_learning_rate(self):
        return self.hyperparams.get_learning_rate()
    
    def get_policy_params(self):
        return self.hyperparams.get_policy_params()
    
    def get_policy_name(self):
        return self.hyperparams.get_policy_name()
    
    def get_VectorReplayBuffer_size(self):
        return self.hyperparams.get_VectorReplayBuffer_size()
    
    def get_trainer_params(self):
        return self.trainer_params
    
    def get_logger_params(self):
        return self.extraParams.get_logger_params()
    
    def get_step_per_episode(self):
        return self.dynamic_params["step_per_episode"]