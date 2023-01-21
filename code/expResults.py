import PATHS

from expParams import ExpParams
from extraParams import ExtraParams
from config import Config
from hyperparams import Hyperparams

import utils_files
import utils_dict

import copy


class ExpResults:
    LAST_TEST_REWARD  = "last_test_reward"
    LAST_TRAIN_REWARD = "last_train_reward"
    ALL_REWARDS       = "all_reward"
    
    def __init__(self, expParams:ExpParams, extraParams:ExtraParams, hyperparams_ids_list:list=None):

        self.expParams   = expParams
        self.extraParams = extraParams
        
        self.exp_params_path = PATHS.RESULTS + "/" + self.expParams.get_id()
        
        if hyperparams_ids_list is None:
            hyperparams_ids_list = ExpResults.find_hyperparams_ids_in_dir(self.exp_params_path)
        self.hyperparams_ids_dict = ExpResults.load_hyperparams(hyperparams_ids_list)
        
        assert set(self.hyperparams_ids_dict.keys()) == set(hyperparams_ids_list)


    @staticmethod
    def find_hyperparams_ids_in_dir(exp_params_path):
        hyperparams_ids_list = []
        subfolders_names_list = utils_files.get_subfolders_names(exp_params_path)
        hyperparams_ids_list = [ subfolder_name for subfolder_name in subfolders_names_list if "hyperparams" in subfolder_name]
        return hyperparams_ids_list
        

    @staticmethod
    def load_hyperparams(hyperparams_ids_list):
        hyperparams_ids_dict = {}
        hyperparams_ids_path = PATHS.IDS_HYPERPARAMS
        for hyperparams_id in hyperparams_ids_list:
            hyperparams_file_path = hyperparams_ids_path+"/"+hyperparams_id +".json"
            hyperparams_dict = utils_files.load_json_file(hyperparams_file_path)
            hyperparams = Hyperparams(hyperparams_dict, id=hyperparams_id, easy=True)
            hyperparams_ids_dict[hyperparams_id]=hyperparams
        return hyperparams_ids_dict

    
    def get_results_according_to_criteria(self, criteria:str, no_reject:bool=False):
        hyperparams_results_dict = {}
        hyperparams_dates_dict = {}
        for hyperparams_id in self.hyperparams_ids_dict:
            results, date_with_ext = self.retrieve_result(hyperparams_id, criteria)
            if results is not None:
                hyperparams_results_dict[hyperparams_id] = results
                hyperparams_dates_dict[hyperparams_id] = date_with_ext
            elif no_reject:
                raise Exception("Error: couldn't retrieve results of hyperparameter id '"+hyperparams_id+"'.")
        return hyperparams_results_dict, hyperparams_dates_dict
    
    
    def retrieve_result(self, hyperparams_id:str, criteria:str):
        # conditions that results must verify
        train_interval_ep = self.extraParams.get_logger_train_interval_ep()
        test_interval_ep  = self.extraParams.get_logger_test_interval_ep()
        true_nb_eps       = self.expParams.get_nb_student()
        eval_frequency    = self.extraParams.get_eval_frequency()
        
        hyperparams_path =  self.exp_params_path+"/"+hyperparams_id
        hyperparams_results = Config.build_results_dict(hyperparams_path, with_extra=True)

        for date_with_ext, date_results_dict in hyperparams_results.items():
            ignore_test=True
            complete, infos = Config.check_complete_results(date_results_dict, train_interval_ep, test_interval_ep, true_nb_eps, eval_frequency, ignore_test=ignore_test)
            if complete:
                if criteria == ExpResults.LAST_TEST_REWARD:
                    result = ExpResults.search_last_test_reward(date_results_dict, true_nb_eps)
                elif criteria == ExpResults.LAST_TRAIN_REWARD:
                    result = ExpResults.search_last_train_reward(date_results_dict, true_nb_eps)
                elif criteria == ExpResults.ALL_REWARDS:
                    result = ExpResults.search_all_rewards(date_results_dict, true_nb_eps)
                else:
                    raise Exception("Error: Criteria "+criteria+" not supported.")
                assert result is not None
                return result, date_with_ext
        print("No results were found for hyperparameter "+hyperparams_id)
        return None
    
    @staticmethod
    def search_last_test_reward(date_results_dict, true_nb_eps):
        path = "test/"+str(true_nb_eps)
        data_dict = utils_dict.get_value_at_path(path, date_results_dict)
        result = data_dict["rew"]
        return result
    
    @staticmethod
    def search_last_train_reward(date_results_dict, true_nb_eps):
        path = "train/"+str(true_nb_eps)
        data_dict = utils_dict.get_value_at_path(path, date_results_dict)
        result = data_dict["rew"]
        return result
    
    def search_all_rewards(date_results_dict, true_nb_eps):
        result = date_results_dict
        for result_type, result_type_dict in date_results_dict.items():
            for epoch in result_type_dict:
                result_type_dict[epoch] = result_type_dict[epoch]["rew"]
        return result
    
    def get_hyperparams_ids_dict(self):
        return copy.deepcopy(self.hyperparams_ids_dict)
    
    def get_corpus_id(self):
        return self.expParams.get_corpus_name()

    def get_seed_id(self):
        return "seed"+str(self.expParams.get_seed())
    
    def get_exp_params_id(self):
        return self.expParams.get_id()
    
    def get_exp_params(self):
        return self.expParams
    
    def get_exp_params_structure(self):
        return self.expParams.get_structure_path()
