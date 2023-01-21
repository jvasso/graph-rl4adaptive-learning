import PATHS
import utils_dict


class ExtraParams:

    def __init__(self, raw_extra_params):
        
        self.extra_params_dict = ExtraParams.preprocess_params(raw_extra_params)
    

    @staticmethod
    def preprocess_params(raw_extra_params):
        # connect trainer and logger
        extra_params_connector = PATHS.EXTRA_PARAMS_BANK_CONNECTOR
        for input_dict_path, output_os_path in extra_params_connector.items():
            utils_dict.connect_dict_to_file(input_dict_path, output_os_path, raw_extra_params)
        return raw_extra_params
    
    
    def get_trainer_params(self):
        return self.extra_params_dict["trainer"]
    
    def get_logger_params(self):
        return self.extra_params_dict["logger"]
    
    def get_logger_train_interval_ep(self):
        return self.extra_params_dict["logger"]["train_interval_ep"]
    
    def get_logger_test_interval_ep(self):
        return self.extra_params_dict["logger"]["test_interval_ep"]
    
    def get_eval_frequency(self):
        return self.extra_params_dict["eval_frequency"]