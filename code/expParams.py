import PATHS
import utils_files
import utils_dict

import copy

class ExpParams:

    def __init__(self, raw_exp_params_dict, id=None, structure_alone=None, easy_init=False):
        
        if easy_init:
            definitive_exp_params_dict = raw_exp_params_dict
            self.easy_init(definitive_exp_params_dict, id, structure_alone)
        else:
            # do not change the order
            self.exp_params_folder_path, self.structure_alone = ExpParams.build_path_to_folder(raw_exp_params_dict)
            self.exp_params_dict = ExpParams.preprocess_params(copy.deepcopy(raw_exp_params_dict))
            self.id, self.already_exists = self.assign_id()

        self.num_students = self.exp_params_dict["env"]["num_students"]
        self.corpus_size  = ExpParams.find_corpus_size(corpus_name = self.exp_params_dict["corpus"])
    
    
    def easy_init(self, definitive_exp_params_dict, id, structure_alone):
        self.exp_params_dict = definitive_exp_params_dict
        self.id = id
        self.already_exists = True
        self.structure_alone = structure_alone
        self.exp_params_folder_path = PATHS.IDS_EXP_PARAMS + "/" + self.structure_alone
    
    
    @staticmethod
    def find_corpus_size(corpus_name):
        corpus_data_path = PATHS.CORPUS_DATA
        file_path = corpus_data_path + "/" + corpus_name + "/corpus.json"
        corpus_dict = utils_files.load_json_file(file_path, no_extra=True)
        corpus_size = len(corpus_dict)
        return corpus_size
    

    @staticmethod
    def preprocess_params(raw_exp_params):
        exp_params_connector = PATHS.EXP_PARAMS_BANK_CONNECTOR
        for input_dict_path, output_os_path in exp_params_connector.items():
            utils_dict.connect_dict_to_file(input_dict_path, output_os_path, raw_exp_params)
        return raw_exp_params
    
    
    def assign_id(self):
        found_file, already_exists = self.check_existence()
        if found_file is None:
            current_date = utils_files.create_unique_date_id()
            id = "exp_params"+current_date
            self.save_params_dict(id)
        else:
            id = found_file.split(".json")[0]
        return id, already_exists
    
    def save_params_dict(self, id):
        file_path = self.exp_params_folder_path + "/" + id + ".json"
        utils_files.save_json_dict_to_path(file_path, self.exp_params_dict)

    
    def check_existence(self):
        already_exists = False
        found_file = utils_files.search_json_in_dir(self.exp_params_folder_path)
        if found_file is not None:
            already_exists = True
            found_dict = utils_files.load_json_file(self.exp_params_folder_path+"/"+found_file)
            if found_dict != self.exp_params_dict:
                raise Exception("Found 2 tests with the same params but different dictionaries.")
        return found_file, already_exists
    
    
    @staticmethod
    def build_path_to_folder(exp_params):
        path_structure_list = PATHS.IDS_EXP_PARAMS_STRUCTURE
        path_list = []
        for key in path_structure_list:
            if key=="seed":
                path_list.append(key+str(exp_params[key]))
            else:
                path_list.append(str(exp_params[key]))
        structure_path = "/".join(path_list)

        ids_exp_params_path  = PATHS.IDS_EXP_PARAMS
        exp_params_folder_path = ids_exp_params_path +"/"+ structure_path

        return exp_params_folder_path, structure_path
    
    def get_id(self):
        return self.id

    def get_nb_student(self):
        return self.num_students
    
    def get_corpus_size(self):
        return self.corpus_size
    
    def get_trainer_params(self):
        return self.exp_params_dict["trainer"]
    
    def get_seed(self):
        return self.exp_params_dict["seed"]
    
    def get_corpus_name(self):
        return self.exp_params_dict["corpus"]
    
    def get_env_params(self):
        return self.exp_params_dict["env"]
    
    def get_structure_path(self):
        return self.structure_alone
        
    def get_max_steps_per_episode(self):
        return self.exp_params_dict["env"]["max_steps_per_episode"]
    
    def get_load_model(self):
        return self.exp_params_dict["load_params"]
    
    def does_already_exist(self):
        return self.already_exists
    
    def pretty_print(self):
        utils_dict.print_dict_pretty(self.exp_params_dict)