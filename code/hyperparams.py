import PATHS
import GLOBAL_VAR

import utils_dict
import utils_files

class Hyperparams:

    def __init__(self, hyperparams_dict: dict=None, dumped_dicts: dict=None, id: str=None, easy=False, from_id=False):
        if easy:
            assert (id is not None) and (hyperparams_dict is not None)
            self.easy_init(hyperparams_dict, id)
        elif from_id:
            assert id is not None
            self.init_from_id(id)
        else:
            assert id is None
            self.hard_init(hyperparams_dict, dumped_dicts)
        
        self.middle_dim         = utils_dict.get_value_at_path(PATHS.MIDDLE_DIM, self.hyperparams_dict)
        self.doc_encoding       = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_DOC_ENCODING, self.hyperparams_dict)
        self.batch_size         = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_BATCH_SIZE, self.hyperparams_dict)
        self.repeat_per_collect = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_REPEAT_PER_COLLECT, self.hyperparams_dict)
        self.step_per_collect   = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_STEP_PER_COLLECT, self.hyperparams_dict)
        self.time_mode          = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TIME_MODE, self.hyperparams_dict)

    def easy_init(self, definitive_hyperparams_dict: dict, id: str):
        self.hyperparams_dict = definitive_hyperparams_dict
        self.infos_dict = "not provided (easy init)"
        self.id = id
        assert self.check_all_clear()
    

    def init_from_id(self, id):
        hyperparams_ids_paths = PATHS.IDS_HYPERPARAMS
        file_path = hyperparams_ids_paths+"/"+id+".json"
        self.hyperparams_dict = utils_files.load_json_file(file_path)
        self.infos_dict = "not provided (init_from_id)"
        self.id = id

    def hard_init(self, raw_hyperparams_dict:dict, dumped_dicts:dict):
        assert dumped_dicts is not None
        self.hyperparams_dict, self.infos_dict = Hyperparams.preprocess_hyperparams(raw_hyperparams_dict)
        self.id, self.is_new = Hyperparams.assign_id(self.hyperparams_dict, dumped_dicts)
        assert self.check_all_clear()
    
    
    @staticmethod
    def assign_id(hyperparams_dict, dumped_dicts):
        filename = utils_files.search_dict_in_dumps(hyperparams_dict, dumped_dicts)
        if filename is None:
            current_date = utils_files.create_unique_date_id()
            id = "hyperparams"+current_date
            Hyperparams.save_hyperparams(id, hyperparams_dict)
            new = True
        else:
            id = filename.split(".json")[0]
            new = False
        return id, new
    

    @staticmethod
    def save_hyperparams(id, hyperparams_dict):
        file_path = PATHS.IDS_HYPERPARAMS + "/"+ id + ".json"
        utils_files.save_json_dict_to_path(file_path, hyperparams_dict)
    

    @staticmethod
    def preprocess_hyperparams(hyperparams_dict):
        assert Hyperparams.check_leaves(hyperparams_dict)

        infos_dict = {}
        infos_dict["gnn_arch_name"]      = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_GNN_ARCH, hyperparams_dict)
        infos_dict["feedback_arch_name"] = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_FEEDBACK_ARCH, hyperparams_dict)
        infos_dict["time_arch_name"]     = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TIME_ARCH, hyperparams_dict)
        infos_dict["features_params"]    = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_WORD_FEATURES, hyperparams_dict)
        
        # connect architectures
        arch_connector = PATHS.ARCH_CONNECTOR
        for input_dict_path, output_os_path in arch_connector.items():
            utils_dict.connect_dict_to_file(input_dict_path, output_os_path, hyperparams_dict)
        
        # connect nodes features
        word_features_connector = PATHS.CONFIG_HYPERPARAMS_WORD_FEATURES_CONNECTOR
        for input_dict_path, output_os_path in word_features_connector.items():
            utils_dict.connect_dict_to_file(input_dict_path, output_os_path, hyperparams_dict)
        
        # connect trainer
        policy_name_path = PATHS.CONFIG_HYPERPARAMS_POLICY_NAME
        policy_name = utils_dict.get_value_at_path(policy_name_path, hyperparams_dict)
        trainer_name = GLOBAL_VAR.POLICY2TRAINER_NAME[policy_name]
        trainer_params = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TRAINER_PARAMS +"/"+trainer_name, hyperparams_dict)
        utils_dict.set_value_at_path(PATHS.CONFIG_HYPERPARAMS_TRAINER_PARAMS, trainer_params, hyperparams_dict)
        
        ## remove pointless layers
        # 1) search for chosen layers
        gnn_arch      = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_GNN_ARCH, hyperparams_dict)
        feedback_arch = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_FEEDBACK_ARCH, hyperparams_dict)
        time_arch     = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TIME_ARCH, hyperparams_dict)
        arch_dict = {"gnn_arch":gnn_arch, "feedback_arch":feedback_arch, "time_arch":time_arch}
        used_layers_dict = {"gnn_arch":[], "feedback_arch":[], "time_arch":[]}
        layer_types_dict = {"gnn_arch":[], "feedback_arch":[], "time_arch":[]}
        for arch_name, arch in arch_dict.items():
            if arch is not None:
                for layer_dict in arch["arch"]:
                    layer_cls  = layer_dict["layer_cls"]
                    layer_type = layer_dict["layer_type"]
                    used_layers_dict[arch_name].append(layer_cls)
                    layer_types_dict[arch_name].append(layer_type)
        
        # if doc_encoding is "one-hot-encoding"
        doc_encoding_path = PATHS.CONFIG_HYPERPARAMS_DOC_ENCODING
        if utils_dict.get_value_at_path(doc_encoding_path, hyperparams_dict) == "one-hot-encoding":

            # if no "doc2doc" ==> add one
            if "doc2doc" not in layer_types_dict["gnn_arch"]:
                hyperparams_dict["gnn_agent"]["general"]["gnn_arch"]["arch"].insert(0, {"layer_type": "doc2doc", "layer_cls": "Linear"})
            
            # if MessagePassing ==> replace with GeneralConv
            layers_list = hyperparams_dict["gnn_agent"]["general"]["gnn_arch"]["arch"]
            hyperparams_dict["gnn_agent"]["general"]["gnn_arch"]["arch"] = [ {"layer_type": "kw2doc", "layer_cls": "GeneralConv"} if layer=={"layer_type":"kw2doc","layer_cls": "MessagePassing"} else layer for layer in layers_list ]
            used_layers_dict["gnn_arch"] = ["GeneralConv" if layer_cls=="MessagePassing" else layer_cls for layer_cls in used_layers_dict["gnn_arch"]]
        
        # if there is no time in gnn arch
        if "merge_t" not in layer_types_dict["gnn_arch"]:
            policy_name_path = PATHS.CONFIG_HYPERPARAMS_POLICY_NAME
            value = "PGPolicy"
            utils_dict.set_value_at_path(policy_name_path, value, hyperparams_dict)
            
            time_mode_path = PATHS.CONFIG_HYPERPARAMS_TIME_MODE
            value = "no"
            utils_dict.set_value_at_path(time_mode_path, value, hyperparams_dict)

            time_arch_path = PATHS.CONFIG_HYPERPARAMS_TIME_ARCH
            value = "none"
            utils_dict.set_value_at_path(time_arch_path, value, hyperparams_dict)
            infos_dict["time_arch_name"] = "none"
        
        # it time_mode == "no"
        time_mode = hyperparams_dict["time"]["time_mode"]
        if time_mode == "no":
            if utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TIME_ARCH,hyperparams_dict) != "none":
                raise Exception("Error: time_mode is 'no' but there still is a time architecture")
        
        # 2) remove layer classes not chosen in arch
        selected_layers = used_layers_dict["gnn_arch"] + used_layers_dict["feedback_arch"] + used_layers_dict["time_arch"]
        final_layers_dict = {}
        layers_dict = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_LAYERS, hyperparams_dict)
        for layer_name, layer_params in layers_dict.items():
            if layer_name in selected_layers:
                final_layers_dict[layer_name] = layer_params
        utils_dict.set_value_at_path(PATHS.CONFIG_HYPERPARAMS_LAYERS, final_layers_dict, hyperparams_dict)
        
        # adjust hyperparams that cancel each other
        if ("GeneralConv" in final_layers_dict):
            if final_layers_dict["GeneralConv"]["attention"] == False:
                final_layers_dict["GeneralConv"]["heads"]=1
                final_layers_dict["GeneralConv"]["attention_type"]="additive"
        
        # connect policy
        policy_params_connector = PATHS.CONFIG_HYPERPARAMS_POLICY_PARAMS_CONNECTOR
        for input_dict_path, output_dict_path in policy_params_connector.items():
            utils_dict.connect_dict_internally(input_dict_path, output_dict_path, hyperparams_dict)
        
        return hyperparams_dict, infos_dict
    

    def check_all_clear(self):
        features_params = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_WORD_FEATURES, self.hyperparams_dict)
        gnn_arch        = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_GNN_ARCH, self.hyperparams_dict)
        feedback_arch   = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_FEEDBACK_ARCH, self.hyperparams_dict)
        time_arch       = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TIME_ARCH, self.hyperparams_dict)
        policy_params   = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_POLICIES_PARAMS, self.hyperparams_dict)
        trainer_params  = utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_TRAINER_PARAMS, self.hyperparams_dict)
        for param in [features_params, gnn_arch, feedback_arch, time_arch, policy_params, trainer_params]:
            if type(param) != dict and param!= "none":
                return False
        return True
        
    

    @staticmethod
    def get_id_from_filename(filename):
        id_str = filename.split("hyperparams")[1].split(".")[0]
        id_int = int(id_str)
        return id_int
    

    @staticmethod
    def check_leaves(d):
        if not isinstance(d, dict):
            return False
        if not d:
            return True
        for k, v in d.items():
            if isinstance(v, list):
                return False
            elif isinstance(v, dict):
                return Hyperparams.check_leaves(v)
        return True
    

    def get_dict(self):
        return self.hyperparams_dict

    def get_id(self):
        return self.id
    
    def get_features_params(self):
        return self.infos_dict["features_params"]
    
    def get_gnn_arch_name(self):
        return self.infos_dict["gnn_arch_name"]
    
    def get_feedback_arch_name(self):
        return self.infos_dict["feedback_arch_name"]
    
    def get_time_arch_name(self):
        return self.infos_dict["time_arch_name"]
    
    def get_trainer_params(self):
        return self.hyperparams_dict["rl_algo"]["trainer_params"]
    
    def get_node_features(self):
        return self.hyperparams_dict["node_features"]
    
    def get_gnn_agent_params(self):
        return self.hyperparams_dict["gnn_agent"]
    
    def get_policy_params(self):
        return self.hyperparams_dict["rl_algo"]["policy_params"]
    
    def get_policy_name(self):
        return self.hyperparams_dict["rl_algo"]["general"]["policy"]
    
    def get_learning_rate(self):
        return self.hyperparams_dict["rl_algo"]["general"]["lr"]
    
    def get_feedback_mode(self):
        return self.hyperparams_dict["feedbacks"]["feedback_mode"]
    
    def get_time_mode(self):
        return self.hyperparams_dict["time"]["time_mode"]
    
    def get_middle_dim(self):
        return self.middle_dim
    
    def get_doc_encoding(self):
        return self.doc_encoding
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_repeat_per_collect(self):
        return self.repeat_per_collect
    
    def get_step_per_collect(self):
        return self.step_per_collect

    def get_VectorReplayBuffer_size(self):
        return self.hyperparams_dict["rl_algo"]["VectorReplayBuffer"]["total_size"]
    
    def print_pretty(self):
        utils_dict.print_dict_pretty(self.hyperparams_dict)