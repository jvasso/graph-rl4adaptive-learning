from resultsManager import ResultsManager
from expResults import ExpResults

import PATHS

from configManager import ConfigManager
from config import Config
from hyperparams import Hyperparams

import utils_files
import utils_dict

import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import copy
import json


class Visualizer:

    def __init__(self, hyperparams_ids_dict: dict, exp_params_ids_list: list, comparison_params=None,
                full_hyperparams_ids_list:list=None,
                hyperparam2mean_score:dict=None, hyperparam2std_score:dict=None,
                final=True):
        
        self.is_final = final
        self.hyperparams_ids_dict = hyperparams_ids_dict
        self.exp_params_ids_list  = exp_params_ids_list
        self.comparison_params    = ["features_params"] if final else comparison_params
        self.hyperparam2mean_score = hyperparam2mean_score
        self.hyperparam2std_score  = hyperparam2std_score
        self.sorted_hyperparams_ids_list = [ hyperparam_id for rank, hyperparam_id in sorted(self.hyperparams_ids_dict.items(), key=lambda item: item[0]) ]
        
        self.configManager  = ConfigManager(hyperparams_ids_list = self.sorted_hyperparams_ids_list,
                                            exp_params_ids_list  = self.exp_params_ids_list,
                                            super_easy_init      = True)
        self.resultsManager = ResultsManager(configManager=self.configManager)
        
        self.num_students = self.configManager.find_num_student()
        self.config_dict_str = self.config_dict_preprocessing(full_hyperparams_ids_list)
        self.corpus_names = self.configManager.get_corpus_list()
        self.corpus2results_dict = self.resultsManager.build_corpus2results_dict(epochs_select_criteria=ExpResults.ALL_REWARDS,
                                                                                seeds_merging_criteria_list=[ResultsManager.MEAN_REW, ResultsManager.STD_REW])


    def config_dict_preprocessing(self, full_hyperparams_ids_list):
        full_hyperparams_list = [ Hyperparams(id=hyperparams_id, from_id=True) for hyperparams_id in full_hyperparams_ids_list ]
        full_hyperparams_dicts_list = [ copy.deepcopy(hyperparams.get_dict()) for hyperparams in full_hyperparams_list ]
        
        for hyperparams_dict in full_hyperparams_dicts_list:
            for param in ["gnn_arch", "feedback_arch", "time_arch"]:
                path  = Visualizer.param2path(param)
                value = utils_dict.get_value_at_path(path, hyperparams_dict)
                if value is not None:
                    arch_list = value["arch"]
                    value = Visualizer.search_arch_name(param, arch_list, hyperparams_dict)
                    utils_dict.set_value_at_path(path, value, hyperparams_dict)
        
        config_dict = ConfigManager.LD2DL(full_hyperparams_dicts_list)
        utils_dict.remove_duplicates(config_dict)
        utils_dict.flatten_tree(config_dict)
        config_dict_stringify_lists = utils_dict.tree_dict_list2str(config_dict)
        config_dict_str = json.dumps(config_dict_stringify_lists, indent=4)
        config_dict_str = utils_dict.remove_spurious_patterns(config_dict_str)
        
        return config_dict_str

    
    def set_episodes_lists(self, episodes_choice:str, force_test_interval=False):
        extraParams = self.configManager.get_extra_params()
        train_interval_ep = extraParams.get_logger_train_interval_ep()
        test_interval_ep  = 5 if force_test_interval else extraParams.get_logger_test_interval_ep()

        expected_eps_train = Config.build_expected_eps(interval_ep=train_interval_ep, true_nb_eps=self.num_students, is_train=True)
        expected_eps_test  = Config.build_expected_eps(interval_ep=test_interval_ep , true_nb_eps=self.num_students, is_train=False)
        episodes_train = sorted([ int(x) for x in expected_eps_train ])
        episodes_test  = sorted([ int(x) for x in expected_eps_test ])
        if episodes_choice == "all":
            return episodes_train, episodes_test
        else:
            max_val = int(episodes_choice.split("/")[1])
            episodes_train = [ episode for episode in episodes_train if episode <= max_val]
            episodes_test  = [ episode for episode in episodes_test  if episode <= max_val]
            return episodes_train, episodes_test


    def get_corpus_sizes(self):
        corpus_sizes = self.configManager.find_corpus_sizes()
        return corpus_sizes
    
    def get_nb_seeds(self):
        nb_seeds = self.configManager.get_nb_seeds()
        return nb_seeds
    

    def plot_results(self, episodes_choice:str="all", legend_mode="mode1"):
        
        force_test_interval = True
        self.episodes_train, self.episodes_test = self.set_episodes_lists(episodes_choice, force_test_interval=force_test_interval)
        corpus_sizes_dict = self.get_corpus_sizes()
        nb_seeds = self.get_nb_seeds()
        
        color_palette = Visualizer.set_graph_style(is_final=self.is_final)
        hyperparam2color, color_random_policy = self.assign_color2hyperparam(color_palette)

        fig_text = plt.figure()
        fig_text.text(0, 0, self.config_dict_str, fontsize=6)

        nb_lines   = 2
        nb_columns = 3
        true_nb_columns = nb_columns + 1
        assert nb_lines*nb_columns == len(self.corpus_names)
        fig_train, axs_train = plt.subplots(nb_lines, true_nb_columns, sharey=False, figsize=(15, 7))
        fig_test, axs_test   = plt.subplots(nb_lines, true_nb_columns, sharey=False, figsize=(15, 7))
        h_pad = 3.5
        w_pad = 2
        fig_train.tight_layout(h_pad=h_pad, w_pad=w_pad)
        fig_test.tight_layout(h_pad=h_pad, w_pad=w_pad)

        # be careful of the order
        statistic_list = ["train", "test"]
        episodes_list  = [self.episodes_train, self.episodes_test]
        figs_types = [fig_train, fig_test]
        axs_types  = [axs_train, axs_test]

        if self.is_final:
            ylabels_list = ["episodic return" for statistic in statistic_list]
        else:
            ylabels_list = [statistic+" reward" for statistic in statistic_list]
        first=True
        if not self.is_final: extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        for plot_idx in range(len(axs_types)):
            fig = figs_types[plot_idx]
            y_label       = ylabels_list[plot_idx]
            statistic = statistic_list[plot_idx]
            episodes  = episodes_list[plot_idx]
            for k in range(len(self.corpus_names)):
                ax_x, ax_y    = Visualizer.idx2array_coordinates(k, nb_columns=nb_columns)
                ax            = axs_types[plot_idx][ax_x, ax_y]
                ax.set_xlabel("number of learners")
                ax.set_ylabel(y_label)
                corpus_name    = self.corpus_names[k]
                corpus_size    = corpus_sizes_dict[corpus_name]
                expectation_random_policy = Visualizer.compute_expectation_random_policy(corpus_size, episodes)
                corpus_results = self.corpus2results_dict[corpus_name]
                ax.set_title(corpus_name)
                idx = 1
                for hyperparam_id in self.sorted_hyperparams_ids_list:
                    hyperparam_dict = utils_files.load_json_file(PATHS.IDS_HYPERPARAMS+"/"+hyperparam_id)
                    hyperparam_results = corpus_results[hyperparam_id]
                    statistic_dict = hyperparam_results[statistic]
                    mean_rewards_list = [statistic_dict[str(ep)]["mean_rew"] for ep in episodes]
                    std_rewards_list  = [statistic_dict[str(ep)]["std_rew"]  for ep in episodes]
                    mean_curve = mean_rewards_list
                    high_curve, low_curve = Visualizer.get_confidence_curves(mean_list=mean_rewards_list, std_list=std_rewards_list, nb_seeds=nb_seeds)
                    label = self.get_label(hyperparam_id, hyperparam_dict, idx, mode=legend_mode)
                    ax.plot(episodes, mean_curve, label=label, color=hyperparam2color[hyperparam_id])
                    ax.fill_between(episodes, low_curve, high_curve,
                                    color=hyperparam2color[hyperparam_id],
                                    alpha=.1)
                    idx += 1
                ax.plot(episodes,[corpus_size]*len(episodes), color="black", linestyle = 'dashed', label="Max achievable return")
                ax.plot(episodes,expectation_random_policy, color=color_random_policy, linestyle="dotted", label="random policy (expected return)")
                # ger labels for legend
                if first:
                    handles, labels = ax.get_legend_handles_labels()
                    first=False
            # legend
            ax_x, ax_y = 0, 3
            ax       = axs_types[plot_idx][ax_x, ax_y]
            subtitle = self.build_subtitle(legend_mode) if not self.is_final else ""
            subtitle_extra_label  = [subtitle]  if not self.is_final else []
            subtitle_extra_handle = [extra]  if not self.is_final else []
            by_label = dict(zip(subtitle_extra_label+labels, subtitle_extra_handle+handles))
            ax.axis('off')
            title = "Hyperparameters" if not self.is_final else None
            bbox_to_anchor_w = 0.732
            bbox_to_anchor_h = 0.97
            frameon = False if self.is_final else True
            fig.legend(by_label.values(), by_label.keys() , title=title,
                        loc='upper left',
                        bbox_to_anchor=(bbox_to_anchor_w, bbox_to_anchor_h),
                        frameon=frameon)
            ax_x, ax_y = 1, 3
            ax_empty   = axs_types[plot_idx][ax_x, ax_y]
            ax_empty.axis('off')

        plt.show()


    @staticmethod
    def compute_expectation_random_policy(corpus_size, episodes):
        expectation = [1 for ep_idx in episodes]
        return expectation


    def build_subtitle(self, legend_mode):
        subtitle = ""
        if self.hyperparam2mean_score is not None:subtitle+="mean"
        if self.hyperparam2std_score is not None: subtitle+="(std)"
        if (self.comparison_params is not None) and (legend_mode=="mode1"):
            if self.hyperparam2mean_score is not None: subtitle+=" - "
            comparison_params_str = " / ".join(self.comparison_params)
            subtitle += comparison_params_str 
        return subtitle
    

    @staticmethod
    def get_confidence_curves(mean_list, std_list, nb_seeds):
        confidence_intervals = [Visualizer.confidence_interval(x_mean, x_std, nb_seeds, confidence=0.95) for x_mean, x_std in zip(mean_list, std_list)]
        high_curve = [interval["high"] for interval in confidence_intervals]
        low_curve  = [interval["low"] for interval in confidence_intervals]
        return high_curve, low_curve
    

    @staticmethod
    def confidence_interval(mean, std, nb_seeds, confidence=0.95):
        if confidence == 0.9:
            factor = 1.645
        elif confidence == 0.95:
            factor = 1.96
        elif confidence == 0.99:
            factor = 2.576
        else:
            raise Exception("Error: Confidence "+str(confidence)+" not supported.")
        error = factor*std/np.sqrt(nb_seeds)
        return {"low":mean-error, "high":mean+error}


    def get_label(self, hyperparams_id:str, hyperparam_dict:dict, idx: int, mode="mode1"):
        
        if self.hyperparam2mean_score is None:
            label = str(idx)+" - "
        elif self.hyperparam2std_score is None:
            mean_score = Visualizer.process_score(str(round(self.hyperparam2mean_score[hyperparams_id],2)))
            label = str(mean_score)+" - "
        else:
            mean_score = Visualizer.process_score(str(round(self.hyperparam2mean_score[hyperparams_id],2)))
            std_score  = Visualizer.process_score(str(round(self.hyperparam2std_score[hyperparams_id],1)))
            label = str(mean_score)+" ("+str(std_score)+") - "
        
        if self.comparison_params is not None:
            self.comparison_params = [self.comparison_params] if type(self.comparison_params)!=list else self.comparison_params
            for param_idx in range(len(self.comparison_params)):
                param = self.comparison_params[param_idx]
                path  = Visualizer.param2path(param)
                value = utils_dict.get_value_at_path(path, hyperparam_dict)
                if value is None: value = "none"
                if param in {"gnn_arch", "time_arch", "feedback_arch"}:
                    arch_list = value["arch"]
                    value = Visualizer.search_arch_name(param, arch_list, hyperparam_dict)
                separator = "" if param_idx==len(self.comparison_params)-1 else " | "
                final_value = Visualizer.filter_value(value)
                if mode == "mode1":
                    label += final_value + separator
                elif mode =="mode2":
                    label += param + ": "+ final_value + separator
                else:
                    raise Exception("Error: mode '"+str(mode)+"' not supported.")
        else:
            key1 = hyperparams_id.split("sec")[1]
            key2 = key1.split("-")[1]
            label += "hyperparam"+key2
        
        if self.is_final:
            if value=="embedding":
                label = "our policy (with Wikipedia2vec embeddings)"
            elif value=="one_hot_encoding":
                label = "our policy (with one-hot-encodings)"
            else:
                raise Exception("Unrecognized hyperparam value '"+str(value)+"' for final plot.")
        return label
    
    @staticmethod
    def process_score(score_str):
        integer_part, decimal_part = score_str.split(".")
        if len(integer_part)==1: integer_part = "0"+integer_part
        if len(decimal_part)==1: decimal_part = decimal_part+"0"
        return integer_part+"."+decimal_part

    @staticmethod
    def search_arch_name(arch_type:str, original_arch:list, hyperparam_dict:dict):
        assert type(original_arch) == list
        arch_type2arch_path = {"gnn_arch":PATHS.BANK_HYPERPARAMS_GNN_ARCH,"feedback_arch":PATHS.BANK_HYPERPARAMS_FEEDBACK_ARCH, "time_arch":PATHS.BANK_HYPERPARAMS_TIME_ARCH}
        architectures_dir = arch_type2arch_path[arch_type]
        architectures_filenames = utils_files.get_yaml_filenames(architectures_dir)
        relevant_arch_names = []
        for filename in architectures_filenames:
            filepath = architectures_dir + "/" + filename
            current_arch = utils_files.load_yaml_file(filepath)
            current_arch_list = current_arch["arch"]
            is_equal = Visualizer.is_equal(original_arch, current_arch_list, arch_type, hyperparam_dict)
            if is_equal:
                arch_name = filename.split(".yaml")[0]
                relevant_arch_names.append(arch_name)
        if len(relevant_arch_names)==1:
            return relevant_arch_names[0]
        elif len(relevant_arch_names)==0:
            raise Exception("Error: the above architecture was not found in '"+str(architectures_dir)+"'.")
        else:
            pprint.pprint(original_arch)
            for arch_name in relevant_arch_names:
                print(arch_name)
            raise Exception("Error: the above architecture was found in several arch names in '"+str(architectures_dir)+"' (see the arch names above).")


    @staticmethod
    def is_equal(original_arch:list, target_arch:list, arch_type:str, hyperparam_dict:dict):
        assert type(original_arch)==list and type(target_arch)==list
        if arch_type=="gnn_arch":
            target_arch1 = target_arch
            target_arch2 = copy.deepcopy(target_arch1)
            target_arch2 = [ {"layer_type":"kw2doc","layer_cls":"GeneralConv"} if layer["layer_cls"]=="MessagePassing" else layer for layer in target_arch2 ]
            target_arch_list = [target_arch1, target_arch2]
            if utils_dict.get_value_at_path(PATHS.CONFIG_HYPERPARAMS_DOC_ENCODING,hyperparam_dict)=="one-hot-encoding":
                target_arch1_bis = copy.deepcopy(target_arch1)
                target_arch1_bis.insert(0,{"layer_type": "doc2doc","layer_cls":"Linear"})
                target_arch_list.append(target_arch1_bis)

                target_arch2_bis = copy.deepcopy(target_arch2)
                target_arch2_bis.insert(0,{"layer_type": "doc2doc","layer_cls":"Linear"})
                target_arch_list.append(target_arch2_bis)
            for arch in target_arch_list:
                if arch == original_arch:
                    return True
            return False
        else:
            return original_arch == target_arch

    
    def assign_color2hyperparam(self, color_palette):
        colors_dict = {}
        for k in range(len(self.sorted_hyperparams_ids_list)):
            hyperparam_id = self.sorted_hyperparams_ids_list[k]
            colors_dict[hyperparam_id] = color_palette[k]
        color_random_policy = color_palette[3]
        return colors_dict, color_random_policy


    @staticmethod
    def idx2array_coordinates(idx, nb_columns=2):
        return idx//nb_columns, idx%nb_columns

    @staticmethod
    def set_graph_style(is_final=False):
        sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=12)     # fontsize of the axes title
        plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
        if not is_final:
            plt.rc('legend', fontsize=8)    # legend fontsize
            plt.rc('font', size=13)          # controls default text sizes
        else:
            plt.rc('legend', fontsize=10)
            plt.rc('font', size=13)
        
        color_palette = sns.color_palette("deep")
        color_palette = color_palette+color_palette+color_palette+color_palette+color_palette+color_palette
        #color_palette  = ["red", "blue", "green", "brown", "black"]

        return color_palette
    

    @staticmethod
    def filter_value(param_value):            
        if param_value == "one-hot-encoding": param_value = "one-hot"
        if param_value == "sinusoidal": param_value = "sin"
        if param_value == "embeddings_standard1": param_value = "emb_std1"
        return str(param_value)


    @staticmethod
    def param2path(param):
        if param == "policy":
            return PATHS.CONFIG_HYPERPARAMS_POLICY_NAME
        elif param == "doc_encoding":
            return PATHS.CONFIG_HYPERPARAMS_DOC_ENCODING
        elif param == "word_encoding":
            return PATHS.CONFIG_HYPERPARAMS_WORD_ENCODING
        elif param == "lr":
            return PATHS.CONFIG_HYPERPARAMS_LEARNING_RATE
        elif param == "doc_encoding":
            return PATHS.CONFIG_HYPERPARAMS_DOC_ENCODING
        elif param == "feedback_mode":
            return PATHS.CONFIG_HYPERPARAMS_FEEDBACK_MODE
        elif param == "features_params":
            return PATHS.CONFIG_HYPERPARAMS_WORD_ENCODING
        elif param == "attention_type":
            return PATHS.CONFIG_HYPERPARAMS_GNN_LAYERS_GENERALCONV_ATTENTION_TYPE
        elif param == "heads":
            return PATHS.CONFIG_HYPERPARAMS_GNN_LAYERS_GENERALCONV_HEADS
        elif param == "aggr":
            return PATHS.CONFIG_HYPERPARAMS_GNN_LAYERS_GENERALCONV_AGGR
        elif param == "repeat_per_collect":
            return PATHS.CONFIG_HYPERPARAMS_REPEAT_PER_COLLECT
        elif param == "gnn_arch":
            return PATHS.CONFIG_HYPERPARAMS_GNN_ARCH
        elif param == "feedback_arch":
            return PATHS.CONFIG_HYPERPARAMS_FEEDBACK_ARCH
        elif param == "time_arch":
            return PATHS.CONFIG_HYPERPARAMS_TIME_ARCH
        elif param == "middle_dim":
            return PATHS.MIDDLE_DIM
        elif param == "time_mode":
            return PATHS.CONFIG_HYPERPARAMS_TIME_MODE
        else:
            raise Exception("Error: param '"+str(param)+"' not supported.")