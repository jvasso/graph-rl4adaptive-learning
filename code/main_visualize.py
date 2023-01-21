
from resultsManager import ResultsManager
from visualizer import Visualizer


selection="best_2"
hyperparams_ids_dict, exp_params_ids_list, full_hyperparams_ids_list, hyperparam2mean_score, hyperparam2std_score = ResultsManager.follow_gridsearch_report(selection=selection)
episodes_choice = "all"
comparison_params = None
visualizer = Visualizer(hyperparams_ids_dict, exp_params_ids_list, comparison_params=comparison_params,
                        full_hyperparams_ids_list=full_hyperparams_ids_list,
                        hyperparam2mean_score=hyperparam2mean_score, hyperparam2std_score=hyperparam2std_score)
legend_mode="mode1"
visualizer.plot_results(episodes_choice=episodes_choice,legend_mode=legend_mode)
