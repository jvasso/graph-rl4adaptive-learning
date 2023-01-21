from configManager import ConfigManager
from resultsManager import ResultsManager
from expResults import ExpResults

import pprint

criteria = ExpResults.LAST_TRAIN_REWARD

configManager = ConfigManager()
resultsManager = ResultsManager(configManager=configManager)

infos = resultsManager.rank_according_to_criteria(criteria)
    
print("\Scores per corpus:")
pprint.pprint(infos["mean_scores_per_corpus"])
print("\nRanking details:")
pprint.pprint(infos["ranking_details"])
print("\nGLOBAL RANKING:")
pprint.pprint(infos["global_ranking"])

resultsManager.save_report()
