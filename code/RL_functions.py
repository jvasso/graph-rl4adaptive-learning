import GLOBAL_VAR
from linearCourseEnv import LinearCourseEnv
from GNNAgent import GNNAgent
from myActor  import MyActor
from myLogger import MyLogger
from corpusGraph import CorpusGraph
from config import Config

import torch
import numpy as np

from gym.spaces import Discrete

from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils import LazyLogger


def run_train_test(config:Config):

    seed                = config.get_seed()
    corpus_name         = config.get_corpus_name()
    corpus_size         = config.get_corpus_size()
    node_features       = config.get_node_features()
    env_params          = config.get_env_params()
    gnn_agent_params    = config.get_gnn_agent_params()
    learning_rate       = config.get_learning_rate()
    policy_params       = config.get_policy_params()
    policy_name         = config.get_policy_name()
    total_size          = config.get_VectorReplayBuffer_size()
    trainer_params      = config.get_trainer_params()
    logger_params       = config.get_logger_params()
    step_per_episode    = config.get_step_per_episode()

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(mode=True)
    torch.autograd.set_detect_anomaly(True)
    
    corpusGraph = CorpusGraph(corpus_name, node_features)
    
    env_verbose = False
    train_envs = DummyVectorEnv([create_env_with_params(corpusGraph, env_params, verbose=env_verbose) for _ in range(1)])
    test_envs  = DummyVectorEnv([create_env_with_params(corpusGraph, env_params, verbose=env_verbose) for _ in range(1)])
    one_env    = LinearCourseEnv(corpusGraph, env_params, verbose=False)

    # policy
    dist = torch.distributions.Categorical    
    kw_size       = corpusGraph.get_kw_size()
    feedback_size = one_env.get_feedback_size()
    time_size     = one_env.get_time_size()
    gnn_agent = GNNAgent(gnn_agent_params, kw_size=kw_size, feedback_size=feedback_size, time_size=time_size)
    action_space = Discrete(corpus_size)
    if policy_name == "PGPolicy":
        actor  = MyActor(gnn_agent).to(GLOBAL_VAR.DEVICE)
        optim  = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        policy = PGPolicy(model=actor, optim=optim, dist_fn=dist, action_space=action_space,**policy_params)
    else:
        raise Exception("Policy '"+str(policy_name)+"' not supported.")
    
    # collector
    replayBuffer    = VectorReplayBuffer(total_size=total_size, buffer_num=len(train_envs))
    train_collector = Collector(policy=policy, env=train_envs, buffer=replayBuffer)
    test_collector  = Collector(policy=policy, env=test_envs)
    
    # logger
    log_file_path = config.create_log_file_path()
    if GLOBAL_VAR.SAVE:
        logger = MyLogger(log_file_path=log_file_path,
                        step_per_episode=step_per_episode,
                        **logger_params)
    else:
        logger = LazyLogger()
    
    trainer = GLOBAL_VAR.POLICY2TRAINER_CLS[policy_name]
    train_result = trainer(policy=policy, train_collector=train_collector, test_collector=test_collector,
                            logger=logger, **trainer_params)
    


def create_env_with_params(corpusGraph, env_params, verbose=False):
    env = LinearCourseEnv(corpusGraph, env_params, verbose=verbose)
    # other operations such as env.seed(np.random.choice(10))
    def create_env():
        return env
    return create_env
