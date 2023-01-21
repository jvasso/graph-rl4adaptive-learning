from corpusGraph import CorpusGraph
from customObservation import CustomObservation

import gym
from gym import spaces

import torch
import numpy as np

class LinearCourseEnv(gym.Env):

    NOT_VISITED        = "nv"
    NOT_UNDERSTOOD     = "nu"
    NOT_UNDERSTOOD_NOW = "nun"
    UNDERSTOOD         = "u"
    TOO_EASY           = "te"
    INSTANT_REWARD = "instant"
    DELAYED_REWARD = "delayed"
        
    def __init__(self, corpusGraph: CorpusGraph, env_params: dict, verbose=False):

        super(LinearCourseEnv, self).__init__()
        self.verbose = verbose
        
        # basic infos
        self.corpusGraph  = corpusGraph
        self.nb_doc       = self.corpusGraph.get_nb_doc()
        self.nb_kw        = self.corpusGraph.get_nb_kw()
        
        # env params
        self.reward_mode       = env_params["reward_mode"]
        self.progression_bonus = env_params["progression_bonus"]
        self.time_penalty      = env_params["time_penalty"]
        self.horizon           = env_params["step_per_episode"]
        assert type(self.horizon) == int
        
        self.time_mode     = env_params["time_mode"]
        self.feedback_mode = env_params["feedback_mode"]
        self.set_feedback_mode(self.feedback_mode)
        
        self.eval_docs  = self.set_eval_docs(env_params["eval_ids"])
        self.max_eval_score = len(self.eval_docs)
        
        # action & observation spaces
        self.action_space      = spaces.Discrete(self.nb_doc)
        self.observation_space = spaces.Dict()

        if self.time_mode == "no":
            self.time_size = 0
        elif self.time_mode == "counter":
            self.time_size = 1
        elif self.time_mode == "sinusoidal":
            self.time_size = 40
        else:
            raise Exception("Error: time mode '"+str(self.time_mode+"' not supported."))

        # statistics
        self.done_count = 0

        self.best_achievable_reward = self.compute_best_reward(verbose=True)
        
        # initialize learner interactions
        self.init_state()
        

    def init_state(self):
        self.user_state = { doc_id:self.init_feedback for doc_id in range(0, self.nb_doc)}
        self.user_state_tensor = torch.cat( [ torch.clone(self.feedback2tensor[feedback]) for doc_id,feedback in self.user_state.items()], dim=0)
        self.step_count = 0
        self.remaining_steps = self.horizon - self.step_count

        # stats
        self.past_actions = []
        self.nb_doc_understood = 0
        self.trajectory_length = 0
        self.cumul_reward      = 0
    

    def step(self, action):
        assert self.action_space.contains(action)
        self.step_count+=1
        self.remaining_steps = self.horizon - self.step_count

        feedback = self.generate_feedback(action)
        self.update_user_state(current_doc=action, current_feedback=feedback)
        done     = self.is_done()
        reward   = self.compute_reward(action, feedback, done)
        obs      = self.build_obs()
        # stats
        if done :
            self.done_count += 1
        self.cumul_reward += reward
        self.past_actions.append(action)
        if self.verbose: print("step :", self.step_count-1, "|| action :", action, "|| reward :", reward)
        info = {}
        return obs, reward, done, info
    
   
    def is_done(self):
        if self.step_count == self.horizon:
            assert self.remaining_steps == 0
            return True
        elif self.all_eval_understood():
            return True
        return False


    def generate_feedback(self, action):
        if self.is_understood(doc=action):
            return self.TOO_EASY
        elif self.has_prerequisites(action):
            return self.UNDERSTOOD
        else:
            return self.NOT_UNDERSTOOD_NOW
    
    
    def all_eval_understood(self):
        for doc in self.eval_docs:
            if not self.is_understood(doc):
                return False
        return True
    
    
    def update_user_state(self, current_doc, current_feedback):
        # update previous docs
        for doc, feedback in self.user_state.items():
            if feedback == LinearCourseEnv.NOT_UNDERSTOOD_NOW:
                self.user_state[doc]        = LinearCourseEnv.NOT_UNDERSTOOD
                self.user_state_tensor[doc] = torch.clone(self.feedback2tensor[LinearCourseEnv.NOT_UNDERSTOOD])
        #update current doc
        self.user_state[current_doc]        = current_feedback
        self.user_state_tensor[current_doc] = torch.clone(self.feedback2tensor[current_feedback])


    def build_obs(self):
        kw_features         = self.corpusGraph.get_kw_features()
        doc_features        = self.corpusGraph.get_doc_features()
        feedback_features   = torch.clone(self.user_state_tensor)
        edge_indices_doc2kw = self.corpusGraph.get_doc2kw_edge_idx()
        edge_indices_kw2doc = self.corpusGraph.get_kw2doc_edge_idx()
        remaining_time      = self.build_remaining_time(self.step_count, self.remaining_steps)
        obs_object = CustomObservation(kw_features=kw_features, doc_features=doc_features, feedback_features=feedback_features,
                                        edge_indices_doc2kw=edge_indices_doc2kw, edge_indices_kw2doc=edge_indices_kw2doc,
                                        remaining_time=remaining_time)
        return obs_object
    

    def build_remaining_time(self, current_step, remaining_steps):
        if self.time_mode == "no":
            return torch.zeros(1)
        elif self.time_mode == "counter":
            return torch.unsqueeze(torch.unsqueeze(torch.tensor(remaining_steps),dim=0),dim=0)
        elif self.time_mode == "sinusoidal":
            return torch.unsqueeze(LinearCourseEnv.sinusoidal_time_encoding(remaining_steps, vector_length=self.time_size),dim=0)
        else:
            raise Exception("Time mode '"+str(self.time_mode)+"' not supported.")
    
    @staticmethod
    def sinusoidal_time_encoding(remaining_steps, vector_length):
        """
        Generate an encoding of the remaining time as a fixed-length vector.
        """
        encoding = torch.zeros(vector_length)
        for i in range(vector_length):
            encoding[i] = torch.sin(torch.tensor(remaining_steps / (10000 ** ((2 * i) / vector_length))))
        return encoding
    

    def compute_reward(self, action, feedback, done):
        new_understanding = 1 if feedback==LinearCourseEnv.UNDERSTOOD else 0
        
        progression_reward  = self.progression_bonus if new_understanding else 0 # reward for understanding new content (regardless of whether it's an eval or not)
        time_penalty        = - np.absolute(self.time_penalty) # time penalty
        instant_reward      = progression_reward + time_penalty # reward to "help" the algorithm (not taking into account eval)
        
        if self.reward_mode == LinearCourseEnv.INSTANT_REWARD:
            instant_eval_reward = self.compute_instant_eval_reward(new_understanding, action)
            return instant_eval_reward + instant_reward
        elif self.reward_mode == LinearCourseEnv.DELAYED_REWARD:
            if done:
                delayed_eval_reward = self.compute_delayed_eval_reward()
                return delayed_eval_reward + instant_reward
            else:
                return instant_reward
        else :
            raise Exception("Unknown reward type:"+self.reward_mode)
    

    def compute_instant_eval_reward(self, new_understanding, action):
        if new_understanding and self.is_eval(action):
            return 1
        return 0
    
    def compute_delayed_eval_reward(self):
        score = 0
        for doc in self.eval_docs:
            if self.is_understood(doc):
                score+=1
        return score

    
    def compute_best_reward(self, verbose=True):
        best_eval_score = self.max_eval_score

        min_trajectory_length = max(self.eval_docs)+1
        best_progression_reward = min_trajectory_length*self.progression_bonus
        best_time_penalty = - min_trajectory_length*np.absolute(self.time_penalty)

        best_reward = best_eval_score + best_progression_reward + best_time_penalty

        if verbose:
            print("##### BEST ACHIEVABLE REWARD: ", best_reward) if self.verbose else 0
        return best_reward
            

    def has_prerequisites(self, doc_idx):
        if doc_idx == 0 :
            return True
        else :
            if self.is_understood(doc_idx-1) :
                return True
            return False
    

    def reset(self, seed=1):
        self.init_state()
        obs = self.build_obs()
        return obs


    def seed(self,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)


    def render(self, mode='human'):
        # Render the environment to the screen
        pass
    
    def set_eval_docs(self, eval_code):
        if eval_code == "all":
            return set(range(self.nb_doc))
        elif eval_code == "last":
            return set([self.nb_doc-1])
        elif type(eval_code) == list:
            assert (min(eval_code) >= 0) and (max(eval_code) <= self.nb_doc)
            return set(eval_code)
        else:
            Exception("Eval code '"+str(eval_code)+"' not supported.")


    def is_eval(self, doc_idx):
        if doc_idx in self.eval_docs :
            return True
        return False
    
    def is_understood(self, doc):
        past_feedback = self.user_state[doc]
        if past_feedback in self.understand_feedbacks:
            return True
        else:
            return False
    

    def set_feedback_mode(self, feedback_mode):
        if feedback_mode == "mode1":
            self.nb_feedback_types  = 4
            self.understand_feedbacks     = {LinearCourseEnv.UNDERSTOOD , LinearCourseEnv.TOO_EASY}
            self.not_understand_feedbacks = {LinearCourseEnv.NOT_VISITED, LinearCourseEnv.NOT_UNDERSTOOD, LinearCourseEnv.NOT_UNDERSTOOD_NOW}
            self.reward_feedbacks         = {LinearCourseEnv.UNDERSTOOD}
            self.feedback2encoding  = {LinearCourseEnv.NOT_VISITED:0, LinearCourseEnv.NOT_UNDERSTOOD:1, LinearCourseEnv.NOT_UNDERSTOOD_NOW:1, LinearCourseEnv.UNDERSTOOD:2, LinearCourseEnv.TOO_EASY:3}
        elif feedback_mode == "mode2":
            self.nb_feedback_types  = 5
            self.understand_feedbacks     = {LinearCourseEnv.UNDERSTOOD , LinearCourseEnv.TOO_EASY}
            self.not_understand_feedbacks = {LinearCourseEnv.NOT_VISITED, LinearCourseEnv.NOT_UNDERSTOOD, LinearCourseEnv.NOT_UNDERSTOOD_NOW}
            self.reward_feedbacks         = {LinearCourseEnv.UNDERSTOOD}
            self.feedback2encoding  = {LinearCourseEnv.NOT_VISITED:0, LinearCourseEnv.NOT_UNDERSTOOD:1, LinearCourseEnv.NOT_UNDERSTOOD_NOW:2, LinearCourseEnv.UNDERSTOOD:3, LinearCourseEnv.TOO_EASY:4}
        elif feedback_mode == "mode3":
            self.nb_feedback_types  = 2
            self.understand_feedbacks     = {LinearCourseEnv.UNDERSTOOD , LinearCourseEnv.TOO_EASY}
            self.not_understand_feedbacks = {LinearCourseEnv.NOT_VISITED, LinearCourseEnv.NOT_UNDERSTOOD, LinearCourseEnv.NOT_UNDERSTOOD_NOW}
            self.reward_feedbacks         = {LinearCourseEnv.UNDERSTOOD}
            self.feedback2encoding  = {LinearCourseEnv.NOT_VISITED:0, LinearCourseEnv.NOT_UNDERSTOOD:0, LinearCourseEnv.NOT_UNDERSTOOD_NOW:0, LinearCourseEnv.UNDERSTOOD:1, LinearCourseEnv.TOO_EASY:1}
        else:
            raise Exception("Error: feedback_mode '"+str(feedback_mode)+"' not supported.")
        self.feedback2tensor = {feedback: torch.unsqueeze(torch.zeros(self.nb_feedback_types).scatter_(0, torch.tensor([feedback_encoding]),1),dim=0) for feedback,feedback_encoding in self.feedback2encoding.items()}
        self.init_feedback = LinearCourseEnv.NOT_VISITED
        self.feedback_size = self.nb_feedback_types


    def get_feedback_size(self):
        return self.feedback_size
    
    def get_time_size(self):
        return self.time_size