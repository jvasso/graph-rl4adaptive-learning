import GLOBAL_VAR

from tianshou.utils.logger.base import BaseLogger, LOG_DATA_TYPE

from typing import Callable, Dict, Optional, Tuple, Union
import utils_files
import utils_dict


class MyLogger(BaseLogger):

    INIT_LOG_DICT = {"train":{}, "test":{}}
    
    def __init__(
        self,
        step_per_episode: int,
        log_file_path: str,
        train_interval_ep: int = 1,
        test_interval_ep: int = 1,
        update_interval_ep: int = 1
    ):
        super().__init__()
        self.step_per_episode = step_per_episode
        self.log_file_path    = log_file_path

        self.train_interval_ep  = train_interval_ep
        self.test_interval_ep   = test_interval_ep
        self.update_interval_ep = update_interval_ep
        
        self.last_log_train_ep  = 0
        self.last_log_test_ep   = 0
        self.last_log_update_ep = 0
        
        self.init_log_file()
    
    
    def init_log_file(self):
        if GLOBAL_VAR.SAVE:
            log_dict = MyLogger.INIT_LOG_DICT
            utils_files.save_pickle_dict_to_path(self.log_file_path, log_dict, is_new=True)
    
    
    def write(self, step_type: str, step: int, new_log_data: LOG_DATA_TYPE):
        """Specify how the writer is used to log data.
        
        :param str step_type: namespace which the data dict belongs to. --> "train/env_ep", "test/env_ep"
        :param int step: stands for the ordinate of the data dict.
        :param dict data: the data to write with format ``{key: value}``.
        """
        if GLOBAL_VAR.SAVE and step_type != "update/gradient_step":

            # load log file
            log_dict = utils_files.load_pickle_file(self.log_file_path)
            assert log_dict is not None

            # update val
            episode_idx = new_log_data["ep_idx"]
            path   = step_type + "/" + str(episode_idx)
            utils_dict.set_value_at_path(path, new_log_data, log_dict, new_path=True)
            
            # save log file
            utils_files.save_pickle_dict_to_path(self.log_file_path, log_dict, is_new=False)

    
    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.
        
        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        # in case
        if step <= self.last_log_train_step:
            raise Exception("Error: step = "+str(step)+" and last_log_train_step = "+str(self.last_log_train_step))
        
        episode_idx = step//self.step_per_episode
        if collect_result["n/ep"] > 0:
            if episode_idx - self.last_log_train_ep >= self.train_interval_ep:
                log_data = {
                    "ep_idx": episode_idx,
                    "step":step,
                    "rew": collect_result["rew"],
                    "length": collect_result["len"],
                }
                self.write("train", step, log_data)
                self.last_log_train_ep = episode_idx
                self.last_log_train_step = step


    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0

        # in case
        # not sure it's useful here
        if step <= self.last_log_test_step:
            raise Exception("Error: step = "+str(step)+" and last_log_test_step = "+str(self.last_log_test_step))
        episode_idx = step//self.step_per_episode
        if episode_idx - self.last_log_test_ep >= self.test_interval_ep:
            log_data = {
                    "ep_idx": episode_idx,
                    "step": step,
                    "rew": collect_result["rew"],
                    "length": collect_result["len"],
                    "rew_std": collect_result["rew_std"],
                    "length_std": collect_result["len_std"]
            }
            self.write("test", step, log_data)
            self.last_log_test_ep = episode_idx
            self.last_log_test_step = step


    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_update_step >= self.update_interval:
            log_data = {f"update/{k}": v for k, v in update_result.items()}
            self.write("update/gradient_step", step, log_data)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        pass

    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        pass

