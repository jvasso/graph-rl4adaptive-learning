import torch
import os
import time

SAVE = "not_assigned"
LOAD = "not_assigned"

IS_REMOTE           = "not_assigned"
USE_MULTIPROCESSING = "not_assigned"

USE_CUDA   = "not_assigned"
CPU_DEVICE = "not_assigned"
NUM_CORES  = "not_assigned"

def SET_SAVE(new_bool: bool):
    global SAVE
    if new_bool == False:
        print("\n\n\n")
        print("############################################")
        print("### !!! WARNING: NOT SAVING ANYTHING !!! ###")
        print("############################################")
        print("\n\n\n")
        time.sleep(1)
    SAVE = new_bool

def SET_LOAD(new_bool: bool):
    global LOAD
    if new_bool == False:
        print("\n\n\n")
        print("#############################################")
        print("### !!! WARNING: NOT LOADING ANYTHING !!! ###")
        print("#############################################")
        print("\n\n\n")
        time.sleep(1)
    LOAD = new_bool

def SET_SAVE_AND_LOAD(save:bool, load:bool):
    SET_SAVE(save)
    SET_LOAD(load)

def SET_USE_MULTIPROCESSING(use_mutliprocessing):
    global USE_MULTIPROCESSING
    USE_MULTIPROCESSING = use_mutliprocessing

def SET_NUM_CORES(num_cores):
    global NUM_CORES
    NUM_CORES = num_cores

def SET_IS_REMOTE(is_remote):
    global IS_REMOTE
    IS_REMOTE = is_remote

def CONFIG_DEVICE():
    global IS_REMOTE, USE_CUDA, DEVICE, CPU_DEVICE

    CPU_DEVICE = torch.device("cpu")
    if IS_REMOTE:
        USE_CUDA = False
        if USE_CUDA:
            if not torch.cuda.is_available(): raise Exception("Error: CUDA not available.")
            gpu_idx = 0
            torch.cuda.set_device(gpu_idx)
            DEVICE = torch.device("cuda:"+str(gpu_idx))
        else:
            DEVICE = CPU_DEVICE
    else:
        USE_CUDA = False
        DEVICE  =  CPU_DEVICE
        os.environ['KMP_DUPLICATE_LIB_OK']='True'


def CONFIGURE_ALL(configurations_dict):
    SET_IS_REMOTE(configurations_dict["IS_REMOTE"])
    SET_USE_MULTIPROCESSING(configurations_dict["USE_MULTIPROCESSING"])
    if configurations_dict["USE_MULTIPROCESSING"]: SET_NUM_CORES(configurations_dict["NUM_CORES"])
    CONFIG_DEVICE()
    SET_SAVE_AND_LOAD(configurations_dict["SAVE"], configurations_dict["LOAD"])


RESULTS_FILES_EXT = ".pkl"

from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
STR2CLS_POLICY      = {"PGPolicy":PGPolicy}
POLICY2TRAINER_NAME = {"PGPolicy":"onpolicy_trainer"}
trainer_name2cls = {"onpolicy_trainer":onpolicy_trainer}
POLICY2TRAINER_CLS = { policy_name: trainer_name2cls[trainer_name] for policy_name,trainer_name in POLICY2TRAINER_NAME.items()}