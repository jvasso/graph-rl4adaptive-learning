
def BUILD_GLOBAL_VAR_DICT(save, load):
    GLOBAL_VAR_DICT = {"SAVE":save, "LOAD":load}
    IS_REMOTE = False
    GLOBAL_VAR_DICT["IS_REMOTE"]=IS_REMOTE
    USE_MULTIPROCESSING = False
    NUM_CORES = 1
    GLOBAL_VAR_DICT["USE_MULTIPROCESSING"]=USE_MULTIPROCESSING
    GLOBAL_VAR_DICT["NUM_CORES"]=NUM_CORES
    return GLOBAL_VAR_DICT