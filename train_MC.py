"""
Multiclass Queuing Network scheduling policy optimization using
Proximal Policy Optimization method with Approximating Martingale-Process variance reduction
PPO:
https://arxiv.org/abs/1707.06347 (by Schulman et al., 2017)
Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf (by Schulman et al., 2017)
Approximating Martingale-Process Method (by Henderson, Glynn, 2002):
https://web.stanford.edu/~glynn/papers/2002/HendersonG02.pdf
"""
import tensorflow as tf
import ray  # package for distributed computations
import numpy as np

from policy import Policy


import os
import argparse
from env import Env
import random
import datetime
import copy
import psutil
import gc

# from memory_profiler import profile

# import multiprocessing
# import logging
# logger = logging.getLogger(__name__)
# if "OMP_NUM_THREADS" not in os.environ and multiprocessing.cpu_count() > 8:
#   logger.warning("[ray] Forcing OMP_NUM_THREADS=1 to avoid performance "
#                 "degradation with many CPUs. You can override this by "
#                 "explicitly setting OMP_NUM_THREADS.")
if True:
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
    #
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ray.init( temp_dir='/dev/shm/ray4',  object_store_memory= 500 * 1000 * 1024 * 1024, num_cpus=50)
# num_gpus=100, object_store_memory=500 * 1000 * 1024 * 1024
# 500 gib

MAX_ACTORS = 80  # max number of parallel simulations
# MAX_EPISODES = 5  # max number of episodes under an actor, when calculating the next states' expected values






def main(  batch_size ):
    """
    Main training loop
    :param: see ArgumentParser below
    """


    #remote_network = ray.remote(Policy)

    simulators=[Policy.remote(j) for j in range(batch_size)]

    for i in range(batch_size):

        print(ray.get(simulators[i].get_num.remote()))



    pass


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Ride-hailing: Train deep-NN policy using Proximal Policy Optimizer')

    parser.add_argument('-b', '--batch_size', type=int, help='Number of parallel actors per training batch',
                        default = 50)


    args = parser.parse_args()
    main(  **vars(args))

