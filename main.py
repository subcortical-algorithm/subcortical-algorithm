import sys
sys.path.insert(0, 'subcortical/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time

import numpy as np
import os.path as osp
from argparse import ArgumentParser

from utils.utils import make_movie
from utils.dataloader import get_dataset
from engine import Engine

def get_dm_config(n):
    # Parameters are set via analysis of the dynamical system. For each scenario, 
    # parameters are not unique. 

    if n == 5:
        dm_config = {"n": 15,
                     "dt": 1,
                     "tau": 20,
                     "alpha": 1.5,
                     "beta": 0.8,
                     "theta": -5,
                     "gamma": 0.1,
                     "Je": 20,
                     "Jm": -8.12,
                     "I0": 10.33}
    if n == 10:
        dm_config = {"n": 10,
                     "dt": 1,
                     "tau": 20,
                     "beta": 1,
                     "theta": 1,
                     "Je": 18,
                     "Jm": -11.71,
                     "I0": 14.33}

    if n == 15:
        dm_config = {"n": 5,
                     "dt": 1,
                     "tau": 20,
                     "beta": 3.2,
                     "theta": 3,
                     "Je": 9,
                     "Jm": -16.5,
                     "I0": 7.18}

    return dm_config



parser = ArgumentParser()
parser.add_argument("dataset_path", help="The location of dataset.")

parser.add_argument("n", type=int, choices=[5, 10, 15], help="The total number of classes to run this demo on.")

parser.add_argument("--do_vis", action="store_true", dest="do_vis_flg", 
    help="Do visualization after finishing the program.")

parser.add_argument("--vis_path", action="append", dest="vis_path", default=None, type=str,
    help="The folders that store visualization data. Can be called multiple times.")

parser.add_argument("--vis_result_path", action="store", dest="vis_res_path", type=str,
    help="The location for storing visualization results.", default=".")

parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
    help="Print status messages to stdout.")

args = parser.parse_args()


# dataset_path = '/home/charlie/dataset/gait_video_data/gait_skeleton_000/'
dataset_path = args.dataset_path



reservoir_config = {"dt": 1,
                    "n_layer": 1,
                    "N0":(72, 96),
                    "N": [1024],
                    "tau": [3],
                    "rho_scale": [1.05],
                    "strength_in": [1]}


target_config = {"type": "ffcurrent",
                 "total_step": 50,
                 "learning_rate": 0.01}


protocol = {"n_epoch":1, "n_train": 1, "n_val":0}

dm_config = get_dm_config(args.n)
n_class = dm_config["n"]

# Define dataset
# Here we have a dataset of 75 categories, we randomly sample 15 categories
# out of the 75 categories. For each categories, we have 5 examples served 
# as training data, 10 examples served as validation data, 35 examples served
# as testing data.
dataset = get_dataset(dataset_path, 
                      n_class=n_class, 
                      n_train=5, 
                      n_validate=10, 
                      n_test=35, 
                      centering=True)

# Define engine
# We pass training protocol, configuration files for reservoir network, decision-
# making network, and targets, along with our dataset to the Engine.
engine = Engine(protocol=protocol, 
                reservoir_config=reservoir_config,
                dm_config=dm_config, 
                target_config=target_config,
                dataset=dataset)

# Run the engine to train the model so as to learn to dicriminate different
# spatiotemporal pattern categories. We set save_activity=True for the purpose
# of visualization.

do_vis_flg = args.do_vis_flg
vis_res_path = args.vis_res_path

start_time = time.time()

engine.run(verbose=args.verbose, save_activity=do_vis_flg, dest=vis_res_path)

if do_vis_flg:
    print("Making movies from visualization results.")

    make_movie(activity_root=osp.join(vis_res_path, 'train'), vis_path_list = args.vis_path)

    make_movie(activity_root=osp.join(vis_res_path, 'test'), vis_path_list = args.vis_path)


print("Done.")

end_time = time.time()

dur_time = end_time - start_time
print("\nTook %.2f seconds." % dur_time)









    