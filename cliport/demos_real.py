""" Created by: King Hang Wong
    Last update: Apr 4
    Desc:   This script creates real-world demonstration data for CLIPort, saving them as episodes.
            Episodes contain tuples of (obs, action, reward=0, info)
    Logic is based on cliport::cliport::demos.py, but highly modified


    # README:
    #
    # This script creates training dataset from ready real-world demonstrations.
    # WARNING: Action poses (i.e. act) should be given in terms of the XXX coordinate frame.
    # If raw, please process action poses using the `process_poses()` function before saving to episodes.
    # Observations (rgb, depth) will be properly formatted in this script.

"""

import os
import hydra
import numpy as np
import random
import cv2

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
from cliport import utils as cp_utils

#
from glob import glob
import logging
from typing import List, Tuple
from collections import namedtuple
import json

### add functions ###

class QuickMetainfo:
    """ Feel free to modify these"""
    BOUNDS = 
    PIX_SIZE =
    DATA_DIR = 

class QuickUtils:
    """ Date : Apr 5"""

    @staticmethod
    def load_eps_images(folder_path)->List[np.ndarray]:
        """
        Loads images for an entire episode
        Currently supports only .png files
        """
        image_paths = sorted(glob(os.path.join(folder_path, "*.png")))  # Sort to ensure correct order
        images = [cv2.imread(img_path) for img_path in image_paths]
        return images
    
    @staticmethod
    def load_eps_actions(action_file)->List[str]:
        """
        Loads action JSONL file for an entire episode
        """
        def load_jsonl(filepath: str) -> list[dict]:
            with open(filepath, 'r') as file:
                return [json.loads(line) for line in file if line.strip()]
    
    @staticmethod
    def get_dummy_cams(n_cams = 1):
        Dummycams = namedtuple('Dummycams', ['cams'])
        Dummycam = namedtuple('Dummycam', '')
        return Dummycams([Dummycam() for i in range(n_cams)])

    @staticmethod
    def get_observation(colors, depths, i)->dict[str, Tuple[np.ndarray]]:
        """ Returns color and depth observation in seamless cliport format"""

        obs = {'color': (), 'depth': ()}

        color, depth = colors[i], depths[i]

        ## single camera setup
        if len(color.shape) == 2:
            obs['color'] += (color,)
            obs['depth'] += (depth,)

        ## unpack multiple camera views
        elif len(color.shape) > 2:
            cams = QuickUtils.get_dummy_cams(len(color.shape))  # simulate
            for i, cam in enumerate(cams):
                # assume (N x H x W x C)
                obs['color'] += (color[i, ...],)
                obs['depth'] += (depth[i, ...],)
        return obs

    @staticmethod
    def get_action(eps_actions, i):
        """ 
        Loads action from a episodic textfile

        Action obj format should be as per cliport:cliport:tasks:task.py
        {   
            'pick pose'    : Tuple[np.ndarray, np.ndarray] ### ((3,), (4,))
            'place pose'   : Tuple[np.ndarray, np.ndarray] ### ((3,), (4,))
        }
        
        Poses should already be transformed to the appropriate coordinate frames
        during demonstration data processing step.
        i.e., no further treatment should be required in this file.
        """
        return eps_actions[i]
    
#### End ####

# @hydra.main(config_path='./cfg', config_name='data')
# TO-DO: remove hydra dependecy
@hydra.main(config_path='./cfg', config_name='mydata')
def main(cfg):

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='myapp.log', level=logging.INFO)

    # Initialize environment and task.
    task = tasks.names[cfg['task']]()   # tasks.names is a dict
    task.mode = cfg['mode']

    # Initialize scripted oracle agent and dataset.
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    logger.info(f"Saving to: {data_path}")
    logger.info(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    #################
    #
    # Sample episode file structure:
    ###
    # data/
    # │── episode_1/
    # │   ├── rgb/
    #         ├── img_1.png
    # │   ├── depth
    #         ├── img_1.png
    # │   ├── segm
    #         ├── img_1.png
    # │   ├── actions.txt
    #
    # │── episode_2/
    # │   ├── ...
    ###
    
    episode_folders = sorted(glob(os.path.join(QuickMetainfo.DATA_DIR, "episode_*")))

    relevant_color_names = cfg['relevant_color_names']   ## 2 at most in set-up
    if len(relevant_color_names) == 2:
            relevant_desc = f'{relevant_color_names[0]} and {relevant_color_names[1]}'
    elif len(relevant_color_names) == 1:
        relevant_desc = f'{relevant_color_names[0]}'
    else:
        logger.warning('Unsupported number of colors.')
        raise NotImplementedError

    # Create training dataset from ready real-world demonstrations.
    # Action poses (i.e. act) should be given in terms of the XXX coordinate frame.
    # could be raw or already transformed. If raw, please process them using the `process_poses()` function before saving to episodes.
    # Observations 

    while dataset.n_episodes < cfg['n']:

        if dataset.n_episodes >= len(episode_folders):
            logger.info("Not enough real-world episodes available.\n Process complete")
            break

        episode = []
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Real world demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        episode_folder = episode_folders[dataset.n_episodes]
        
        eps_rgbs = QuickUtils.load_eps_images(os.path.join(episode_folder, 'rgb_images/'))
        eps_depths = QuickUtils.load_eps_images(os.path.join(episode_folder, 'depth_images/'))
        
        action_file = os.path.join(episode_folder, "actions.txt")
        eps_actions : List[dict] = QuickUtils.load_eps_actions(action_file)

        reward = 0   # placeholder ; CLIPort does not use reward.
        info = dict() ## requires key-value pair: ['lang_goal'] : str

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        ##  Parse each obs, act in the episode.
        for i in range(min(len(eps_rgbs), len(eps_actions), task.max_steps)):

            obs = QuickUtils.get_observation(eps_rgbs, eps_depths, i) ## package rgb and depth into cliport format
            act = QuickUtils.get_action(eps_actions, i)

            info['lang_goal'] = "pack all the {colors} blocks into the white bowl".format(colors=relevant_desc)
            episode.append(obs, act, 0, info)
        
        # demonstration completed
        info['lang_goal'] = task.task_completed_desc  # "done packing blocks."
        episode.append((obs, None, 0, info))

        # assume all RW demonstrations are successful and completed
        dataset.add(seed, episode)


if __name__ == '__main__':
    main()
