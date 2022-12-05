import os, sys
import copy
import json
import argparse
import yaml
from glob import glob


import numpy as np
import matplotlib.pyplot as plt

class ParamNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def overwrite(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if k in self.__dict__.keys() and v is not None:
                self.__dict__[k] = v

class ParamLib:
    def __init__(self, config_path: str):
        self.config_path = config_path

        # Create all parameter dictionaries
        self.main = ParamNamespace()
        self.paths = ParamNamespace()
        self.preprocessing = ParamNamespace()
        self.model = ParamNamespace()
        self.driving = ParamNamespace()

        # Load config file with parametrization, create paths and do sys.path.inserts
        self.load_config_file(self.config_path)
        #self.create_dir_structure()
        self.add_system_paths()

    def load_config_file(self, path: str):
        """
        Loads a config YAML file and sets the different dictionaries.
        Args:
            path: path to some configuration file in yaml format

        Returns:
        """

        with open(path, 'r') as stream:
            try:
                config_file = yaml.safe_load(stream)
            except yaml.YAMLError as exception:
                print(exception)

        # Copy yaml content to the different dictionaries.
        vars(self.main).update(config_file['main'])
        vars(self.paths).update(config_file['paths'])
        vars(self.preprocessing).update(config_file['preprocessing'])
        vars(self.model).update(config_file['model'])

    def create_dir_structure(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        for name, path in vars(self.paths).items():
            # exclude all paths to files
            if len(path.split('.')) == 1:
                if not os.path.exists(path):
                    os.makedirs(path)

    def add_system_paths(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        sys.path.insert(0, self.paths.package)
        #sys.path.insert(0, os.path.join(self.paths.package, 'utils'))


def chunks(lst: object, n: object) -> object:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_configurations(root_dir, object_selected, task_variant, task, prefix, num_episodes=20):
    os.makedirs(root_dir, exist_ok=True)
    save_dir_template = os.path.join(root_dir, f"{prefix}_{task}_{object_selected}")
    for seed in range(num_episodes):
        save_dir = save_dir_template + f"_{task_variant}"+f"_seed{seed:03d}"
        yield object_selected, seed, save_dir

def get_demonstrations(data_dir, object_selected, task_variant, task, file_prefix: str = "demo", num_episodes: int = 20):
    demo_cfgs = get_configurations(root_dir=data_dir, prefix=file_prefix, num_episodes=num_episodes, object_selected=object_selected, task_variant=task_variant, task=task)
    recordings = list()
    for _, demo_seed, demo_dir in demo_cfgs:
        recordings.append(demo_dir)
    return recordings


# Some helper functions to load the data from numpy format
def get_image(demo_dir, frame_index, depth=False):
    arr = np.load(os.path.join(demo_dir, f"frame_{frame_index:06d}.npz"))
    rgb_gripper = arr["rgb_gripper"]
    return rgb_gripper


def get_reward(demo_dir):
    frame_names = sorted(glob(f"{demo_dir}/frame_*.npz"))
    rew = np.load(frame_names[-1])["rew"].item()
    return rew


def get_len(demo_dir):
    frame_names = sorted(glob(f"{demo_dir}/frame_*.npz"))
    return len(frame_names)


def get_info(demo_dir, frame_index):
    arr = np.load(os.path.join(demo_dir, f"frame_{frame_index:06d}.npz"), allow_pickle=True)
    return arr["info"].item()




if __name__ == "__main__":
    data_dir = "/tmp/flow_dataset"
    task = "shape_sorting"
    object_selected = "oval"  # trapeze, oval, semicricle
    task_variant = "rP"  # rotation plus (+-pi)

    export_dir = "/tmp/flow_dataset_contrastive/"


    demo_cfgs = get_configurations(prefix="demo")
    recordings = []
    for _, demo_seed, demo_dir in demo_cfgs:
        recordings.append(demo_dir)

    print("Number of recordings:", len(recordings))
    print("first", recordings[0])
    print("last ", recordings[-1])

    all_recordings = sorted(glob(f"{root_dir}/demo_*"))
    print(len(all_recordings))

    # Loop through demonstrations and plot successive frames
    for demo_idx in range(len(recordings)):
        # loop through frames
        for frame_index in range(get_len(recordings[demo_idx])):
            # get image
            image = get_image(recordings[demo_idx], frame_index)
            # set up matplotlip plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(image)
            plt.show()