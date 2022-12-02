import os
import copy
import json
import logging
from glob import glob

import numpy as np
import matplotlib.pyplot as plt



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