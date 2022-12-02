import os
import copy
import json
import logging
from glob import glob
from collections import defaultdict
import ray
from ray.util.multiprocessing import Pool


import numpy as np
import matplotlib.pyplot as plt
import torch

import utils


def create_single_demo_graph(recordings: list, demo_idx: int):

    edges = list()
    edge_time_delta = list()
    node_feats = torch.empty((0, 256, 256, 3))

    node_idx = 0
    node_idx2frame = defaultdict()

    print("demo_idx: ", demo_idx)
    # loop through frames
    curr_demo_node_idcs = list()
    for frame_idx in range(utils.get_len(recordings[demo_idx])):
        curr_image = torch.tensor(utils.get_image(recordings[demo_idx], frame_idx)).unsqueeze(0)

        node_idx2frame[node_idx] = (demo_idx, frame_idx)

        curr_demo_node_idcs.append(node_idx)
        node_feats = np.vstack((node_feats, curr_image))
        node_idx += 1

    # get all combinations of nodes in the current demo
    for i in range(len(curr_demo_node_idcs)):
        for j in range(len(curr_demo_node_idcs)):
            if i != j and j > i:
                edges.append((curr_demo_node_idcs[i], curr_demo_node_idcs[j]))
                edge_time_delta.append(node_idx2frame[j][1] - node_idx2frame[i][1])

    # reverse node_idx2frame in one line
    demoframe2node_idx = {str(v): k for k, v in node_idx2frame.items()}

    edges = torch.tensor(edges).long()
    edge_time_delta = torch.tensor(edge_time_delta).long()

    torch.save(node_feats, os.path.join(export_dir, "seed" + str(demo_idx)) + '-node-feats.pth')
    torch.save(edges, os.path.join(export_dir, "seed" + str(demo_idx)) + '-edges.pth')
    torch.save(edge_time_delta, os.path.join(export_dir, "seed" + str(demo_idx)) + '-edge-time-delta.pth')
    print("saved to: ", os.path.join(export_dir, str(demo_idx)))

    with open(os.path.join(export_dir, "seed" + str(demo_idx)) + '_node_idx2frame.json', 'w') as outfile:
        json.dump(node_idx2frame, outfile)

    with open(os.path.join(export_dir, "seed" + str(demo_idx)) + '_demoframe2node_idx.json', 'w') as outfile:
        json.dump(demoframe2node_idx, outfile)

def process_chunk(data):
    # Load the data for this chunk.
    recordings, demo_idcs_chunk = data

    for demo_idx in demo_idcs_chunk:
        create_single_demo_graph(recordings, demo_idx)


if __name__ == "__main__":

    num_cpus = 8

    data_dir = "/tmp/flow_dataset"
    task = "shape_sorting"
    object_selected = "oval"  # trapeze, oval, semicircle
    task_variant = "rP"  # rotation plus (+-pi)

    export_dir = "/tmp/flow_dataset_contrastive/demo_" + task + "_" + object_selected + "_" + task_variant
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    recordings = utils.get_demonstrations(data_dir=data_dir,
                                          num_episodes=1000,
                                          task=task,
                                          object_selected=object_selected,
                                          task_variant=task_variant,
                                          file_prefix="demo")

    # chunking of demo=seed indices
    demo_idcs = list(range(0, len(recordings)))
    chunk_size = int(np.ceil(len(recordings) / num_cpus))
    demo_idx_chunks = list(utils.chunks(demo_idcs, chunk_size))

    chunk_data = list()
    for idx_chunk in demo_idx_chunks:
        chunk_data.append((recordings, idx_chunk))

    ray.init(num_cpus=num_cpus,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    pool = Pool()
    pool.map(process_chunk, [data for data in chunk_data])
