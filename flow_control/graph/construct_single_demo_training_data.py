import os
import copy
import json
import argparse
from glob import glob
from collections import defaultdict
import ray
from ray.util.multiprocessing import Pool


import numpy as np
import matplotlib.pyplot as plt
import torch

from flow_control.graph.utils import ParamLib, get_keyframe_info, get_len, get_image, get_demonstrations, chunks


def create_single_demo_graph(recordings: list, demo_idx: int):

    keyframe_info = get_keyframe_info(recordings[demo_idx])

    edges = list()
    pos_edges = torch.empty((0,1))
    edge_time_delta = torch.empty((0,1))
    node_feats = torch.empty((0, 256, 256, 3))
    node_times = torch.empty((0,1))

    node_idx = 0
    node_idx2frame = defaultdict()

    print("demo_idx: ", demo_idx)
    # loop through frames
    curr_demo_node_idcs = list()
    for frame_idx in range(get_len(recordings[demo_idx])):
        if str(frame_idx) in keyframe_info.keys():
            curr_image = torch.tensor(get_image(recordings[demo_idx], frame_idx)).unsqueeze(0)

            node_idx2frame[node_idx] = (demo_idx, frame_idx)

            curr_demo_node_idcs.append(node_idx)
            node_feats = np.vstack((node_feats, curr_image))
            node_times = np.vstack((node_times, np.array([frame_idx])))
            node_idx += 1

    # get all combinations of nodes in the current demo
    for i in range(len(curr_demo_node_idcs)):
        for j in range(len(curr_demo_node_idcs)):
            if i != j and j > i:
                edges.append((curr_demo_node_idcs[i], curr_demo_node_idcs[j]))
                edge_time_delta = np.vstack((edge_time_delta, np.array([node_idx2frame[j][1] - node_idx2frame[i][1]])))

                if j-i == 1:
                    pos_edges = np.vstack((pos_edges, np.array([1])))
                else:
                    pos_edges = np.vstack((pos_edges, np.array([0])))

    # reverse node_idx2frame in one line
    demoframe2node_idx = {str(v): k for k, v in node_idx2frame.items()}

    edges = torch.tensor(edges).long()
    pos_edges = torch.tensor(pos_edges).long()
    edge_time_delta = torch.tensor(edge_time_delta).long()
    node_times = torch.tensor(node_times).long()

    print(edges.shape, pos_edges.shape, edge_time_delta.shape)

    torch.save(node_feats, os.path.join(export_dir, str(demo_idx)) + '-node-feats.pth')
    torch.save(node_times, os.path.join(export_dir, str(demo_idx)) + '-node-times.pth')
    torch.save(edges, os.path.join(export_dir, str(demo_idx)) + '-edge-index.pth')
    torch.save(pos_edges, os.path.join(export_dir, str(demo_idx)) + '-pos-edges.pth')
    torch.save(edge_time_delta, os.path.join(export_dir, str(demo_idx)) + '-edge-time-delta.pth')
    print("saved to: ", os.path.join(export_dir, str(demo_idx)+'-*.pth'))

    with open(os.path.join(export_dir, str(demo_idx)) + '-node_idx2frame.json', 'w') as outfile:
        json.dump(node_idx2frame, outfile)

    with open(os.path.join(export_dir, str(demo_idx)) + '-demoframe2node_idx.json', 'w') as outfile:
        json.dump(demoframe2node_idx, outfile)

def process_chunk(data):
    # Load the data for this chunk.
    recordings, demo_idcs_chunk = data

    for demo_idx in demo_idcs_chunk:
        create_single_demo_graph(recordings, demo_idx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    # Namespace-specific arguments (namespace: preprocessing)
    parser.add_argument('--workers', type=int, help='define number of workers used for preprocessing')

    opt = parser.parse_args()
    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)

    print("# WORKERS: ", params.preprocessing.workers)

    config_str = "demo_" + params.preprocessing.task + "_" + params.preprocessing.object_selected + "_" + params.preprocessing.task_variant
    export_dir = os.path.join(params.paths.dataroot, params.preprocessing.rel_export_dir, config_str)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    recordings = get_demonstrations(data_dir=os.path.join(params.paths.dataroot, params.preprocessing.raw_data),
                                    num_episodes=76,
                                    task=params.preprocessing.task,
                                    object_selected=params.preprocessing.object_selected,
                                    task_variant=params.preprocessing.task_variant,
                                    file_prefix="demo")

    # chunking of demo=seed indices
    demo_idcs = list(range(0, len(recordings)))
    chunk_size = int(np.ceil(len(recordings) / params.preprocessing.workers))
    demo_idx_chunks = list(chunks(demo_idcs, chunk_size))

    chunk_data = list()
    for idx_chunk in demo_idx_chunks:
        chunk_data.append((recordings, idx_chunk))

    ray.init(num_cpus=params.preprocessing.workers,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # LEAVE FOR DEBUGGING
    # process_chunk(chunk_data[0])
    
    pool = Pool()
    pool.map(process_chunk, [data for data in chunk_data])
