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
from PIL import Image

from flow_control.graph.utils import ParamLib, chunks, get_co3d_demos, get_image_co3d, get_co3d_demo_tokens, get_frame_idx_co3d


def create_single_demo_graph(seq_img: dict, seq_dep: dict, demo_token: str):

    edges = list()
    pos_edges = list()
    edge_time_delta = list()
    node_feats = torch.empty((0, 256, 256, 3))
    node_times = list()

    node_idx = 0
    node_idx2frame = defaultdict()

    print("demo_idx: ", demo_token)

    obj_token, seq_token = demo_token.split("/")

    # loop through frames
    curr_demo_node_idcs = list()
    for frame_idx, frame_file in enumerate(seq_img[obj_token][seq_token]):
        curr_image = torch.tensor(get_image_co3d(frame_file)).unsqueeze(0)

        node_idx2frame[node_idx] = (demo_token, frame_idx)
        curr_demo_node_idcs.append(node_idx)
        node_feats = torch.cat([node_feats, curr_image], dim=0)
        node_times.append(frame_idx)
        node_idx += 1

    # get all combinations of nodes in the current demo
    for i in range(len(curr_demo_node_idcs)):
        for j in range(len(curr_demo_node_idcs)):
            if i != j and j > i:
                edges.append((curr_demo_node_idcs[i], curr_demo_node_idcs[j]))
                edge_time_delta.append(node_idx2frame[j][1] - node_idx2frame[i][1])

                if j-i == 1:
                    pos_edges.append(1)
                else:
                    pos_edges.append(0)
    # reverse node_idx2frame in one line
    demoframe2node_idx = {str(v): k for k, v in node_idx2frame.items()}

    edges = torch.tensor(edges).long()
    pos_edges = torch.tensor(pos_edges).long()
    edge_time_delta = torch.tensor(edge_time_delta).long()
    node_times = torch.tensor(node_times).long()

    print(edges.shape, pos_edges.shape, edge_time_delta.shape)

    torch.save(node_feats, export_dir + "-".join([obj_token, seq_token]) + '-node-feats.pth')
    torch.save(node_times, export_dir + "-".join([obj_token, seq_token]) + '-node-times.pth')
    torch.save(edges, export_dir + "-".join([obj_token, seq_token]) + '-edge-index.pth')
    torch.save(pos_edges, export_dir + "-".join([obj_token, seq_token]) + '-pos-edges.pth')
    torch.save(edge_time_delta, export_dir + "-".join([obj_token, seq_token]) + '-edge-time-delta.pth')
    print("saved to: ", export_dir + "-".join([obj_token, seq_token]) + '-*.pth')

    with open(export_dir + "-".join([obj_token, seq_token]) + '-node_idx2frame.json', 'w') as outfile:
        json.dump(node_idx2frame, outfile)

    with open(export_dir + "-".join([obj_token, seq_token]) + '-demoframe2node_idx.json', 'w') as outfile:
        json.dump(demoframe2node_idx, outfile)

def process_chunk(data):
    # Load the data for this chunk.
    seq_img, seq_dep, demo_token_chunks = data
    for demo_token in demo_token_chunks:
        create_single_demo_graph(seq_img, seq_dep, demo_token)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    parser.add_argument('--workers', type=int, help='define number of workers used for preprocessing')

    opt = parser.parse_args()
    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)

    print("# WORKERS: ", params.preprocessing.workers)


    export_dir = os.path.join(params.paths.dataroot, params.preprocessing.rel_export_dir)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    objects_included = ["/apple/", "/ball/", "/book/", "/bottle/", "/bowl/", "/cup/", "/orange/", "/remote/", "/vase/"]

    seq_img, seq_dep = get_co3d_demos(data_dir=os.path.join(params.paths.dataroot, params.preprocessing.raw_data),
                                        fixed_lag=10,
                                        object_types=objects_included,
                                        )    

    demo_tokens = get_co3d_demo_tokens(data_dir=os.path.join(params.paths.dataroot, params.preprocessing.raw_data),
                                        object_types=objects_included)
    
    # chunking of demo=seed indices
    chunk_size = int(np.ceil(len(demo_tokens) / params.preprocessing.workers))
    demo_idx_chunks = list(chunks(demo_tokens, chunk_size))

    chunk_data = list()
    for idx_chunk in demo_idx_chunks:
        chunk_data.append((seq_img, seq_dep, idx_chunk))

    ray.init(num_cpus=params.preprocessing.workers,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # LEAVE FOR DEBUGGING
    process_chunk(chunk_data[0])
    
    # pool = Pool()
    # pool.map(process_chunk, [data for data in chunk_data])
