import os
import copy
import json
import logging
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

import utils

export_dir = "/tmp/flow_dataset_contrastive/"

def create_graph_data(recordings: list, demo_idcs: list):

    edges = list()
    edge_time_delta = list()
    node_feats = torch.empty((0, 256, 256, 3))

    node_idx = 0
    node_idx2frame = defaultdict()

    # loop through all demonstrations
    for demo_idx in demo_idcs:
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

    torch.save(node_feats, os.path.join(export_dir, str(demo_idcs)) + '-node-feats.pth')
    torch.save(edges, os.path.join(export_dir, str(demo_idcs)) + '-edges.pth')
    torch.save(edge_time_delta, os.path.join(export_dir, str(demo_idcs)) + '-edge-time-delta.pth')
    print("saved to: ", os.path.join(export_dir, str(demo_idcs)))

    with open(os.path.join(export_dir, str(demo_idcs)) + '_node_idx2frame.json', 'w') as outfile:
        json.dump(node_idx2frame, outfile)

    with open(os.path.join(export_dir, str(demo_idcs)) + '_demoframe2node_idx.json', 'w') as outfile:
        json.dump(demoframe2node_idx, outfile)

if __name__ == "__main__":
    recordings = utils.get_recordings(file_prefix="demo")
    create_graph_data(recordings, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # create_graph_data(recordings, [0, 1])
