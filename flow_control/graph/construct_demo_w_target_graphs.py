import os
import copy
import json
import argparse
from glob import glob
from collections import defaultdict
import ray
from ray.util.multiprocessing import Pool


import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from flow_control.graph.utils import ParamLib, get_keyframe_info, get_len, get_image, get_depth, get_pose, get_demonstrations, get_mask, get_info, chunks
from robot_io.utils.utils import depth_img_to_uint16
from flow_control.utils_coords import get_pos_orn_diff


def create_single_demo_graph(recordings: list, demo_idx: int):

    try:
        keyframe_info = get_keyframe_info(recordings[demo_idx])

        edges = list()
        triplet_edges = list()
        edge_time_delta = list()
        node_feats = torch.empty((0, 256, 256, 4))
        pose_feats = torch.empty((0, 4, 4))
        node_times = list()

        edge_pos_diff = list()
        edge_rot_diff = list()

        node_idx = 0
        node_idx2frame = defaultdict()

        print("demo_idx: ", demo_idx)

        cond_node_idx = None
    
        # Construct node features by looping through frames
        curr_demo_node_idcs = list()
        for frame_idx in range(get_len(recordings[demo_idx])):
            if str(frame_idx) in keyframe_info.keys():
                curr_image = torch.tensor(get_image(recordings[demo_idx], frame_idx)).unsqueeze(0)
                depth = get_depth(recordings[demo_idx], frame_idx) # depth_img_to_uint16(, max_depth=4)
                curr_depth = torch.tensor(depth.astype('float32')).unsqueeze(0).unsqueeze(-1)
                rgbd = torch.cat([curr_image, curr_depth], dim=-1)

                pose = get_pose(recordings[demo_idx], frame_idx)

                node_idx2frame[node_idx] = (demo_idx, frame_idx)

                curr_demo_node_idcs.append(node_idx)
                node_feats = torch.cat([node_feats, rgbd], dim=0)
                pose_feats = torch.cat([pose_feats, torch.tensor(pose).reshape(1, 4, 4)])
                
                node_times.append(node_idx/len(keyframe_info.keys()))
                print(node_idx/len(keyframe_info.keys()))
                node_idx += 1

                if get_info(recordings[demo_idx], frame_idx)["wp_name"] == "grasp":
                    cond_node_idx = node_idx-1
                plt.imshow(get_image(recordings[demo_idx], frame_idx))
                plt.imshow(get_mask(recordings[demo_idx], frame_idx), alpha=0.5)
                plt.savefig("/home/buechner/servoing/flowcontrol/flow_control/graph/prints/cond_{}_{}_{}.png".format(demo_idx,frame_idx, get_info(recordings[demo_idx], frame_idx)["wp_name"]))

        # stable_mask = None
        # contains_start = False
        # # Go through all keyframes and get the target frame
        # for frame_idx in range(get_len(recordings[demo_idx])):
        #     info = get_info(recordings[demo_idx], frame_idx)
        #     if info["wp_name"] == "start" and contains_start == False:
        #         contains_start = True
        #     if info["wp_name"] == "grasp_close":
        #         stable_mask = get_mask(recordings[demo_idx], frame_idx)
        #     if stable_mask is not None:
        #         cond_mask = stable_mask
            
        #     if info["wp_name"] == "insert_over-open" and get_info(recordings[demo_idx], frame_idx+1)["wp_name"] == "insert_over-open_open" and contains_start:

        #         cond_image = get_image(recordings[demo_idx], frame_idx)

        #         # DEBUGGING Plot cond_image and cond_mask on top
        #         plt.imshow(cond_image)
        #         plt.imshow(cond_mask, alpha=0.5)
        #         plt.savefig("/home/buechner/servoing/flowcontrol/flow_control/graph/prints/cond_{}_{}_{}.png".format(demo_idx,frame_idx, info["wp_name"]))

        #         cond_image = torch.tensor(get_image(recordings[demo_idx], frame_idx)).unsqueeze(0)
        #         cond_depth = get_depth(recordings[demo_idx], frame_idx) # depth_img_to_uint16(, max_depth=4)
        #         cond_depth = torch.tensor(depth.astype('float32')).unsqueeze(0).unsqueeze(-1)
        #         cond_mask = torch.tensor(cond_mask.astype('float32')).unsqueeze(0).unsqueeze(-1)
                

        #         cond_rgbd_mask = torch.cat([cond_image, cond_mask], dim=-1)


        # Construct all edges plus edge features excluding self-loops
        for i in curr_demo_node_idcs:
            for j in curr_demo_node_idcs:
                if i != j: # and j > i:
                    edges.append((i,j))
                    edge_time_delta.append(node_idx2frame[j][1] - node_idx2frame[i][1]) # can also be negative

                    pos_diff, rot_diff = get_pos_orn_diff(get_pose(recordings[demo_idx], node_idx2frame[i][1]),
                                                        get_pose(recordings[demo_idx], node_idx2frame[j][1]))

                    edge_pos_diff.append(pos_diff)
                    edge_rot_diff.append(rot_diff)
            # right diagonal forward-directed edges
            if i <= max(curr_demo_node_idcs)-2:
                triplet_edges.append((i, i+1, i+2))
            # left diagonal backward-directed edges
            if i >= 2 and params.preprocessing.bidir:
                triplet_edges.append((i, i-1, i-2))

        # reverse node_idx2frame in one line
        demoframe2node_idx = {str(v): k for k, v in node_idx2frame.items()}

        edges = torch.tensor(edges).long()
        triplet_edges = torch.tensor(triplet_edges).long()
        edge_time_delta = torch.tensor(edge_time_delta).long()
        edge_pos_diff = torch.tensor(edge_pos_diff).float()
        edge_rot_diff = torch.tensor(edge_rot_diff).float()
        node_times = torch.tensor(node_times)
        cond_frame = torch.tensor(cond_rgbd_mask).unsqueeze(0)

        print(edges.shape, edge_time_delta.shape)

        assert edge_pos_diff.shape[0] == edges.shape[0]
        torch.save(node_feats, os.path.join(export_dir, str(demo_idx)) + '-node-feats.pth')
        torch.save(pose_feats, os.path.join(export_dir, str(demo_idx)) + '-pose-feats.pth') 
        torch.save(node_times, os.path.join(export_dir, str(demo_idx)) + '-node-times.pth')
        torch.save(edges, os.path.join(export_dir, str(demo_idx)) + '-edge-index.pth')
        torch.save(triplet_edges, os.path.join(export_dir, str(demo_idx)) + '-triplet-edges.pth')
        torch.save(edge_time_delta, os.path.join(export_dir, str(demo_idx)) + '-edge-time-delta.pth')
        torch.save(edge_pos_diff, os.path.join(export_dir, str(demo_idx)) + '-edge-pos-diff.pth')
        torch.save(edge_rot_diff, os.path.join(export_dir, str(demo_idx)) + '-edge-rot-diff.pth')
        torch.save(cond_frame, os.path.join(export_dir, str(demo_idx)) + '-cond-frame.pth')

        print("saved to: ", os.path.join(export_dir, str(demo_idx)+'-*.pth'))

        with open(os.path.join(export_dir, str(demo_idx)) + '-node_idx2frame.json', 'w') as outfile:
            json.dump(node_idx2frame, outfile)

        with open(os.path.join(export_dir, str(demo_idx)) + '-demoframe2node_idx.json', 'w') as outfile:
            json.dump(demoframe2node_idx, outfile)
    except Exception as e:
        print(e)
        return False

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
                                    num_episodes=params.preprocessing.num_episodes,
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

    # LEAVE FOR DEBUGGING
    process_chunk(chunk_data[-1])

    # ray.init(num_cpus=params.preprocessing.workers,
    #          include_dashboard=False,
    #          _system_config={"automatic_object_spilling_enabled": True,
    #                          "object_spilling_config": json.dumps(
    #                              {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # pool = Pool()
    # pool.map(process_chunk, [data for data in chunk_data])
