import torch_geometric.data
from abc import ABC
import json
from glob import glob
import torch



class DisjDemoGraphDataset(torch_geometric.data.Dataset, ABC):

    def __init__(self, path, split, split_idx):
        super(DisjDemoGraphDataset, self).__init__()

        self.node_feats_files = []
        self.node_times_files = []
        self.edges_files = []
        self.pos_edges_files = []
        self.edge_time_delta_files = []
        self.edge_pos_diff_files = []
        self.edge_rot_diff_files = []
        self.node2idxframe_files = []
        self.demoframe2node_idx_files = []

        self.node_feats_files.extend(glob(path + '*/*-node-feats.pth'))
        self.node_times_files.extend(glob(path + '*/*-node-times.pth'))
        self.edges_files.extend(glob(path + '*/*-edge-index.pth'))
        self.pos_edges_files.extend(glob(path + '*/*-pos-edges.pth'))
        self.edge_time_delta_files.extend(glob(path + '*/*-edge-time-delta.pth'))
        self.edge_pos_diff_files.extend(glob(path + '*/*-edge-pos-diff.pth'))
        self.edge_rot_diff_files.extend(glob(path + '*/*-edge-rot-diff.pth'))
        self.node2idxframe_files.extend(glob(path + '*/*-node_idx2frame.json'))
        self.demoframe2node_idx_files.extend(glob(path + '*/*-demoframe2node_idx.json'))

        self.split = split
        self.split_idx = split_idx

        self.node_feats_files = sorted(self.node_feats_files)
        self.node_times_files = sorted(self.node_times_files)
        self.edges_files = sorted(self.edges_files)
        self.pos_edges_files = sorted(self.pos_edges_files)
        self.edge_time_delta_files = sorted(self.edge_time_delta_files)
        self.edge_pos_diff_files = sorted(self.edge_pos_diff_files)
        self.edge_rot_diff_files = sorted(self.edge_rot_diff_files)
        self.node2idxframe_files = sorted(self.node2idxframe_files)
        self.demoframe2node_idx_files = sorted(self.demoframe2node_idx_files)

        pop_items = []
        # Remove files based on split_idx
        for i in range(len(self.node_feats_files)):
            if split == 'train':
                sample_no = int(self.node_feats_files[i].split('/')[-1].split('-')[0])
                if sample_no > self.split_idx:
                    pop_items.append(i)
            if split == 'test':
                if int(self.node_feats_files[i].split('/')[-1].split('-')[0]) <= self.split_idx:
                    pop_items.append(i)

        for index in sorted(pop_items, reverse=True):
            del self.node_feats_files[index]
            del self.node_times_files[index]
            del self.edges_files[index]
            del self.pos_edges_files[index]
            del self.edge_time_delta_files[index]
            del self.edge_pos_diff_files[index]
            del self.edge_rot_diff_files[index]
            del self.node2idxframe_files[index]
            del self.demoframe2node_idx_files[index]

        print("Found {} samples in path {}".format(len(self.node_feats_files), path))

    def __len__(self):
        return len(self.node_feats_files)

    def __getitem__(self, index):
        # Return reduced data object if the index is in the index_filter (to save time)

        node_feats = torch.load(self.node_feats_files[index])
        node_times = torch.load(self.node_times_files[index])
        edges = torch.load(self.edges_files[index])
        pos_edges = torch.load(self.pos_edges_files[index])
        edge_time_delta = torch.load(self.edge_time_delta_files[index])
        edge_pos_diff = torch.load(self.edge_pos_diff_files[index])
        edge_rot_diff = torch.load(self.edge_rot_diff_files[index])


        #print(edges.shape, pos_edges.shape, edge_time_delta.shape)
        # load json
        with open(self.node2idxframe_files[index]) as json_file:
            node_idx2frame = json.load(json_file)

        with open(self.demoframe2node_idx_files[index]) as json_file:
            demoframe2node_idx = json.load(json_file)

        data = torch_geometric.data.Data(x=node_feats.reshape(node_feats.shape[0], -1),
                            edge_index=edges.t(),
                            pos_edge_mask=pos_edges.reshape(-1).bool(),
                            edge_time_delta=edge_time_delta.reshape(-1),
                            node_times=node_times.reshape(node_feats.shape[0]),
                            edge_pos_diff=edge_pos_diff.reshape(-1),
                            edge_rot_diff=edge_rot_diff.reshape(-1),
                            # node_idx2frame=node_idx2frame,
                            # demoframe2node_idx=demoframe2node_idx,
                            )

        # Apply structured_negative_sampling per batch
        data.pos_edge_index = torch.index_select(data.edge_index, 1, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
        neg_edges = torch_geometric.utils.structured_negative_sampling(data.pos_edge_index, num_nodes=data.x.shape[0])
        data.neg_edge_index = torch.stack([neg_edges[0], neg_edges[2]])

        return data


class CommonObjectsV2GraphDataset(torch_geometric.data.Dataset, ABC):

    def __init__(self, path, split, split_idx):
        super(CommonObjectsV2GraphDataset, self).__init__()

        self.node_feats_files = []
        self.node_times_files = []
        self.edges_files = []
        self.pos_edges_files = []
        self.edge_time_delta_files = []
        self.node2idxframe_files = []
        self.demoframe2node_idx_files = []

        self.node_feats_files.extend(glob(path + '/*-node-feats.pth'))
        self.node_times_files.extend(glob(path + '/*-node-times.pth'))
        self.edges_files.extend(glob(path + '/*-edge-index.pth'))
        self.pos_edges_files.extend(glob(path + '/*-pos-edges.pth'))
        self.edge_time_delta_files.extend(glob(path + '/*-edge-time-delta.pth'))
        self.node2idxframe_files.extend(glob(path + '/*-node_idx2frame.json'))
        self.demoframe2node_idx_files.extend(glob(path + '/*-demoframe2node_idx.json'))

        self.split = split
        self.split_idx = split_idx


        if split == "train":
            a = 0
            b = split_idx
        elif split == "test":
            a = split_idx
            b = len(self.node_feats_files)
        else:
            a, b = 0, len(self.node_feats_files)

        a = 0
        b = -1

        self.node_feats_files = sorted(self.node_feats_files)[a:b]
        self.node_times_files = sorted(self.node_times_files)[a:b]
        self.edges_files = sorted(self.edges_files)[a:b]
        self.pos_edges_files = sorted(self.pos_edges_files)[a:b]
        self.edge_time_delta_files = sorted(self.edge_time_delta_files)[a:b]
        self.node2idxframe_files = sorted(self.node2idxframe_files)[a:b]
        self.demoframe2node_idx_files = sorted(self.demoframe2node_idx_files)[a:b]

        print("Found {} samples in path {}".format(len(self.node_feats_files), path))

    def __len__(self):
        return len(self.node_feats_files)

    def __getitem__(self, index):
        # Return reduced data object if the index is in the index_filter (to save time)

        node_feats = torch.load(self.node_feats_files[index])
        node_times = torch.load(self.node_times_files[index])
        edges = torch.load(self.edges_files[index])
        pos_edges = torch.load(self.pos_edges_files[index])
        edge_time_delta = torch.load(self.edge_time_delta_files[index])

        #print(edges.shape, pos_edges.shape, edge_time_delta.shape)
        # load json
        with open(self.node2idxframe_files[index]) as json_file:
            node_idx2frame = json.load(json_file)

        with open(self.demoframe2node_idx_files[index]) as json_file:
            demoframe2node_idx = json.load(json_file)

        data = torch_geometric.data.Data(x=node_feats.reshape(node_feats.shape[0], -1),
                            edge_index=edges.t(),
                            pos_edge_mask=pos_edges.reshape(-1).bool(),
                            edge_time_delta=edge_time_delta.reshape(-1),
                            node_times=node_times.reshape(node_feats.shape[0]),
                            # node_idx2frame=node_idx2frame,
                            # demoframe2node_idx=demoframe2node_idx,
                            )

        # Apply structured_negative_sampling per batch
        data.pos_edge_index = torch.index_select(data.edge_index, 1, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
        neg_edges = torch_geometric.utils.structured_negative_sampling(data.pos_edge_index, num_nodes=data.x.shape[0])
        data.neg_edge_index = torch.stack([neg_edges[0], neg_edges[2]])

        return data
