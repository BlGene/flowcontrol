import torch_geometric.data
from abc import ABC
import json
import os
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import ToTensor


class DemoData(torch_geometric.data.Dataset, ABC):

    def __init__(self, path, split, split_idx, num_demos, part):
        super(DemoData, self).__init__()

        self.split_idx = split_idx
        self.part = part

        self.demo_img_files = []
        self.live_img_files = []
        self.mask_files = []

        self.demo_img_files.extend(glob(path + '*/demo_imgs/*_' + self.part + '.jpg'))
        self.live_img_files.extend(glob(path + '*/live_imgs/*_' + self.part + '.jpg'))
        self.mask_files.extend(glob(path + '*/masks/*_' + self.part + '.jpg'))

        self.demo_img_files = sorted(self.demo_img_files)[0:num_demos]
        self.live_img_files = sorted(self.live_img_files)[0:num_demos]
        self.mask_files = sorted(self.mask_files)[0:num_demos]

        if part is not "*":
            pop_items = []
            # Remove files based on split_idx
            for i in range(len(self.demo_img_files)):
                if split == 'train':
                    sample_no = int(self.demo_img_files[i].split('/')[-1].split('_')[0])
                    if sample_no > self.split_idx:
                        pop_items.append(i)
                if split == 'test':
                    if int(self.demo_img_files[i].split('/')[-1].split('_')[0]) <= self.split_idx:
                        pop_items.append(i)

            for index in sorted(pop_items, reverse=True):
                del self.demo_img_files[index]
                del self.live_img_files[index]
                del self.mask_files[index]

        with open(os.path.join(path, "trapeze_sim/rewards.json")) as json_file:
            self.rewards = json.load(json_file)

        print("Found {} samples in path {}".format(len(self.demo_img_files), path))

    def __len__(self):
        return len(self.demo_img_files)

    def __getitem__(self, index):
        # Return reduced data object if the index is in the index_filter (to save time)
        
        # Multi-part scenario: Based on filename extract the corresp. demo idx and source the reward from the json file
        demo_token = self.demo_img_files[index].split("/")[-1].split("_")[0]

        mask_image = self.mask_files[index]
        mask_img = Image.open(mask_image).convert("L")
        mask_img = ToTensor()(mask_img).view(-1, 1, 256, 256)

        jpg_demo_image = self.demo_img_files[index]
        demo_img = Image.open(jpg_demo_image)
        demo_img = ToTensor()(demo_img).view(-1, 3, 256, 256)
        demo_img = torch.cat([demo_img, mask_img], dim=1)

        jpg_live_image = self.live_img_files[index]
        live_img = Image.open(jpg_live_image)
        live_img = ToTensor()(live_img).view(-1, 3, 256, 256)
        live_img = torch.cat([live_img, mask_img], dim=1)

        reward = torch.tensor(self.rewards[demo_token])

        x = torch.cat([live_img, demo_img], dim=0)

        edges = list()
        edges.append((0, 1))
        edges = torch.tensor(edges).t()

        data = torch_geometric.data.Data(x=x, edge_index=edges, reward=reward)
        
        return data


class DisjBidirDemoGraphDataset(torch_geometric.data.Dataset, ABC):

    def __init__(self, path, split, split_idx):
        super(DisjBidirDemoGraphDataset, self).__init__()

        self.node_feats_files = []
        self.node_times_files = []
        self.edges_files = []
        self.triplet_edge_files = []
        self.edge_time_delta_files = []
        self.edge_pos_diff_files = []
        self.edge_rot_diff_files = []
        self.node2idxframe_files = []
        self.demoframe2node_idx_files = []

        self.node_feats_files.extend(glob(path + '*/*-node-feats.pth'))
        self.node_times_files.extend(glob(path + '*/*-node-times.pth'))
        self.edges_files.extend(glob(path + '*/*-edge-index.pth'))
        self.triplet_edge_files.extend(glob(path + '*/*-triplet-edges.pth'))
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
        self.triplet_edge_files = sorted(self.triplet_edge_files)
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
            del self.triplet_edge_files[index]
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
        triplet_edges = torch.load(self.triplet_edge_files[index])
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
                            edge_time_delta=edge_time_delta.reshape(-1),
                            node_times=node_times.reshape(node_feats.shape[0]),
                            edge_pos_diff=edge_pos_diff.reshape(-1),
                            edge_rot_diff=edge_rot_diff.reshape(-1),
                            )

        # Get triplet edges
        pos_edges = list()
        neg_edges = list()

        pos_edge_idcs = torch.empty((0), dtype=torch.bool)
        neg_edge_idcs = torch.empty((0), dtype=torch.bool)
        for triplet_idx, triplet in enumerate(triplet_edges):
            i,j,k = triplet
            pos_edges.append((i.item(),j.item()))
            neg_edges.append((i.item(),k.item()))

            pos_edge_idcs = torch.cat((pos_edge_idcs, torch.where((data.edge_index[0] == i) & (data.edge_index[1] == j))[0]), dim=0)
            neg_edge_idcs = torch.cat((neg_edge_idcs, torch.where((data.edge_index[0] == i) & (data.edge_index[1] == k))[0]), dim=0)

        data.pos_edge_index = torch.tensor(pos_edges).t()
        data.neg_edge_index = torch.tensor(neg_edges).t()
        data.pos_edge_idcs = pos_edge_idcs
        data.neg_edge_idcs = neg_edge_idcs
        
        return data


class DisjDemoGraphDataset(torch_geometric.data.Dataset, ABC):

    def __init__(self, path, split, split_idx):
        super(DisjDemoGraphDataset, self).__init__()

        self.node_feats_files = []
        self.node_times_files = []
        self.edges_files = []
        self.pos_edges_files = []
        self.triplet_edge_files = []
        self.edge_time_delta_files = []
        self.edge_pos_diff_files = []
        self.edge_rot_diff_files = []
        self.node2idxframe_files = []
        self.demoframe2node_idx_files = []

        self.node_feats_files.extend(glob(path + '*/*-node-feats.pth'))
        self.node_times_files.extend(glob(path + '*/*-node-times.pth'))
        self.edges_files.extend(glob(path + '*/*-edge-index.pth'))
        self.pos_edges_files.extend(glob(path + '*/*-pos-edges.pth'))
        self.triplet_edge_files.extend(glob(path + '*/*-triplet-edges.pth'))
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
        self.triplet_edge_files = sorted(self.triplet_edge_files)
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
            del self.triplet_edge_files[index]
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
        triplet_edges = torch.load(self.triplet_edge_files[index])
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
                            #pos_edge_mask=pos_edges.reshape(-1).bool(),
                            edge_time_delta=edge_time_delta.reshape(-1),
                            node_times=node_times.reshape(node_feats.shape[0]),
                            edge_pos_diff=edge_pos_diff.reshape(-1),
                            edge_rot_diff=edge_rot_diff.reshape(-1),
                            # node_idx2frame=node_idx2frame,
                            # demoframe2node_idx=demoframe2node_idx,
                            )

        # Apply structured_negative_sampling per batch
        #data.pos_edge_index = torch.index_select(data.edge_index, 1, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
        # neg_edges = torch_geometric.utils.structured_negative_sampling(data.pos_edge_index, num_nodes=data.x.shape[0])
        # data.neg_edge_index = torch.stack([neg_edges[0], neg_edges[2]])

        # Get negative edges
        pos_edges = list()
        neg_edges = list()
        pos_edge_idcs = torch.empty((0), dtype=torch.bool)
        neg_edge_idcs = torch.empty((0), dtype=torch.bool)
        for triplet_idx, triplet in enumerate(triplet_edges):
            i,j,k = triplet
            pos_edges.append((i.item(),j.item()))
            neg_edges.append((i.item(),k.item()))
            pos_edge_idcs = torch.cat((pos_edge_idcs, torch.where((data.edge_index[0] == i) & (data.edge_index[1] == j))[0]), dim=0)
            neg_edge_idcs = torch.cat((neg_edge_idcs, torch.where((data.edge_index[0] == i) & (data.edge_index[1] == k))[0]), dim=0)

        data.pos_edge_index = torch.tensor(pos_edges).t()
        data.neg_edge_index = torch.tensor(neg_edges).t()
        data.pos_edge_idcs = pos_edge_idcs
        data.neg_edge_idcs = neg_edge_idcs
        
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
