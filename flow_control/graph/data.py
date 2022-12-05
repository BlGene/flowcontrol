import torch_geometric.data
from abc import ABC
import json
from glob import glob
import torch



class DisjDemoGraphDataset(torch_geometric.data.Dataset, ABC):

    def __init__(self, path):
        super(DisjDemoGraphDataset, self).__init__()

        self.node_feats_files = []
        self.edges_files = []
        self.edge_time_delta_files = []

        self.node_feats_files.extend(glob(path + '/*-node-feats.pth'))
        self.edge_files.extend(glob(path + '/*-edges.pth'))
        self.edge_time_delta_files.extend(glob(path + '/*-edge-time-delta.pth'))

        self.node_feats_files = sorted(self.node_feats_files)
        self.edges_files = sorted(self.edges_files)
        self.edge_time_delta_files = sorted(self.edge_time_delta_files)


        with open(os.path.join(export_dir, "seed" + str(demo_idx)) + '_node_idx2frame.json', 'w') as outfile:
            json.dump(node_idx2frame, outfile)

        with open(os.path.join(export_dir, "seed" + str(demo_idx)) + '_demoframe2node_idx.json', 'w') as outfile:
            json.dump(demoframe2node_idx, outfile)



        print(len(self.node_feats_files))
        print("Found {} samples in path {}".format(len(self.rgb_files), path))


    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        # Return reduced data object if the index is in the index_filter (to save time)

        node_feats = torch.load(self.node_feats_files[index])
        edges = torch.load(self.edges_files[index])
        edge_time_delta_files = torch.load(self.edge_time_delta_files[index])

        data = torch_geometric.data.Data(x=node_feats,
                                         edge_index=edges.t().contiguous(),
                                         edge_time_delta=edge_time_delta_files,
                                         )

        return data
