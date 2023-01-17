import os
import copy
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor

import os, psutil
import argparse

from torch.nn import MarginRankingLoss
from torch_geometric.utils import batched_negative_sampling, structured_negative_sampling

import torch_geometric.data
import torch_geometric.data.batch
from torch_geometric.utils import coalesce, degree, remove_self_loops

from utils import ParamLib
import gnn
from data import DisjDemoGraphDataset



        


class Trainer():

    def __init__(self, params, model, dataloader_train, dataloader_test, dataloader_trainoverfit, optimizer):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataloader_trainoverfit = dataloader_trainoverfit
        self.params = params
        self.optimizer = optimizer
        self.edge_criterion = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0

        # plt.ion()  # turns on interactive mode
        self.figure, self.axarr = plt.subplots(1, 2)

    def batched_structured_negative_sampling(self, edge_index, batch):
        """Batched structured negative sampling.

        Args:
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): The batch tensor specifiying the batch idx for every node.

        :rtype: :class:`LongTensor
        """

        print(edge_index.shape)
        split = degree(batch[edge_index[0]], dtype=torch.long).tolist()
        edge_indices = torch.split(edge_index, split, dim=1)

        num_src = degree(batch, dtype=torch.long)
        cumsum = torch.cat([batch.new_zeros(1), num_src.cumsum(0)[:-1]])

        neg_edge_indices = []
        for i, edge_index_batch in enumerate(edge_indices):
            edge_index_batch = edge_index_batch - cumsum[i]

            print(edge_index_batch.shape)
            neg_edges_i, neg_edges_j, neg_edges_k = structured_negative_sampling(edge_index=edge_index_batch,
                                                                                 contains_neg_self_loops=False)
            neg_edges_i, neg_edges_j, neg_edges_k = neg_edges_i.view(-1,1), neg_edges_j.view(-1,1), neg_edges_k.view(-1,1)
            neg_edge_index = torch.cat([neg_edges_i, neg_edges_k], dim=1).T

            neg_edge_index += cumsum[i]
            neg_edge_indices.append(neg_edge_index)

        return torch.cat(neg_edge_indices, dim=1)



    def train(self, epoch):

        self.model.train()
        print('Training')

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):
            self.optimizer.zero_grad()

            data = data.to(self.params.model.device)

            out_edge_attr, node_cos_sim = self.model(data.x, data.edge_index)
            # Get all edge features for positive edges
            out_edge_attr_pos = torch.index_select(out_edge_attr, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask)) 
            out_edge_attr_neg, node_cos_sim_neg = self.model(data.x, data.neg_edge_index)

            time_i, time_j = data.node_times[data.pos_edge_index[0,:]], data.node_times[data.pos_edge_index[1,:]]
            time_k = data.node_times[data.neg_edge_index[1,:]]

            time_diff_pos = torch.abs(time_j - time_i)
            time_diff_neg = torch.abs(time_k - time_i)

            ssl_edge_label = torch.ones_like(time_diff_pos)
            ssl_edge_label[time_diff_neg <= time_diff_pos] = -1

            try:
                loss_dict = {
                    'sim_loss': torch.nn.MarginRankingLoss(margin=0.0)(out_edge_attr_pos, out_edge_attr_neg, ssl_edge_label),
                }
            except Exception as e:
                print(e)
                continue

            loss = sum(loss_dict.values())
            tqdm.write(loss)
            loss.backward()
            self.optimizer.step()

            self.total_step += 1

    def eval(self, epoch, split='test'):

        self.model.eval()
        print('Evaluating on {}'.format(split))

        if split == 'test':
            dataloader = self.dataloader_test
        elif split == 'trainoverfit':
            dataloader = self.dataloader_trainoverfit

        dataloader_progress = tqdm(dataloader, desc='Evaluating on {}'.format(split))

        for i_val, data in enumerate(dataloader_progress):

            with torch.no_grad():
                out = self.model(data)
                out_i, out_j = out[data.edge_index[0, :]], out[data.edge_index[1, :]]

            # loss and optim
            try:
                loss_dict = {
                    'appear_loss': torch.nn.MarginRankingLoss(margin=0.0)(out_i, out_j),
                }
            except Exception as e:
                print(e)
                continue

            loss = sum(loss_dict.values())

def main():
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    # Namespace-specific arguments (namespace: model)
    parser.add_argument('--lr', type=str, help='model path')
    parser.add_argument('--epochs', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    print("Batch size summed over all GPUs: ", params.model.batch_size)

    # if not params.main.disable_wandb:
    #     wandb.login()
    #     wandb.init(
    #         entity='martinbchnr',
    #         project=params.main.project,
    #         notes='v1',
    #         settings=wandb.Settings(start_method="fork"),
    #     )
    #     wandb.config.update(params.paths)
    #     wandb.config.update(params.model)
    #     wandb.config.update(params.preprocessing)


    model = gnn.DisjGNN()
    model = model.to(params.model.device)

    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # define own collator that skips bad samples
    train_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset,
                              params.paths.config_name)
    # test_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "test",
    #                          params.paths.config_name)

    dataset_train = DisjDemoGraphDataset(path=train_path, split="train", split_idx=4000)
    # dataset_test = DisjDemoGraphDataset(path=train_path, split="test", split_idx=4000)

    dataloader_obj = torch_geometric.loader.DataLoader
    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    # dataloader_test = dataloader_obj(dataset_test,
    #                                  batch_size=1,
    #                                  num_workers=1,
    #                                  shuffle=False)
    # dataloader_trainoverfit = dataloader_obj(dataset_train,
    #                                          batch_size=1,
    #                                          num_workers=1,
    #                                          shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_train, dataloader_train, optimizer)

    for epoch in range(params.model.num_epochs):
        trainer.train(epoch)

        # if not params.main.disable_wandb:
        #     wandb_run_name = wandb.run.name
        #
        #     fname = 'lane_mp/lanemp_{}_{:03d}.pth'.format(wandb_run_name, epoch)
        #
        #     # save checkpoint locally and in wandb
        #     torch.save(model.state_dict(), params.paths.checkpoints + fname)
        #     wandb.save(params.paths.home + fname)
        #
        # # Evaluate
        # trainer.eval(epoch, split='test', log_images=True)

if __name__ == '__main__':
    main()



