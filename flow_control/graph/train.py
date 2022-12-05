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

import os, psutil
import argparse

from torch.nn import MarginRankingLoss


import torch_geometric.data
import torch_geometric.data.batch

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

        it = iter(self.dataloader_train)
        i = 0
        while i < 1:
            i += 1
            self.one_sample_data = next(it)

    def train(self, epoch):

        self.model.train()
        print('Training')

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):


            self.optimizer.zero_grad()

            if self.params.model.dataparallel:
                data = [item.to(self.device) for item in data]
            else:
                data = data.to(self.device)

            # loss and optim
            out_edge_attr = self.model(data)

            torch.empty_like(data.edge_index, dtype=torch.long)
            # loop through all batches
            for i in range(len(data.batch)):
                mask = data.batch == i
                # get all edges in this batch
                # TODO: implement batched negative sampling
                # TODO: implement batched negative sampling
                edge_index = data.edge_index[0, :][]
                for i in range(data.edge_index.shape[1]):

            # loop through all edges in data.edge_index
            for edge_idx, edge in enumerate(data.edge_index):
                if edge[0] == edge[1]:
                    edge[1]

            # Obtain edges be


            try:
                loss_dict = {
                    'appear_loss': torch.nn.MarginRankingLoss(margin=0.0)(out_i, out_j),
                }
            except Exception as e:
                print(e)
                continue

            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()

            # if not self.params.main.disable_wandb:
            #     wandb.log({"train/loss_total": loss.item()})
            #     wandb.log({"train/edge_loss": loss_dict['edge_loss'].item()})
            #     wandb.log({"train/node_loss": loss_dict['node_loss'].item()})
            #     wandb.log({"train/endpoint_loss": loss_dict['endpoint_loss'].item()})

            # text = 'Epoch {} / {} step {} / {}, train loss = {:03f} | Batch time: {:.3f} | Data time: {:.3f}'. \
            #     format(epoch, self.params.model.num_epochs, step + 1, len(self.dataloader_train), loss.item(),
            #            t_end - t_start, 0.0)
            # train_progress.set_description(text)

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

        # if not self.params.main.disable_wandb:
        #     wandb.log(log_dict)
        #     wandb.log(metrics_dict_mean)


def main():
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Namespace-specific arguments (namespace: model)
    parser.add_argument('--lr', type=str, help='model path')
    parser.add_argument('--epochs', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    print("Batch size summed over all GPUs: ", params.model.batch_size)

    if not params.main.disable_wandb:
        wandb.login()
        wandb.init(
            entity='martinbchnr',
            project=params.main.project,
            notes='v1',
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(params.paths)
        wandb.config.update(params.model)
        wandb.config.update(params.preprocessing)


    model = gnn.DisjGNN()
    model = model.to(params.model.device)

    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # define own collator that skips bad samples
    train_path = os.path.join(params.paths.dataroot_ssd, params.paths.rel_dataset, "preprocessed", "train",
                              params.paths.config_name)
    test_path = os.path.join(params.paths.dataroot_ssd, params.paths.rel_dataset, "preprocessed", "test",
                             params.paths.config_name)

    dataset_train = DisjDemoGraphDataset(path=train_path, split="train", split_idx=4000)
    dataset_test = DisjDemoGraphDataset(path=train_path, split="test", split_idx=4000)

    dataloader_obj = torch_geometric.loader.DataLoader
    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    dataloader_test = dataloader_obj(dataset_test,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False)
    dataloader_trainoverfit = dataloader_obj(dataset_train,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False)

    trainer = Trainer(model, dataloader_train, dataloader_test, dataloader_trainoverfit, optimizer)

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



