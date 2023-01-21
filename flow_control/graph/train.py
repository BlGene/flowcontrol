import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

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


import torch_geometric.data
import torch_geometric.data.batch

from torchmetrics.functional.classification.average_precision import average_precision
from torchmetrics.functional.classification.precision_recall import recall


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


    def train(self, epoch):

        self.model.train()
        print('Training')

        pos_cos_sim_list = list()
        neg_cos_sim_list = list()

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):
            
            self.optimizer.zero_grad()
            data = data.to(self.params.model.device)

            # All graph edges
            out_edge_attr, x_out, edge_pos_diff, edge_rot_diff = self.model(data.x, data.edge_index)
            
            # Attain positive edges
            # edge_attr_pos = torch.index_select(out_edge_attr, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask)).reshape(-1) 
            # # edge_cos_sim_pos = torch.index_select(edge_cos_sim, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask)).reshape(-1) 
            
            # # Attain negative edges
            # out_edge_attr_neg, x_neg_out = self.model(data.x, data.neg_edge_index)
            # out_edge_attr_neg = out_edge_attr_neg.reshape(-1)

            # time_i, time_j = data.node_times[data.pos_edge_index[0,:]], data.node_times[data.pos_edge_index[1,:]]
            # time_k = data.node_times[data.neg_edge_index[1,:]]
            # x_k = x_out[data.neg_edge_index[1,:]]

            # time_diff_pos = torch.abs(time_j - time_i)
            # time_diff_neg = torch.abs(time_k - time_i)

            # ssl_edge_label = torch.ones_like(time_diff_pos)
            # ssl_edge_label[time_diff_neg <= time_diff_pos] = -1

            # edge_cos_sim_mask = copy.deepcopy(data.pos_edge_mask)
            # edge_cos_sim_mask[edge_cos_sim_mask == 0] = -1

            # x_i, x_j = x_out[data.edge_index[0,:]], x_out[data.edge_index[1,:]]
            # x_i_pos = torch.index_select(x_i, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))   
            # x_j_pos = torch.index_select(x_j, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
            # node_cos_sim_pos = torch.cosine_similarity(x_i_pos, x_j_pos, dim=1)
            # node_cos_sim_neg = torch.cosine_similarity(x_i_pos, x_k, dim=1)

            # node_sim_pos = torch.mean(edge_attr_pos)
            # node_sim_neg = torch.mean(out_edge_attr_neg)


            # pos_cos_sim_list.append(torch.mean(node_cos_sim_pos).item())
            # neg_cos_sim_list.append(torch.mean(node_cos_sim_neg).item())

            loss_dict = {
                #'bce_loss_pos': torch.nn.BCEWithLogitsLoss()(edge_attr_pos, torch.ones_like(edge_attr_pos)),
                #'bce_loss_neg': torch.nn.BCEWithLogitsLoss()(out_edge_attr_neg, torch.zeros_like(out_edge_attr_neg)),
                # 'ranking_loss': torch.nn.MarginRankingLoss(margin=0.0)(edge_attr_pos, out_edge_attr_neg, ssl_edge_label),
                #'triplet_loss': torch.nn.TripletMarginLoss(margin=0.0)(x_i_pos, x_j_pos, x_k),
                "pos_loss": torch.nn.L1Loss()(edge_pos_diff, data.edge_pos_diff),
                "rot_loss": torch.nn.L1Loss()(edge_rot_diff, data.edge_rot_diff),

            }

            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()

            if not self.params.main.disable_wandb:
                wandb.log({"train/loss_total": loss.item()})
                # wandb.log({"train/ranking_loss": loss_dict['ranking_loss'].item()})
                # wandb.log({"train/triplet_loss": loss_dict['triplet_loss'].item()})
                # wandb.log({"train/bce_loss_pos": loss_dict['bce_loss_pos'].item()})
                # wandb.log({"train/bce_loss_neg": loss_dict['bce_loss_neg'].item()})

                wandb.log({"train/pos_loss": loss_dict['pos_loss'].item()})
                wandb.log({"train/rot_loss": loss_dict['rot_loss'].item()})
                # wandb.log({"train/pos_cos_sim": torch.mean(node_cos_sim_pos).item()})
                # wandb.log({"train/neg_cos_sim": torch.mean(node_cos_sim_neg).item()})
                # wandb.log({"train/pos_sim": node_sim_pos.item()})
                # wandb.log({"train/neg_sim": node_sim_neg.item()})

            self.total_step += 1
        
        pos_cos_sim_mean = np.nanmean(pos_cos_sim_list)
        neg_cos_sim_mean = np.nanmean(neg_cos_sim_list)
        print('pos_cos_sim_mean: {}'.format(pos_cos_sim_mean))
        print('neg_cos_sim_mean: {}'.format(neg_cos_sim_mean))

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
    train_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset)
    #, params.paths.config_name)
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



