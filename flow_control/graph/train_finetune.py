import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import copy
import json
import argparse
from tqdm import tqdm
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch_geometric.data
import torch_geometric.data.batch

import matplotlib.pyplot as plt
import wandb
from torchmetrics.functional.classification.average_precision import binary_average_precision
from torchmetrics.functional.classification.precision_recall import binary_recall
from PIL import Image
# np.random.seed(42)
# torch.manual_seed(42)

from utils import ParamLib
import gnn
from data import DemoData

# print the used pytorch seed
print('Pytorch seed: ', torch.initial_seed())
print('Numpy seed: ', np.random.get_state()[1][0])



class Trainer():

    def __init__(self, params, model, dataloader_train, dataloader_test, dataloader_trainoverfit, optimizer):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataloader_trainoverfit = dataloader_trainoverfit
        self.params = params
        self.optimizer = optimizer
        self.edge_crit = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0

        # plt.ion()  # turns on interactive mode
        self.figure, self.axarr = plt.subplots(1, 2)


    def train(self, epoch):

        metrics_dict = defaultdict(list)

        self.model.train()
        print('Training')

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):
        

            self.optimizer.zero_grad()
            data = data.to(self.params.model.device)

            out_x = self.model.img_encoder.forward(data.x)

            # Construct edge features and concatenate
            x_out_i, x_out_j = out_x[data.edge_index[0,:]].reshape(-1, 64), out_x[data.edge_index[1,:]].reshape(-1, 64)
            score = torch.sigmoid(torch.nn.CosineSimilarity(dim=1)(x_out_i, x_out_j))               
            
            loss_dict = {
                "bce_loss": torch.nn.BCELoss()(score, data.reward),
            }

            # calculate average precision
            ap = binary_average_precision(score, data.reward)
            metrics_dict['ap'].append(ap.item())

            # calculate recall
            rec = binary_recall(score, data.reward)
            metrics_dict['recall'].append(rec.item())


            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()
            
            if not self.params.main.disable_wandb:
                wandb.log({"train/loss_total": loss.item()})
                for key, value in loss_dict.items():
                    wandb.log({"{}/{}".format("train", key): value.item()})
                wandb.log({"train/ap": ap.item()})
                wandb.log({"train/recall": rec.item()})

            metrics_dict['loss'].append(loss.item())
            for key, value in loss_dict.items():
                metrics_dict[key].append(value.item())


            self.total_step += 1

        text = 'Epoch {} / {} step {} / {}, train loss = {:03f}.'. \
            format(epoch, self.params.model.num_epochs, self.total_step + 1, len(self.dataloader_train), loss.item())
        train_progress.set_description(text)

    

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

    model = gnn.DisjGNN(params=params)
    state_dict = torch.load(os.path.join(params.paths.checkpoints, params.eval.checkpoint),
                            map_location=torch.device(params.model.device)
                            )
    model.load_state_dict(state_dict)
    model = model.to(params.model.device)

    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # define own collator that skips bad samples
    data_path = os.path.join(params.paths.dataroot, params.paths.finetune_dataset)

    dataset_train = DemoData(path=data_path, split="train")
    # dataset_test = DisjBidirDemoGraphDataset(path=data_path, split="test", split_idx=params.model.split_idx)

    dataloader_obj = torch_geometric.loader.DataLoader
    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    # dataloader_test = dataloader_obj(dataset_test,
    #                                  batch_size=params.model.batch_size,
    #                                  num_workers=2,
    #                                  shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_train, dataloader_train, optimizer)

    for epoch in range(params.model.num_epochs):
        trainer.train(epoch)

        if not params.main.disable_wandb:
            wandb_run_name = wandb.run.name
        
            fname = 'servo_simnet_{}_{:03d}.pth'.format(wandb_run_name, epoch)
        
            # save checkpoint locally and in wandb
            torch.save(model.state_dict(), params.paths.checkpoints + fname)
            wandb.save(params.paths.checkpoints + fname)
        #
        # Evaluate
        # trainer.eval(epoch, split='test', log_images=True)

if __name__ == '__main__':
    main()



