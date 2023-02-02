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
from torchmetrics.functional.classification.average_precision import average_precision
from torchmetrics.functional.classification.precision_recall import recall
from PIL import Image
np.random.seed(42)
torch.manual_seed(42)

from utils import ParamLib
import gnn
from data import DisjBidirDemoGraphDataset

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
        self.edge_criterion = torch.nn.BCELoss()
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
            
            acc = list()

            self.optimizer.zero_grad()
            data = data.to(self.params.model.device)

            # All graph edges
            out_edge_attr, x_out, edge_pos_diff, edge_rot_diff = self.model(data.x, data.edge_index)

            triplet_node_feats = torch.empty((0, 3*x_out.shape[1])).to(self.params.model.device)
            triplet_edge_scores = torch.empty((0, 2)).to(self.params.model.device)
            # get triplet node features
            for pos_edge, neg_edge in zip(data.pos_edge_index.t(), data.neg_edge_index.t()):
                i, j = pos_edge[0].item(), pos_edge[1].item()
                k = neg_edge[1].item()
                pos_edge_idx = torch.where((data.edge_index[0] == i) & (data.edge_index[1] == j))
                neg_edge_idx = torch.where((data.edge_index[0] == i) & (data.edge_index[1] == k))
                # node features
                x_out_i = x_out[i]
                x_out_j = x_out[j]
                x_out_k = x_out[k]
                triplet_node_feats = torch.cat((triplet_node_feats, torch.cat((x_out_i, x_out_j, x_out_k)).reshape(1,-1)), dim=0)

                # edge features
                out_edge_attr_pos = out_edge_attr[pos_edge_idx].reshape(-1)
                out_edge_attr_neg = out_edge_attr[neg_edge_idx].reshape(-1)
                triplet_edge_scores = torch.cat((triplet_edge_scores, torch.cat((out_edge_attr_pos, out_edge_attr_neg)).reshape(1,-1)), dim=0)

                # compute cos-similarity accuracy among node features
                sim_pos = torch.nn.CosineSimilarity(dim=0)(x_out_i, x_out_j)
                sim_neg = torch.nn.CosineSimilarity(dim=0)(x_out_i, x_out_k)
                acc.append(sim_pos.item() > sim_neg.item())
                # acc.append(edge_pos_diff[pos_edge_idx].item() < edge_pos_diff[neg_edge_idx].item())
                # acc.append(out_edge_attr_pos.item() > out_edge_attr_neg.item())

            edge_ranking_labels = torch.ones((triplet_edge_scores.shape[0])).to(self.params.model.device)            

            loss_dict = {
                "rnk_loss": torch.nn.MarginRankingLoss(margin=0.5)(triplet_edge_scores[:,0], triplet_edge_scores[:,1], edge_ranking_labels),
                'tpl_loss': torch.nn.TripletMarginLoss(margin=0.5)(triplet_node_feats[:,0*x_out.shape[1]:1*x_out.shape[1]], 
                                                                   triplet_node_feats[:,1*x_out.shape[1]:2*x_out.shape[1]], 
                                                                   triplet_node_feats[:,2*x_out.shape[1]:3*x_out.shape[1]]),
                "pos_loss": 10*torch.nn.L1Loss()(edge_pos_diff.squeeze(1), data.edge_pos_diff),
                "rot_loss": torch.nn.L1Loss()(edge_rot_diff.squeeze(1), data.edge_rot_diff),
            }

            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()

            wandb.log({"train/loss_total": loss.item()})
            wandb.log({"train/sim_acc": np.mean(acc)})
            if not self.params.main.disable_wandb:
                for key, value in loss_dict.items():
                    wandb.log({"{}/{}".format("train", key): value.item()})

            if not self.params.main.disable_wandb and step % 7 == 0:
                anchor_idx = np.random.choice(list(range(len(data.x))))

                outgoing_edge_cols = torch.where(data.edge_index[0,:] == anchor_idx)[0]
                num_j = outgoing_edge_cols.shape[0]

                # Assign plot indices
                other_idcs = [data.edge_index[1,edge_col] for edge_col in outgoing_edge_cols]
                other_idcs.append(anchor_idx)
                other_idcs = sorted(other_idcs)

                # Plot anchor
                try: 
                    fig, axarr = plt.subplots(1, num_j+1)
                    x_anchor = data.x[anchor_idx].view(256, 256, 5)[:,:,0:3].cpu().detach().numpy()
                    axarr[other_idcs.index(anchor_idx)].imshow(x_anchor.astype(np.uint8))
                    axarr[other_idcs.index(anchor_idx)].axis('off')

                    # Plot other frames
                    for rel_j_idx, edge_col in enumerate(outgoing_edge_cols):
                        j = data.edge_index[1,edge_col]
                        x_j = data.x[j].view(256, 256, 5)[:,:,0:3].cpu().detach().numpy()
                        axarr[other_idcs.index(j)].imshow(x_j.astype(np.uint8))
                        axarr[other_idcs.index(j)].axis('off')
                        sim_anchor_j = torch.nn.CosineSimilarity(dim=0)(x_out[anchor_idx], x_out[j])

                        axarr[other_idcs.index(j)].text(0.0, 390.0, '{:.4f},\n {:.4f}'.format(sim_anchor_j.item(), 
                                                                                            out_edge_attr[edge_col].item()), fontsize=10)

                    plt.tight_layout()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    # log wandb image
                    wandb.log({"train/plot": wandb.Image(fig)})
                    plt.close()
                except Exception as e:
                    print(e)
                    pass

            metrics_dict['loss'].append(loss.item())
            for key, value in loss_dict.items():
                metrics_dict[key].append(value.item())

            self.total_step += 1

        text = 'Epoch {} / {} step {} / {}, train loss = {:03f}.'. \
            format(epoch, self.params.model.num_epochs, self.total_step + 1, len(self.dataloader_train), loss.item())
        train_progress.set_description(text)

    def eval(self, epoch, split='test', log_images=False):

        metrics_dict = defaultdict(list)

        self.model.eval()
        print('Evaluating on {}'.format(split))

        if split == 'test':
            dataloader = self.dataloader_test

        dataloader_progress = tqdm(dataloader, desc='Evaluating on {}'.format(split))

        for i_val, data in enumerate(dataloader_progress):

            with torch.no_grad():
                data = data.to(self.params.model.device)
                
                acc = list()

                # All graph edges
                out_edge_attr, x_out, edge_pos_diff, edge_rot_diff = self.model(data.x, data.edge_index)

                x_out_i, x_out_j = x_out[data.edge_index[0,:]], x_out[data.edge_index[1,:]]
                x_out_ij = torch.cat((x_out_i, x_out_j), dim=1)

                triplet_node_feats = torch.empty((0, 3*x_out.shape[1])).to(self.params.model.device)
                triplet_edge_scores = torch.empty((0, 2)).to(self.params.model.device)
                # get triplet node features
                for pos_edge_idx, neg_edge_idx in zip(data.pos_edge_idcs, data.neg_edge_idcs):
                    # node features
                    x_out_i = x_out[data.edge_index[0,pos_edge_idx]]
                    x_out_j = x_out[data.edge_index[1,pos_edge_idx]]
                    x_out_k = x_out[data.edge_index[1,neg_edge_idx]]
                    triplet_node_feats = torch.cat((triplet_node_feats, torch.cat((x_out_i, x_out_j, x_out_k)).reshape(1,-1)), dim=0)

                    # edge features
                    out_edge_attr_pos = out_edge_attr[pos_edge_idx]
                    out_edge_attr_neg = out_edge_attr[neg_edge_idx]
                    triplet_edge_scores = torch.cat((triplet_edge_scores, torch.cat((out_edge_attr_pos, out_edge_attr_neg)).reshape(1,-1)), dim=0)

                    # compute cos-similarity accuracy among node features
                    sim_pos = torch.nn.CosineSimilarity(dim=0)(x_out_i, x_out_j)
                    sim_neg = torch.nn.CosineSimilarity(dim=0)(x_out_i, x_out_k)
                    acc.append(sim_pos.item() > sim_neg.item())
                    # acc.append(edge_pos_diff[pos_edge_idx].item() < edge_pos_diff[neg_edge_idx].item())
                    # acc.append(out_edge_attr_pos.item() > out_edge_attr_neg.item())

                edge_ranking_labels = torch.ones((triplet_edge_scores.shape[0])).to(self.params.model.device)

                loss_dict = {
                    "rnk_loss": torch.nn.MarginRankingLoss(margin=1.0)(triplet_edge_scores[:,0], triplet_edge_scores[:,1], edge_ranking_labels),
                    'tpl_loss': torch.nn.TripletMarginLoss(margin=0.0)(triplet_node_feats[:,0*x_out.shape[1]:1*x_out.shape[1]], 
                                                                        triplet_node_feats[:,1*x_out.shape[1]:2*x_out.shape[1]], 
                                                                        triplet_node_feats[:,2*x_out.shape[1]:3*x_out.shape[1]]),
                    "pos_loss": 10*torch.nn.L1Loss()(edge_pos_diff.squeeze(1), data.edge_pos_diff),
                    "rot_loss": torch.nn.L1Loss()(edge_rot_diff.squeeze(1), data.edge_rot_diff),
                }
                loss = sum(loss_dict.values())
                

                if not self.params.main.disable_wandb and i_val % 10 == 0:
                    anchor_idx = np.random.choice(list(range(len(data.x))))

                    outgoing_edge_cols = torch.where(data.edge_index[0,:] == anchor_idx)[0]
                    num_j = outgoing_edge_cols.shape[0]

                    # Assign plot indices
                    other_idcs = [data.edge_index[1,edge_col] for edge_col in outgoing_edge_cols]
                    other_idcs.append(anchor_idx)
                    other_idcs = sorted(other_idcs)

                    try: 
                        # Plot anchor
                        fig, axarr = plt.subplots(1, num_j+1)
                        x_anchor = data.x[anchor_idx].view(256, 256, 5)[:,:,0:3].cpu().detach().numpy()
                        axarr[other_idcs.index(anchor_idx)].imshow(x_anchor.astype(np.uint8))

                        # Plot other frames
                        for rel_j_idx, edge_col in enumerate(outgoing_edge_cols):
                            j = data.edge_index[1,edge_col]
                            x_j = data.x[j].view(256, 256, 5)[:,:,0:3].cpu().detach().numpy()
                            axarr[other_idcs.index(j)].imshow(x_j.astype(np.uint8))
                            axarr[other_idcs.index(j)].axis('off')
                            sim_anchor_j = torch.nn.CosineSimilarity(dim=0)(x_out[anchor_idx], x_out[j])

                            axarr[other_idcs.index(j)].text(0.0, 390.0, '{:.4f},\n {:.4f}'.format(sim_anchor_j.item(), 
                                                                                                out_edge_attr[edge_col].item()), fontsize=8)

                        plt.tight_layout()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        # log wandb image
                        wandb.log({"test/plot": wandb.Image(fig)})
                        plt.close()
                    except Exception as e:
                        print(e)


                metrics_dict['sim_acc'].append(np.mean(acc))
                metrics_dict['loss'].append(loss.item())
                for key, value in loss_dict.items():
                    metrics_dict[key].append(value.item())

            text = 'Epoch {} / {} step {} / {}, test loss = {:03f}.'. \
                format(epoch, self.params.model.num_epochs, self.total_step + 1, len(dataloader), loss.item())
            dataloader_progress.set_description(text)

        metrics_dict = {key: np.nanmean(value) for key, value in metrics_dict.items()}
        if not self.params.main.disable_wandb:
            for key, value in metrics_dict.items():
                wandb.log({"{}/{}".format(split, key): value})
        


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
    model = model.to(params.model.device)
    
    if params.model.num_img_chs == 3:
        tqdm.write("Modalities: RGB")
    elif params.model.num_img_chs == 4:
        tqdm.write("Modalities: RGB-D or RGB-MASK")
    elif params.model.num_img_chs == 5:
        tqdm.write("Modalities: RGB-D-MASK")

    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # define own collator that skips bad samples
    data_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset) #, params.paths.config_name)

    dataset_train = DisjBidirDemoGraphDataset(path=data_path, split="train", split_idx=params.model.split_idx)
    dataset_test = DisjBidirDemoGraphDataset(path=data_path, split="test", split_idx=params.model.split_idx)

    dataloader_obj = torch_geometric.loader.DataLoader
    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    dataloader_test = dataloader_obj(dataset_test,
                                     batch_size=params.model.batch_size,
                                     num_workers=2,
                                     shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_test, dataloader_test, optimizer)

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
        trainer.eval(epoch, split='test', log_images=True)

if __name__ == '__main__':
    main()



