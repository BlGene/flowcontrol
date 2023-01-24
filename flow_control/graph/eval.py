import os
import argparse
import yaml
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch_geometric
import wandb
import matplotlib.pyplot as plt

from utils import ParamLib
import gnn
from data import DisjDemoGraphDataset


class Evaluator():

    def __init__(self, params, model, dataloader_train, dataloader_test):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.params = params
        self.edge_criterion = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0


    def eval(self, epoch, split='test', log_images=False):

        metrics_dict = defaultdict(list)

        self.model.eval()
        print('Evaluating on {}'.format(split))

        if split == 'test':
            dataloader = self.dataloader_test
        elif split == 'train':
            dataloader = self.dataloader_train

        dataloader_progress = tqdm(dataloader, desc='Evaluating on {}'.format(split))

        for i_val, data in enumerate(dataloader_progress):

            with torch.no_grad():
                data = data.to(self.params.model.device)
                
                # All graph edges
                out_edge_attr, x_out, edge_pos_diff, edge_rot_diff = self.model(data.x, data.edge_index)
                edge_attr_pos = torch.index_select(out_edge_attr, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask)).reshape(-1) 

                out_edge_attr_neg, _, _, _ = self.model(data.x, data.neg_edge_index)
                out_edge_attr_neg = out_edge_attr_neg.reshape(-1)

                x_i, x_j = x_out[data.edge_index[0,:]], x_out[data.edge_index[1,:]]
                x_k = x_out[data.neg_edge_index[1,:]]
                x_ij = torch.cat((x_i, x_j), dim=1)
                x_i_pos = torch.index_select(x_i, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
                x_j_pos = torch.index_select(x_j, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))

                edge_ranking_label = torch.ones_like(out_edge_attr_neg)

                loss_dict = {
                    "rnk_loss": torch.nn.MarginRankingLoss(margin=1.0)(edge_attr_pos, out_edge_attr_neg, edge_ranking_label),
                    'tpl_loss': torch.nn.TripletMarginLoss(margin=0.0)(x_i_pos, x_j_pos, x_k),
                    "pos_loss": 10*torch.nn.L1Loss()(edge_pos_diff.squeeze(1), data.edge_pos_diff),
                    "rot_loss": torch.nn.L1Loss()(edge_rot_diff.squeeze(1), data.edge_rot_diff),
                }
                loss = sum(loss_dict.values())

                metrics_dict['loss'].append(loss.item())
                for key, value in loss_dict.items():
                    metrics_dict[key].append(value.item())

                # Compute and log similarity accuracy
                for pos_edge_idx, _ in enumerate(data.pos_edge_index.t()):
                    x_i_out = x_out[data.pos_edge_index[0,pos_edge_idx]]
                    x_j_out = x_out[data.pos_edge_index[1,pos_edge_idx]]
                    x_k_out = x_out[data.neg_edge_index[1,pos_edge_idx]]

                    i, j, k = data.pos_edge_index[0,pos_edge_idx], data.pos_edge_index[1,pos_edge_idx], data.neg_edge_index[1,pos_edge_idx]

                    sim_pos = torch.nn.CosineSimilarity(dim=0)(x_i_out, x_j_out)
                    sim_neg = torch.nn.CosineSimilarity(dim=0)(x_i_out, x_k_out)
                    metrics_dict['sim_acc'].append(sim_pos.item() > sim_neg.item())            

            # Loop through all edge starting at randomly sampled node
            for anchor_idx, node_feat in enumerate(data.x):
                print(anchor_idx)
                outgoing_edge_cols = torch.where(data.edge_index[0,:] == anchor_idx)[0]
                num_j = outgoing_edge_cols.shape[0]

                fig, axarr = plt.subplots(1, num_j+1)


                x_anchor = data.x[anchor_idx].view(256, 256, 4)[:,:,0:3].cpu().detach().numpy()
                axarr[0].imshow(x_anchor.astype(np.uint8))

                for rel_j_idx, edge_col in enumerate(outgoing_edge_cols):
                    j = data.edge_index[1,edge_col]
                    x_j = data.x[j].view(256, 256, 4)[:,:,0:3].cpu().detach().numpy()
                    axarr[rel_j_idx+1].imshow(x_j.astype(np.uint8))
                    axarr[rel_j_idx+1].axis('off')
                    sim_anchor_j = torch.nn.CosineSimilarity(dim=0)(x_out[anchor_idx], x_out[j])

                    axarr[rel_j_idx+1].text(0.0, 390.0, '{:.4f}'.format(sim_anchor_j.item()), fontsize=10)

                plt.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                # log wandb image
                wandb.log({"plot": wandb.Image(fig)})
                plt.close()
                # except Exception as e:
                #     print(e)

        metrics_dict = {key: np.nanmean(value) for key, value in metrics_dict.items()}
        if not self.params.main.disable_wandb:
            for key, value in metrics_dict.items():
                wandb.log({"{}/{}".format(split, key): value})



def main():
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

    # Namespace-specific arguments (namespace: evall)
    parser.add_argument('--checkpoints', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.eval.overwrite(opt)

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
    state_dict = torch.load(os.path.join(params.paths.checkpoints, params.eval.checkpoint),
                            map_location=torch.device('cuda')
                            )
    model.load_state_dict(state_dict)
    model = model.to(params.eval.device)

    weights = [w for w in model.parameters() if w.requires_grad]

    # define own collator that skips bad samples
    data_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset) #, params.paths.config_name)

    dataset_train = DisjDemoGraphDataset(path=data_path, split="train", split_idx=params.model.split_idx)
    dataset_test = DisjDemoGraphDataset(path=data_path, split="test", split_idx=params.model.split_idx)

    dataloader_obj = torch_geometric.loader.DataLoader
    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    dataloader_test = dataloader_obj(dataset_test,
                                     batch_size=params.model.batch_size,
                                     num_workers=2,
                                     shuffle=False)

    evaluator = Evaluator(params, model, dataloader_train, dataloader_test)

    evaluator.eval(0, split='train', log_images=True)

if __name__ == '__main__':
    main()
