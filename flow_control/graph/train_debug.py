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
from data import DisjDemoGraphDataset

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


    def train(self, epoch):

        metrics_dict = defaultdict(list)

        self.model.train()
        print('Training')

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):
            
            self.optimizer.zero_grad()
            data = data.to(self.params.model.device)

            # All graph edges
            out_edge_attr, x_out, edge_pos_diff, edge_rot_diff = self.model(data.x, data.edge_index)
            
            out_edge_attr_neg, _, _, _ = self.model(data.x, data.neg_edge_index)
            

            x_out_i, x_out_j, x_out_k = x_out[data.edge_index[0,:]], x_out[data.edge_index[1,:]], x_out[data.neg_edge_index[1,:]]
            edge_attr_pos = torch.index_select(out_edge_attr, 0, data.pos_edge_idcs).reshape(-1) 
            out_edge_attr_neg = out_edge_attr_neg.reshape(-1)
            


            edge_ranking_label = torch.ones_like(out_edge_attr_neg)

            edge_cos_sim_mask = copy.deepcopy(data.pos_edge_mask)
            edge_cos_sim_mask[edge_cos_sim_mask == 0] = -1


            x_out_ij = torch.cat((x_out_i, x_out_j), dim=1)
            x_i_pos = torch.index_select(x_out_i, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
            x_j_pos = torch.index_select(x_out_j, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))


            if not self.params.main.disable_wandb and step % 7 == 0:
                anchor_idx = np.random.choice(list(range(len(data.x))))

                outgoing_edge_cols = torch.where(data.edge_index[0,:] == anchor_idx)[0]
                num_j = outgoing_edge_cols.shape[0]

                # Assign plot indices
                other_idcs = [data.edge_index[1,edge_col] for edge_col in outgoing_edge_cols]
                other_idcs.append(anchor_idx)
                other_idcs = sorted(other_idcs)

                # Plot anchor
                fig, axarr = plt.subplots(1, num_j+1)
                x_anchor = data.x[anchor_idx].view(256, 256, 4)[:,:,0:3].cpu().detach().numpy()
                axarr[other_idcs.index(anchor_idx)].imshow(x_anchor.astype(np.uint8))
                axarr[other_idcs.index(anchor_idx)].axis('off')

                # Plot other frames
                for rel_j_idx, edge_col in enumerate(outgoing_edge_cols):
                    j = data.edge_index[1,edge_col]
                    x_j = data.x[j].view(256, 256, 4)[:,:,0:3].cpu().detach().numpy()
                    axarr[other_idcs.index(j)].imshow(x_j.astype(np.uint8))
                    axarr[other_idcs.index(j)].axis('off')
                    sim_anchor_j = torch.nn.CosineSimilarity(dim=0)(x_out[anchor_idx], x_out[j])

                    axarr[other_idcs.index(j)].text(0.0, 390.0, '{:.4f},\n {:.4f}'.format(sim_anchor_j.item(), 
                                                                                        out_edge_attr[edge_col].item()), fontsize=10)

                    #     axarr[2].text(0.0, 390.0, 'cos_sim: {:.4f}, \npos: {:.4f}, \nrot: {:.4f}, \n rnk: {:.4f}'.format(sim_neg.item(),
                    #                                                                                 edge_pos_diff[edge_idx_neg].item(),
                    #                                                                                 edge_rot_diff[edge_idx_neg].item(),
                    #                                                                                 out_edge_attr_neg[pos_pair_idx].item(),), fontsize=10)


                plt.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                # log wandb image
                wandb.log({"train/plot": wandb.Image(fig)})
                plt.close()


            acc = list()
            # Compute similarity accuracy
            for pos_edge_idx, pos_edge in enumerate(data.pos_edge_index.t()):
                x_i_out = x_out[data.pos_edge_index[0,pos_edge_idx]]
                x_j_out = x_out[data.pos_edge_index[1,pos_edge_idx]]
                x_k_out = x_out[data.neg_edge_index[1,pos_edge_idx]]

                i, j, k = data.pos_edge_index[0,pos_edge_idx], data.pos_edge_index[1,pos_edge_idx], data.neg_edge_index[1,pos_edge_idx]

                sim_pos = torch.nn.CosineSimilarity(dim=0)(x_i_out, x_j_out)
                sim_neg = torch.nn.CosineSimilarity(dim=0)(x_i_out, x_k_out)
                acc.append(sim_pos.item() > sim_neg.item())



            loss_dict = {
                "rnk_loss": torch.nn.MarginRankingLoss(margin=1.0)(edge_attr_pos, out_edge_attr_neg, edge_ranking_label),
                'tpl_loss': torch.nn.TripletMarginLoss(margin=0.0)(x_i_pos, x_j_pos, x_k),
                "pos_loss": 10*torch.nn.L1Loss()(edge_pos_diff.squeeze(1), data.edge_pos_diff),
                "rot_loss": torch.nn.L1Loss()(edge_rot_diff.squeeze(1), data.edge_rot_diff),
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
                wandb.log({"train/rnk_loss": loss_dict['rnk_loss'].item()})
                wandb.log({"train/tpl_loss": loss_dict['tpl_loss'].item()})
                wandb.log({"train/sim_acc": np.mean(acc)})
                #wandb.log({"train/sim_loss": loss_dict["sim_loss"].item()})

                # if step % 20 == 0:
                #     x_out_feats = x_out.cpu().detach().numpy()
                #     x_img = data.x.view(-1, 256, 256, 4)[:,:,:,0:3].cpu().detach().numpy().astype("uint8")

                #     images = [wandb.Image(Image.fromarray(im).resize((64,64))) for idx, im in enumerate(x_img)]
                #     # labels = np.array(list(range(0, x_out.shape[0])))
                #     # labels = [str(entry) for entry in data.batch.cpu().detach().numpy().tolist()] # seq number
                #     labels = data.node_times.cpu().detach().numpy().tolist() # time

                #     df = pd.DataFrame()
                #     df["image"] = images
                #     df["feats"] = [feat for idx, feat in enumerate(x_out_feats)]
                #     df["seq"] = [label for idx, label in enumerate(labels)] 
                #     table = wandb.Table(columns=df.columns.to_list(), data=df.values)
                #     wandb.log({"servoing": table})

            

            metrics_dict['loss'].append(loss.item())
            for key, value in loss_dict.items():
                metrics_dict[key].append(value.item())


            

            self.total_step += 1
        
        # pos_cos_sim_mean = np.nanmean(pos_cos_sim_list)
        # neg_cos_sim_mean = np.nanmean(neg_cos_sim_list)
        # print('pos_cos_sim_mean: {}'.format(pos_cos_sim_mean))
        # print('neg_cos_sim_mean: {}'.format(neg_cos_sim_mean))

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
                
                # All graph edges
                out_edge_attr, x_out, edge_pos_diff, edge_rot_diff = self.model(data.x, data.edge_index)
                # Attain positive edges
                edge_attr_pos = torch.index_select(out_edge_attr, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask)).reshape(-1) 
                # # edge_cos_sim_pos = torch.index_select(edge_cos_sim, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask)).reshape(-1) 
                
                # # Attain negative edges
                out_edge_attr_neg, _, _, _ = self.model(data.x, data.neg_edge_index)
                out_edge_attr_neg = out_edge_attr_neg.reshape(-1)

                # time_i, time_j = data.node_times[data.pos_edge_index[0,:]], data.node_times[data.pos_edge_index[1,:]]
                # time_k = data.node_times[data.neg_edge_index[1,:]]
                x_k = x_out[data.neg_edge_index[1,:]]

                # time_diff_pos = torch.abs(time_j - time_i)
                # time_diff_neg = torch.abs(time_k - time_i)

                edge_ranking_label = torch.ones_like(out_edge_attr_neg)

                edge_cos_sim_mask = copy.deepcopy(data.pos_edge_mask)
                edge_cos_sim_mask[edge_cos_sim_mask == 0] = -1

                x_i, x_j = x_out[data.edge_index[0,:]], x_out[data.edge_index[1,:]]
                x_ij = torch.cat((x_i, x_j), dim=1)
                x_i_pos = torch.index_select(x_i, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))
                x_j_pos = torch.index_select(x_j, 0, torch_geometric.utils.mask_to_index(data.pos_edge_mask))

            
            try:
                loss_dict = {
                #'bce_loss_pos': torch.nn.BCEWithLogitsLoss()(edge_attr_pos, torch.ones_like(edge_attr_pos)),
                #'bce_loss_neg': torch.nn.BCEWithLogitsLoss()(out_edge_attr_neg, torch.zeros_like(out_edge_attr_neg)),
                #"sim_loss": 0.1*torch.nn.CosineEmbeddingLoss()(x_i, x_j, edge_cos_sim_mask),
                "rnk_loss": torch.nn.MarginRankingLoss(margin=1.0)(edge_attr_pos, out_edge_attr_neg, edge_ranking_label),
                'tpl_loss': torch.nn.TripletMarginLoss(margin=0.0)(x_i_pos, x_j_pos, x_k),
                "pos_loss": 10*torch.nn.L1Loss()(edge_pos_diff.squeeze(1), data.edge_pos_diff),
                "rot_loss": torch.nn.L1Loss()(edge_rot_diff.squeeze(1), data.edge_rot_diff),
                }
            except Exception as e:
                print(e)
                continue
                
            acc = list()
            
            # Compute similarity accuracy
            for pos_edge_idx, _ in enumerate(data.pos_edge_index.t()):
                x_i_out = x_out[data.pos_edge_index[0,pos_edge_idx]]
                x_j_out = x_out[data.pos_edge_index[1,pos_edge_idx]]
                x_k_out = x_out[data.neg_edge_index[1,pos_edge_idx]]

                i, j, k = data.pos_edge_index[0,pos_edge_idx], data.pos_edge_index[1,pos_edge_idx], data.neg_edge_index[1,pos_edge_idx]

                sim_pos = torch.nn.CosineSimilarity(dim=0)(x_i_out, x_j_out)
                sim_neg = torch.nn.CosineSimilarity(dim=0)(x_i_out, x_k_out)
                acc.append(sim_pos.item() > sim_neg.item())

            loss = sum(loss_dict.values())

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


    model = gnn.DisjGNN()
    model = model.to(params.model.device)

    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

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

    trainer = Trainer(params, model, dataloader_train, dataloader_train, dataloader_test, optimizer)

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



