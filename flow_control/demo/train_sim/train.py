import ipdb
import torch
import torch.nn as nn
import numpy as np
import wandb
import os
import os.path as osp
import cv2
from PIL import Image
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from data import DLDataset
from network import SimilarityNet
# from .plot_hist import plot_hist
# from .validate import validate
from tqdm import tqdm


wandb.init(project="shape_sorting_3_part")
torch.manual_seed(100)


def validate(model, val_loader, batch_size):
    print("Validating:")
    model.eval()

    correct_preds = 0
    correct_preds1 = 0
    correct_preds0 = 0

    num_rew1 = 0
    num_rew0 = 0

    for lp0, lp1, lp2, dp0, dp1, dp2, rewards in tqdm(val_loader):
        p0 = torch.cat((lp0, dp0), dim=1).cuda()
        p1 = torch.cat((lp1, dp1), dim=1).cuda()
        p2 = torch.cat((lp2, dp2), dim=1).cuda()
        rewards = rewards.cuda()

        inv_rewards = 1 - rewards
        num_rew0 += inv_rewards.float().sum()
        num_rew1 += rewards.float().sum()

        out_0 = model(p0)
        out_1 = model(p1)
        out_2 = model(p2)

        out = out_0 * out_1 * out_2

        out_rewards = out[:, 0] >= 0.5

        correct = out_rewards == rewards
        correct_1 = correct * rewards
        correct_0 = correct * (1 - rewards)

        correct_preds += correct.float().sum()
        correct_preds0 += correct_0.float().sum()
        correct_preds1 += correct_1.float().sum()

    accuracy = correct_preds * 100.0 / (len(val_loader) * batch_size)
    accuracy0 = correct_preds0 * 100.0 / num_rew0
    accuracy1 = correct_preds1 * 100.0 / num_rew1


    wandb.log({'accuracy': accuracy, 'accuracy 0': accuracy0, 'accuracy 1': accuracy1})


def train():
    num_epochs = 1000
    batch_size = 100
    lr = 0.001
    val_freq = 5
    save_freq = 10
    transforms = T.Compose([T.ToTensor()])
    
    load_model = None

    data_dir = "/misc/student/nayaka/paper/flowcontrol/flow_control/demo"

    train_dataset = DLDataset(live_img_dir=f"{data_dir}/train_sim/live_imgs",
                              demo_img_dir=f"{data_dir}/train_sim/demo_imgs",
                              rew_file=f"{data_dir}/tmp_new/cnn_run/rewards.json",
                              run_dir=f"{data_dir}/tmp_new/cnn_run",
                              train=True,
                              transform=transforms)

    val_dataset = DLDataset(live_img_dir=f"{data_dir}/train_sim/live_imgs",
                              demo_img_dir=f"{data_dir}/train_sim/demo_imgs",
                              rew_file=f"{data_dir}/tmp_new/cnn_run/rewards.json",
                              run_dir=f"{data_dir}/tmp_new/cnn_run",
                              train=False,
                              transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = nn.DataParallel(SimilarityNet(), device_ids=[0,1])
    model = model.cuda()

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00005, eps=1e-8)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs,
    #                                                 steps_per_epoch=len(train_loader), pct_start=0.05,
    #                                                 cycle_momentum=False, anneal_strategy='linear')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    criterion = torch.nn.BCELoss()

    # with torch.no_grad():
    #     validate(model, val_loader, batch_size=batch_size)

    for epoch_idx, epoch in enumerate(range(num_epochs)):
        print(f"Epoch: {epoch}")

        total_loss = 0.0
        model.train()

        cur_lr = float(scheduler.get_last_lr()[0])

        wandb.log({'Epoch Index': epoch_idx, "LR": cur_lr})

        for lp0, lp1, lp2, dp0, dp1, dp2, rewards in tqdm(train_loader):

            optimizer.zero_grad()

            p0 = torch.cat((lp0, dp0), dim=1).cuda()
            p1 = torch.cat((lp1, dp1), dim=1).cuda()
            p2 = torch.cat((lp2, dp2), dim=1).cuda()

            rewards = rewards.cuda()

            out_0 = model(p0)
            out_1 = model(p1)
            out_2 = model(p2)

            out = out_0 * out_1 * out_2
            
            loss = criterion(out[:, 0], rewards)

            loss.backward()

            total_loss += loss.item()

            wandb.log({'loss': loss.item()})

            optimizer.step()

        wandb.log({'total_loss': total_loss})

        scheduler.step()

        if epoch_idx % val_freq == 0:
            model.eval()

            with torch.no_grad():
                validate(model, val_loader, batch_size=batch_size)

        if epoch_idx % save_freq == 0 or epoch_idx == num_epochs - 1:
            torch.save(model.state_dict(), f'{data_dir}/train_sim/models/simNet_{epoch_idx}.pth')


if __name__ == '__main__':
    train()
