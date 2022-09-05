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
from data import RewardDataset, LiveDataset, DemoDataset
from network import EncoderNet
from plot_hist import plot_hist
from validate import validate

wandb.init(project="cnn_selection_unweighted")
torch.manual_seed(100)

def train():
    num_epochs = 1000
    batch_size = 80
    transforms = T.Compose([T.ToTensor()])
    target_file = './full_dataset.npz'
    target_data = np.load(target_file)['arr_0']
    load_model = None  # './models/model_unweighted_189.pth'

    training_data_live = LiveDataset(live_loc='./live_dir', transform=transforms)
    training_data_demo = DemoDataset(demo_loc='./demo_dir', transform=transforms)
    val_data_live = LiveDataset(live_loc='./live_dir_test', transform=transforms)

    train_loader_live = DataLoader(training_data_live, batch_size=1, shuffle=True)
    train_loader_demo = DataLoader(training_data_demo, batch_size=batch_size, shuffle=True)
    val_loader_live = DataLoader(val_data_live, batch_size=1, shuffle=False)

    model = EncoderNet().cuda()
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # criterion = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([5.0, 20.0]).cuda())
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch_idx, epoch in enumerate(range(num_epochs)):
        print(f"Epoch: {epoch}")
        # ipdb.set_trace()
        total_loss = 0.0
        correct = 0
        total = 0
        probs = []
        model.train()
        for live_imgs, live_idx in train_loader_live:
            live_imgs = live_imgs.repeat(batch_size, 1, 1, 1).cuda()
            for demo_imgs, demo_idx in train_loader_demo:
                demo_imgs = demo_imgs.cuda()
                targets = torch.Tensor([target_data[live_idx, demo_idx]]).to(torch.int64)
                targets = F.one_hot(targets, num_classes=2)[0].float().cuda()

                out = model(live_imgs, demo_imgs)

                optimizer.zero_grad()
                loss = criterion(out, targets)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

                model.eval()

                with torch.no_grad():
                    out1 = torch.softmax(out, dim=1)
                    out1 = out1.detach().cpu().numpy()
                    out1 = out1[:, 1]
                    probs.append(out1)

                prediction = torch.softmax(out, dim=1)
                prediction_argmax = torch.argmax(prediction, dim=1)
                targets_argmax = torch.argmax(targets, dim=1)

                correct += (prediction_argmax == targets_argmax).float().sum()

        print(correct, len(training_data_live) * len(training_data_demo))
        train_acc = 100.0 * correct / (len(training_data_live) * len(training_data_demo))

        # ipdb.set_trace()

        probs = np.array(probs)
        plot_hist(probs, epoch_idx)
        # validate(model)

        wandb.log({'train_accuracy': train_acc, 'loss': total_loss})

        print(f"Total Loss: {total_loss}, Accuracy: {train_acc}")

        if epoch_idx % 1 == 0 or epoch_idx == num_epochs - 1:
            torch.save(model.state_dict(), f'./models/model_unweighted_{epoch_idx}.pth')

        if epoch_idx % 1 == 0:
            # Run Validation
            with torch.no_grad():
                val_correct = 0
                val_loss = 0.0
                for live_imgs_val, live_idx_val in val_loader_live:
                    live_imgs_val = live_imgs_val.repeat(batch_size, 1, 1, 1).cuda()
                    for demo_imgs, demo_idx in train_loader_demo:
                        demo_imgs = demo_imgs.cuda()
                        targets = torch.Tensor([target_data[live_idx_val, demo_idx]]).to(torch.int64)
                        targets = F.one_hot(targets, num_classes=2)[0].float().cuda()

                        out = model(live_imgs_val, demo_imgs)

                        loss = criterion(out, targets)
                        val_loss += loss.item()

                        prediction = torch.softmax(out, dim=1)
                        prediction_argmax = torch.argmax(prediction, dim=1)
                        targets_argmax = torch.argmax(targets, dim=1)

                        val_correct += (prediction_argmax == targets_argmax).float().sum()
                val_acc = 100.0 * val_correct / (len(val_data_live) * len(training_data_demo))
                wandb.log({'val_accuracy': val_acc, 'val_loss': val_loss})

if __name__ == '__main__':
    train()
