import ipdb
import torch
import torch.nn as nn
import numpy as np
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

# torch.manual_seed(100)
def validate(model=None):
    model_idx = 70
    model = EncoderNet().cuda()
    # model.load_state_dict(torch.load(f'./models/model_unweighted_{model_idx}.pth'))
    model.load_state_dict(torch.load(f'./model.pth'))
    model.eval()
    transforms = T.Compose([T.ToTensor()])
    batch_size = 74

    validation_data_live = LiveDataset(live_loc='./live_imgs_test_new_env', transform=transforms, train=False)
    validation_data_demo = DemoDataset(demo_loc='./demo_dir_new_env', transform=transforms)
    val_loader_live = DataLoader(validation_data_live, batch_size=1)
    val_loader_demo = DataLoader(validation_data_demo, batch_size=batch_size)

    bidx = {}

    probs = []

    full_target = np.zeros((len(validation_data_live) * len(validation_data_demo), 2))
    full_preds = np.zeros_like(full_target)

    for live_imgs, live_idx in val_loader_live:
        live_imgs = live_imgs.repeat(batch_size, 1, 1, 1).cuda()

        for demo_imgs, demo_idx in val_loader_demo:
            demo_imgs = demo_imgs.cuda()
            # targets = torch.Tensor([target_data[live_idx, demo_idx]]).to(torch.int64)
            # targets = F.one_hot(targets, num_classes=2)[0].float().cuda()

            # ipdb.set_trace()

            out = model(live_imgs, demo_imgs)
            out = torch.softmax(out, dim=1)
            out = out.detach().cpu().numpy()
            out = out[:, 1]
            probs.append(out)

            best_idx = np.argmax(out)
            # print(out[best_idx])

            bidx[live_idx.item() + 100] = best_idx
    print(bidx)
    probs = np.array(probs)
    print(probs.shape)
    # plot_hist(probs, model_idx)
    np.savez('./probs_ml_new_env.npz', probs)
if __name__ == '__main__':
    validate()


