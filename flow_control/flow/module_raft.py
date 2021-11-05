"""
Compute flow using RAFT
"""
import os
import time
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image


try:
    import raft
    from utils.utils import InputPadder
    from utils.utils import forward_interpolate
except ModuleNotFoundError as error:
    print("RAFT module not found.")
    print("try: export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core")
    raise ModuleNotFoundError(
        "Module RAFT not found. Did you install it?"
    ) from error


torch.backends.cudnn.benchmark = True


class FlowModule:
    """
    Compute flow using RAFT method
    """
    def __init__(self, size=None, iterations=20):

        parser = argparse.ArgumentParser()
        parser.add_argument('--model')
        parser.add_argument('--path')
        parser.add_argument('--small', action='store_true')
        parser.add_argument('--mixed_precision', action='store_true')
        parser.add_argument('--alternate_corr', action='store_true')
        args = parser.parse_args([])

        flownet_variant = "RAFT"
        self.method_name = flownet_variant

        model = torch.nn.DataParallel(raft.RAFT(args))

        raft_root = Path(raft.__file__).parent.parent
        raft_root = raft_root / 'models' / 'raft-things.pth'
        logging.info("Loading RAFT model, may take a bit...")
        try:
            model.load_state_dict(torch.load(raft_root))
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "RAFT weights not found in \'" + str(raft_root) + "\', "
                + "did you download them to \'RAFT/models/*\'") from error

        model = model.module
        model = model.cuda()
        model = model.eval()
        self.model = model

        self.flow_prev = None
        self.iterations = iterations

    def _totorch(self, array):
        """
        Converts a numpy array to torch

        Args:
            array: [h,w,c] dtype=*
        Returns:
            tensor: [1,h,w,c] dtype=float
        """
        return torch.from_numpy(array)[None].float().permute(0, 3, 1, 2).cuda()

    def step(self, img0, img1):
        """
        compute flow

        Args:
            img0: [h,w,3] dtype=uint8
            img1: [h,w,3] dtype=uint8
        Returns:
            flow: [h,w,2] dtype=float
        """

        with torch.no_grad():

            img0 = torch.from_numpy(img0).float().permute(2, 0, 1)[None].cuda()
            img1 = torch.from_numpy(img1).float().permute(2, 0, 1)[None].cuda()
            padder = InputPadder(img0.shape)

            image1, image2 = padder.pad(img0, img1)

            flow_low, flow_up = self.model(
                image1, image2,
                flow_init=self.flow_prev,
                iters=self.iterations,
                test_mode=True
            )

            self.flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            return padder.unpad(flow_up[0]).permute(1, 2, 0).detach().cpu().numpy()

    def warp(self, x, flow):
        """
        Warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: [B, C, H, W] (im2)
            flow: [B, 2, H, W] flow

        Returns:
            warped: [B, C, H, W]
        """
        B, _, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().cuda()

        vgrid = torch.autograd.Variable(grid) + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).cuda()
        mask = torch.nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def warp_image(self, x, flow):
        """
        Warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: img2 as numpy array [H, W, C]
            flow: as numpy array [H, W, 2]

        Returns:
            warped: [H, W, 2] numpy
        """
        x = torch.from_numpy(x)[None].float().permute(0, 3, 1, 2).cuda()
        flow = torch.from_numpy(flow)[None].float().permute(0, 3, 1, 2).cuda()
        output = self.warp(x, flow)
        return output[0].permute(1, 2, 0).detach().cpu().numpy()

    def warp_mask(self, mask, img0, img1):
        """
        Warps the mask aligned with img0 to img1

        Args:
            mask: [h, w, 1] dtype=bool
            img0: [h, c, 3] dtype=uint8
            img1: [h, c, 3] dtype=uint8

        Returns:
            mask: [h, w, 1] dtype=bool mask aligned with img1
        """
        with torch.no_grad():

            mask = self._totorch(mask)
            img0 = self._totorch(img0)
            img1 = self._totorch(img1)

            padder = InputPadder(img0.shape)
            img0, img1 = padder.pad(img0, img1)

            fwd = torch.cat([img0, img1], 0)
            bwd = torch.cat([img1, img0], 0)

            # Compute the forward and backward flow
            _, flow = self.model(fwd, bwd, iters=20, test_mode=True)

            # Warp mask into next
            mask_warp = self.warp(mask, flow[1:])
            flow_warp = self.warp(flow[:1], flow[1:])
            flow_orig = flow[1:]

            # Create occlusion mask
            t1 = torch.sum((flow_orig + flow_warp) ** 2, dim=1, keepdim=True)
            t2 = torch.sum(flow_orig ** 2, dim=1, keepdim=True)
            t3 = torch.sum(flow_warp ** 2, dim=1, keepdim=True)
            mask_occ = 1.0 - (t1 > 0.01 * (t2 + t3) + 0.5).float()

            mask = (mask_warp * mask_occ)
            return mask.permute(0, 2, 3, 1)[0].bool().cpu().numpy()


def read_flo_as_float32(filename):
    '''read .flo files'''
    with open(filename, 'rb') as file:
        magic = np.fromfile(file, np.float32, count=1)
        assert magic == 202021.25, "Magic number incorrect. Invalid .flo file"
        width = np.fromfile(file, np.int32, count=1)[0]
        height = np.fromfile(file, np.int32, count=1)[0]
        data = np.fromfile(file, np.float32, count=2 * height * width)
    data_2d = np.resize(data, (height, width, 2))
    return data_2d


def test_flow_module():
    """
    test the fow module
    """
    test_dir = "/home/argusm/lang/flownet2/data/FlyingChairs_examples"
    dict_fn = dict(img0='0000000-img0.ppm', img1='0000000-img1.ppm')

    for image_name, image_file in dict_fn.items():
        path = os.path.join(test_dir, image_file)
        dict_fn[image_name] = path
        assert os.path.isfile(path)

    image1 = np.asarray(Image.open(dict_fn['img0'])).copy()
    image2 = np.asarray(Image.open(dict_fn['img1'])).copy()

    flow_module = FlowModule()

    start = time.time()
    tmp = flow_module.step(image1, image2)
    end = time.time()
    print("time 1", end - start)

    start = time.time()
    tmp = flow_module.step(image1, image2)
    end = time.time()
    print("time 2", end - start)

    data = read_flo_as_float32(os.path.join(test_dir, "0000000-gt.flo"))
    print("shape", data.shape)

    l_2 = np.linalg.norm(data - tmp)
    print("l2", l_2, "should be 2363.2214")


if __name__ == "__main__":
    test_flow_module()
