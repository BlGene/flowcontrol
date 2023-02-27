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
    import unimatch.unimatch as U
    from utils.utils import InputPadder

except ModuleNotFoundError as error:
    unimatch_export_cmd = "export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/unimatch/"
    print(f"Unimatch module not found.\n try: {unimatch_export_cmd}")
    raise ModuleNotFoundError(
        f"Unimatch module not found. Install & try: {unimatch_export_cmd}"
    ) from error

torch.backends.cudnn.benchmark = True

class FlowModule:
    """
    Compute flow using RAFT method
    """
    def __init__(self, size=None):

        parser = argparse.ArgumentParser()

        # model: learnable parameters
        parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
        parser.add_argument('--num_scales', default=1, type=int,
                            help='feature scales: 1/8 or 1/8 + 1/4')
        parser.add_argument('--feature_channels', default=128, type=int)
        parser.add_argument('--upsample_factor', default=8, type=int)
        parser.add_argument('--num_head', default=1, type=int)
        parser.add_argument('--ffn_dim_expansion', default=4, type=int)
        parser.add_argument('--num_transformer_layers', default=6, type=int)
        parser.add_argument('--reg_refine', action='store_true',
                            help='optional task-specific local regression refinement')

        # model: parameter-free
        parser.add_argument('--attn_type', default='swin', type=str,
                            help='attention function')
        parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                            help='number of splits in attention')
        parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                            help='correlation radius for matching, -1 indicates global matching')
        parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                            help='self-attention radius for propagation, -1 indicates global attention')
        parser.add_argument('--num_reg_refine', default=1, type=int,
                            help='number of additional local regression refinement')

        args = parser.parse_args([])

        self.method_name = "Unimatch"
        self.attn_type = args.attn_type
        self.attn_splits_list = args.attn_splits_list
        self.corr_radius_list = args.corr_radius_list
        self.prop_radius_list = args.prop_radius_list
        self.num_reg_refine = args.num_reg_refine

        model = U.UniMatch(feature_channels=args.feature_channels,
                           num_scales=args.num_scales,
                           upsample_factor=args.upsample_factor,
                           num_head=args.num_head,
                           ffn_dim_expansion=args.ffn_dim_expansion,
                           num_transformer_layers=args.num_transformer_layers,
                           reg_refine=args.reg_refine,
                           task=args.task).cuda()

        unimatch_root = Path(U.__file__).parent.parent
        unimatch_root = unimatch_root / 'models' / 'gmflow-scale1-mixdata.pth'

        logging.info("Loading UniMatch model, may take a bit...")
        try:
            checkpoint = torch.load(unimatch_root)
            model.load_state_dict(checkpoint['model'])
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "UniMatch weights not found in \'" + str(unimatch_root) + "\', "
                + "did you download them to \'unimatch/models/*\'") from error

        model = model.cuda()
        model = model.eval()
        self.model = model

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

            results_dict = self.model(image1, image2,
                                      attn_type=self.attn_type,
                                      attn_splits_list=self.attn_splits_list,
                                      corr_radius_list=self.corr_radius_list,
                                      prop_radius_list=self.prop_radius_list,
                                      num_reg_refine=self.num_reg_refine,
                                      task='flow')

            flow_preds = results_dict['flow_preds']
            return padder.unpad(flow_preds[0][0]).permute(1, 2, 0).detach().cpu().numpy()

    def warp(self, x, flow, mode="bilinear"):
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
        mask = torch.ones(x.size()).cuda()

        if mode == "bilinear":
            output = torch.nn.functional.grid_sample(x, vgrid)
            mask = torch.nn.functional.grid_sample(mask, vgrid)
        elif mode == "bicubic":
            output = torch.nn.functional.grid_sample(x, vgrid, mode="bicubic")
            output = torch.clip(output, 0, 255)
            mask = torch.nn.functional.grid_sample(mask, vgrid, mode="bicubic")
        else:
            raise ValueError

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

    test_dir = "/home/argusm/Downloads/"
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
    print("l2", l_2, "should be 2472.06")


if __name__ == "__main__":
    test_flow_module()
