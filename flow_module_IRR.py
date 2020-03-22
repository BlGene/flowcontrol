## Portions of Code from, copyright 2018 Jochen Gast
from __future__ import absolute_import, division, print_function

import os
import collections
import logging

import scipy.misc
import torch
import torch.nn as nn
import numpy as np

# Third party modules needed to import package dependencies
try:
    import png
except ImportError:
    print("try: pip install pypng")
    raise

import irr.configuration as config
import irr.commandline as commandline
import irr.tools as tools
from irr.tools import MovingAverage
# for evaluation
from irr.utils.flow import flow_to_png, flow_to_png_middlebury
from irr.utils.flow import write_flow, write_flow_png


from pdb import set_trace

def tensor2float_dict(tensor_dict):
    return {key: tensor.item() for key, tensor in tensor_dict.items()}

def setup_logging_and_parse_arguments_deploy():
    import irr.logger as logger
    from irr.commandline import _parse_arguments, postprocess_args
    # ----------------------------------------------------------------------------
    # Get parse commandline and default arguments
    # ----------------------------------------------------------------------------
    irr_dir = os.path.dirname(logger.__file__)
    irr_model = "IRR_PWC"
    irr_weights = "./saved_check_point/pwcnet/IRR-PWC_things3d/checkpoint_best.ckpt"

    args, defaults = _parse_arguments()
    args.model = irr_model
    args.checkpoint = os.path.join(irr_dir, irr_weights)
    args.checkpoint_include_params = "[*]"
    args.checkpoint_exclude_params = "[]"

    # ----------------------------------------------------------------------------
    # Setup logbook before everything else
    # ----------------------------------------------------------------------------
    logger.configure_logging(filename=None)

    # ----------------------------------------------------------------------------
    # Postprocess
    # ----------------------------------------------------------------------------
    args = postprocess_args(args)
    return args



class FlowModule:
    def __init__(self, desc="Evaluation Epoch", size=None):
        # Change working directory
        #os.chdir(os.path.dirname(os.path.realpath(__file__)))

        # Parse commandline arguments
        args = setup_logging_and_parse_arguments_deploy()

        self._args = args
        self._desc = desc

        # Set random seed, possibly on Cuda
        config.configure_random_seed(args)

        # Configure model and loss
        model_and_loss = config.configure_model_and_loss(args)

        # Resume from checkpoint if available
        checkpoint_saver, checkpoint_stats = config.configure_checkpoint_saver(args, model_and_loss)

        # Cuda auto-tune optimization
        if args.cuda:
            torch.backends.cudnn.benchmark = True


        self._model_and_loss = model_and_loss

        # ---------------------------------------
        # Tell model that we want to evaluate
        # ---------------------------------------
        self._model_and_loss.eval()


    def step(self, image1, image2):
        from torchvision import transforms as vision_transforms
        tt = vision_transforms.ToTensor()
        example_dict = {
            "input1": tt(image1).unsqueeze_(0),
            "input2": tt(image2).unsqueeze_(0) }
        # produces C, H, W

        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=False)

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        with torch.no_grad():
            output_dict = self._model_and_loss._model(example_dict)
        
        # -------------------------------------------------------------
        # Return output array
        # -------------------------------------------------------------
        flow_arr = output_dict["flow"].cpu().numpy()[0].transpose(1,2,0)
        return flow_arr


def test_flow_module():
    # ---------------------------------------------------
    # Construct holistic recorder for epoch
    # ---------------------------------------------------
    flow_module = FlowModule(desc="Deploy")

    from PIL import Image
    test_dir = "/home/argusm/lang/flownet2/data/FlyingChairs_examples"
    dict_fn = dict(img0='0000000-img0.ppm', img1='0000000-img1.ppm')

    for k, v in dict_fn.items():
        fn = os.path.join(test_dir, v)
        dict_fn[k] = fn
        assert(os.path.isfile(fn))


    image1 = Image.open(dict_fn['img0'])
    image2 = Image.open(dict_fn['img1'])
        
    import time
    start = time.time()
    output = flow_module.step(image1, image2)
    end = time.time()
    print(end - start)

    start = time.time()
    output = flow_module.step(image1, image2)
    end = time.time()
    print(end - start)
    print(output.shape)

    Image.fromarray(flow_to_png_middlebury(output)).save("test_flow.png")

    from irr.datasets.common import read_flo_as_float32
    data = read_flo_as_float32(os.path.join(test_dir, "0000000-gt.flo"))

    l2 = np.linalg.norm(data-output)

    print(l2)

    print("done.")

if __name__ == "__main__":
    test_flow_module()
