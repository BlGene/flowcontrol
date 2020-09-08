"""
Compute flow with Iterative Residual Refinement.
"""
import os
import time

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as vision_transforms

# do this to import irr
import gym_grasping.update_pythonpath  # noqa # pylint: disable=unused-import
from gym_grasping.utils import save_args, load_args
import irr.logger as logger
import irr.configuration as config
from irr.commandline import _parse_arguments, postprocess_args
# for demo
from irr.utils.flow import flow_to_png_middlebury
from irr.datasets.common import read_flo_as_float32


def tensor2float_dict(tensor_dict):
    """
    convert tesnor to dict of floats.
    """
    return {key: tensor.item() for key, tensor in tensor_dict.items()}


class FlowModule:
    """
    FlowModule usig IRR method.
    """

    def __init__(self, desc="Evaluation Epoch", size=None,
                 arg_file="./flow_module_IRR_args.json"):
        if __name__ == "__main__":
            # Parse commandline arguments
            args = overload_default_args()
        else:
            # Don't try to parse commandsline args (e.g. for with IPython)
            args = load_args(arg_file)

        self.method_name = args.model
        args = setup_logging_and_process_args(args)
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
            # reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # fast
            # torch.backends.cudnn.benchmark = True

        self._model_and_loss = model_and_loss

        # ---------------------------------------
        # Tell model that we want to evaluate
        # ---------------------------------------
        self._model_and_loss.eval()

    def step(self, image1, image2):
        """
        compute flow
        """
        tt_f = vision_transforms.ToTensor()
        example_dict = {"input1": tt_f(image1).unsqueeze_(0),
                        "input2": tt_f(image2).unsqueeze_(0)}
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
        flow_arr = output_dict["flow"].cpu().numpy()[0].transpose(1, 2, 0)
        return flow_arr


def overload_default_args(save_filename=None):
    """
    This function overloads the default args for our module.
    Setup logging and parse arguments for deploy network
    Returns:
        args object used by module
    """
    # ----------------------------------------------------------------------------
    # Get parse commandline and default arguments
    # ----------------------------------------------------------------------------
    irr_model = "IRR_PWC"
    irr_weights = "./saved_check_point/pwcnet/IRR-PWC_things3d/checkpoint_best.ckpt"

    args, _ = _parse_arguments()
    args.model = irr_model
    irr_dir = os.path.dirname(logger.__file__)
    args.checkpoint = os.path.join(irr_dir, irr_weights)
    args.checkpoint_include_params = "[*]"
    args.checkpoint_exclude_params = "[]"

    return args


def setup_logging_and_process_args(args):
    """
    sets up loggin and processes args.

    Args:
        unprocessed args (i.e. serializable)

    Returns:
        processed args (i.e. with classes)

    """
    # ----------------------------------------------------------------------------
    # Setup logbook before everything else
    # ----------------------------------------------------------------------------
    logger.configure_logging(filename=None)

    # ----------------------------------------------------------------------------
    # Postprocess
    # ----------------------------------------------------------------------------
    args = postprocess_args(args)
    return args


def test_flow_module():
    """
    Tests the flow module with examaple images.
    """
    # ---------------------------------------------------
    # Construct holistic recorder for epoch
    # ---------------------------------------------------
    flow_module = FlowModule(desc="Deploy")

    test_dir = "/home/argusm/lang/flownet2/data/FlyingChairs_examples"
    dict_fn = dict(img0='0000000-img0.ppm', img1='0000000-img1.ppm')
    for key, val in dict_fn.items():
        filename = os.path.join(test_dir, val)
        dict_fn[key] = filename
        assert os.path.isfile(filename)

    image1 = Image.open(dict_fn['img0'])
    image2 = Image.open(dict_fn['img1'])

    # run twice to get measurement without setup
    start = time.time()
    output = flow_module.step(image1, image2)
    end = time.time()
    print(end - start)

    data = read_flo_as_float32(os.path.join(test_dir, "0000000-gt.flo"))
    l_2 = np.linalg.norm(data-output, axis=2).mean()

    start = time.time()
    output = flow_module.step(image1, image2)
    end = time.time()
    print(end - start)

    png_middlebury = flow_to_png_middlebury(output.transpose(2, 0, 1))
    Image.fromarray(png_middlebury).save("test_flow.png")

    print(l_2)
    l_2 = np.linalg.norm(data-output, axis=2).mean()
    print(l_2)

    print("done.")


if __name__ == "__main__":
    arg_file = "./flow_module_IRR_args.json"
    if not os.path.isfile(arg_file):
        save_args(overload_default_args(), filename=arg_file)

    test_flow_module()
