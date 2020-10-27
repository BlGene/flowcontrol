"""
Compute flow using FlowNet2
"""
import os
import time
import tempfile
from math import ceil
import numpy as np
from PIL import Image

try:
    import caffe
except ModuleNotFoundError:
    print("try: export PYTHONPATH=${PYTHONPATH}:/home/argusm/lang/flownet2/python")
    print("and: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/argusm/local/miniconda3/envs/bullet/lib")
    print("and: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/misc/software/lmdb/mdb-mdb/libraries/liblmdb")
    raise

# set the correct path here unless gym_grasping and flownet2 are in same dir
FLOWNET2_PATH = None


def flownet2_path_guess():
    """
    get paths relative to flownet2 module location
    """
    path = os.path.dirname(os.path.abspath(__file__))
    flownet2_dir = os.path.abspath(os.path.join(path, "../../../flownet2"))

    if os.path.isdir(flownet2_dir):
        return flownet2_dir

    raise ValueError


if FLOWNET2_PATH is None:
    FLOWNET2_PATH = flownet2_path_guess()


class FlowModule:
    """
    Compute flow using FlowNet2 method
    """

    def __init__(self, size=(84, 84)):
        height, width = size
        self.width = width
        self.height = height

        flownet_variant = "FlowNet2"
        self.method_name = flownet_variant
        caffemodel = "./models/{0}/{0}_weights.caffemodel.h5".format(flownet_variant)
        deployproto = "./models/{0}/{0}_deploy.prototxt.template".format(flownet_variant)
        caffemodel = os.path.join(FLOWNET2_PATH, caffemodel)
        deployproto = os.path.join(FLOWNET2_PATH, deployproto)

        proto_vars = {}
        proto_vars['TARGET_WIDTH'] = width
        proto_vars['TARGET_HEIGHT'] = height

        divisor = 64.
        proto_vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
        proto_vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)
        proto_vars['SCALE_WIDTH'] = width / float(proto_vars['ADAPTED_WIDTH'])
        proto_vars['SCALE_HEIGHT'] = height / float(proto_vars['ADAPTED_HEIGHT'])

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

        proto = open(deployproto).readlines()
        for line in proto:
            for key, value in proto_vars.items():
                tag = "$%s$" % key
                line = line.replace(tag, str(value))

            tmp.write(line)

        tmp.flush()

        caffe.set_logging_disabled()
        caffe.set_device(0)
        caffe.set_mode_gpu()
        print("Loading flownet model, may take a bit...")
        self.net = caffe.Net(tmp.name, caffemodel, caffe.TEST)

    def step(self, img0, img1):
        """
        compute flow

        Args:
            img0: [h,w,3] dtype=uint8
            img1: [h,w,3] dtype=uint8
        Returns:
            flow: [h,w,2] dtype=float
        """
        num_blobs = 2
        input_data = []
        if len(img0.shape) < 3:
            input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:
            input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        if len(img1.shape) < 3:
            input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:
            input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[self.net.inputs[blob_idx]] = input_data[blob_idx]
        self.net.forward(**input_dict)
        flow = np.squeeze(self.net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        return flow


def read_flo_as_float32(filename):
    '''read .flo files'''
    with open(filename, 'rb') as file:
        magic = np.fromfile(file, np.float32, count=1)
        assert magic == 202021.25, "Magic number incorrect. Invalid .flo file"
        width = np.fromfile(file, np.int32, count=1)[0]
        height = np.fromfile(file, np.int32, count=1)[0]
        data = np.fromfile(file, np.float32, count=2*height*width)
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

    image1 = np.asarray(Image.open(dict_fn['img0']))
    image2 = np.asarray(Image.open(dict_fn['img1']))

    shape = image1.shape[0:2]
    flow_module = FlowModule(size=shape)

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

    l_2 = np.linalg.norm(data-tmp)
    print("l2", l_2, "should be 3339.7834")


if __name__ == "__main__":
    test_flow_module()
