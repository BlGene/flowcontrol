import os
import sys
import tempfile
from math import ceil

import numpy as np
from pdb import set_trace

try:
    import caffe
except ModuleNotFoundError:
    print("try: export PYTHONPATH=${PYTHONPATH}:/home/argusm/lang/flownet2/python")
    print("and: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/argusm/local/miniconda3/envs/bullet/lib")
    print('and export LD_LIBRARY_PATH="/misc/software/lmdb/mdb-mdb/libraries/liblmdb:${LD_LIBRARY_PATH}"')
    raise

# set the correct path here unless gym_grasping and flownet2 are in same dir
flownet2_path = None

import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed

def flownet2_path_guess():
    import gym_grasping
    grasping_dir = os.path.dirname(sys.modules['gym_grasping'].__file__)
    flownet2_dir = os.path.abspath(grasping_dir+"/../../flownet2")
    if os.path.isdir(flownet2_dir):
        return flownet2_dir
    else:
        print(grasping_dir,flownet2_dir)
        raise ValueError

if flownet2_path is None:
    flownet2_path = flownet2_path_guess()

class FlowModule:
    def __init__(self, size=(84,84)):
        height, width = size
        self.width = width
        self.height = height

        self.colorwheel = self.makeColorwheel()

        flownet_variant = "FlowNet2"
        caffemodel = "./models/{0}/{0}_weights.caffemodel.h5".format(flownet_variant)
        deployproto = "./models/{0}/{0}_deploy.prototxt.template".format(flownet_variant)
        caffemodel = os.path.join(flownet2_path, caffemodel)
        deployproto = os.path.join(flownet2_path, deployproto)

        vars = {}
        vars['TARGET_WIDTH'] = width
        vars['TARGET_HEIGHT'] = height

        divisor = 64.
        vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
        vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)
        vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
        vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

        proto = open(deployproto).readlines()
        for line in proto:
            for key, value in vars.items():
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
        num_blobs = 2
        input_data = []
        if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[self.net.inputs[blob_idx]] = input_data[blob_idx]
        self.net.forward(**input_dict)
        flow = np.squeeze(self.net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        return flow

    def makeColorwheel(self):

            #  color encoding scheme

            #   adapted from the color circle idea described at
            #   http://members.shaw.ca/quadibloc/other/colint.htm

            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR

            colorwheel = np.zeros([ncols, 3]) # r g b

            col = 0
            #RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
            col += RY

            #YG
            colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
            colorwheel[col:YG+col, 1] = 255;
            col += YG;

            #GC
            colorwheel[col:GC+col, 1]= 255
            colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
            col += GC;

            #CB
            colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
            colorwheel[col:CB+col, 2] = 255
            col += CB;

            #BM
            colorwheel[col:BM+col, 2]= 255
            colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
            col += BM;

            #MR
            colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
            colorwheel[col:MR+col, 0] = 255
            return 	colorwheel

    def computeColor(self, u, v):
            colorwheel = self.colorwheel
            u = np.nan_to_num(u)
            v = np.nan_to_num(v)

            ncols = colorwheel.shape[0]
            radius = np.sqrt(u**2 + v**2)
            a = np.arctan2(-v, -u) / np.pi
            fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
            k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
            k1 = k0+1;
            k1[k1 == ncols] = 0
            f = fk - k0

            img = np.empty([k1.shape[0], k1.shape[1],3])
            ncolors = colorwheel.shape[1]
            for i in range(ncolors):
                    tmp = colorwheel[:,i]
                    col0 = tmp[k0]/255
                    col1 = tmp[k1]/255
                    col = (1-f)*col0 + f*col1
                    idx = radius <= 1
                    col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius
                    col[~idx] *= 0.75 # out of range
                    img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

            return img.astype(np.uint8)

    def computeImg(self, flow, dynamic_range=True):
            eps = sys.float_info.epsilon

            if dynamic_range:
                UNKNOWN_FLOW_THRESH = 1e9
                UNKNOWN_FLOW = 1e10

                u = flow[: , : , 0]
                v = flow[: , : , 1]
                maxu = -999
                maxv = -999

                minu = 999
                minv = 999

                maxrad = -1
                #fix unknown flow
                greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
                greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
                u[greater_u] = 0
                u[greater_v] = 0
                v[greater_u] = 0
                v[greater_v] = 0

                maxu = max([maxu, np.amax(u)])
                minu = min([minu, np.amin(u)])

                maxv = max([maxv, np.amax(v)])
                minv = min([minv, np.amin(v)])

                rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
                maxrad = max([maxrad, np.amax(rad)])
                print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

                u = u/(maxrad+eps)
                v = v/(maxrad+eps)
            else:
                maxrad = (20*20+20*20)**.5
                flow_scaled = flow / (maxrad+eps)
                flow_scaled = np.clip(flow_scaled,-1,1)
                u = flow_scaled[: , : , 0]
                v = flow_scaled[: , : , 1]

            img = self.computeColor(u, v)
            return img


if __name__ == "__main__":
    module = FlowModule()
    import matplotlib.pyplot as plt

    field = module.field
    plt.imshow(np.linalg.norm(field,axis=2))
    plt.show()


    set_trace()



