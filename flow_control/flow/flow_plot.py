"""Plot Optical Flow in colors"""
import sys
import numpy as np


class FlowPlot:
    '''Plot optical flow'''

    def __init__(self):
        self.colorwheel = self.make_color_wheel()

    @staticmethod
    def make_color_wheel():
        """
        create a colorwheel
        """
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

        colorwheel = np.zeros([ncols, 3])  # r g b

        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
        col += RY

        # YG
        colorwheel[col:YG+col, 0] = 255 - np.floor(255*np.arange(0, YG, 1)/YG)
        colorwheel[col:YG+col, 1] = 255
        col += YG

        # GC
        colorwheel[col:GC+col, 1] = 255
        colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
        col += GC

        # CB
        colorwheel[col:CB+col, 1] = 255 - np.floor(255*np.arange(0, CB, 1)/CB)
        colorwheel[col:CB+col, 2] = 255
        col += CB

        # BM
        colorwheel[col:BM+col, 2] = 255
        colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
        col += BM

        # MR
        colorwheel[col:MR+col, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
        colorwheel[col:MR+col, 0] = 255
        return colorwheel

    def compute_color(self, u, v):
        '''compute the flow colored image'''
        colorwheel = self.colorwheel
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

        ncols = colorwheel.shape[0]
        radius = np.sqrt(u**2 + v**2)
        a = np.arctan2(-v, -u) / np.pi
        fk = (a+1) / 2 * (ncols-1)  # -1~1 maped to 1~ncols
        k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
        k1 = k0+1
        k1[k1 == ncols] = 0
        f = fk - k0

        img = np.empty([k1.shape[0], k1.shape[1], 3])
        ncolors = colorwheel.shape[1]
        for i in range(ncolors):
            tmp = colorwheel[:, i]
            col0 = tmp[k0]/255
            col1 = tmp[k1]/255
            col = (1-f)*col0 + f*col1
            idx = radius <= 1
            # increase saturation with radius
            col[idx] = 1 - radius[idx]*(1-col[idx])
            col[~idx] *= 0.75  # out of range
            img[:, :, 2-i] = np.floor(255*col).astype(np.uint8)

        return img.astype(np.uint8)

    def compute_image(self, flow, dynamic_range=True):
        '''compute rgb colored image'''
        eps = sys.float_info.epsilon

        if dynamic_range:
            unknown_flow_thresh = 1e9

            u = flow[:, :, 0]
            v = flow[:, :, 1]
            maxu = -999
            maxv = -999

            minu = 999
            minv = 999

            maxrad = -1
            # fix unknown flow
            greater_u = np.where(u > unknown_flow_thresh)
            greater_v = np.where(v > unknown_flow_thresh)
            u[greater_u] = 0
            u[greater_v] = 0
            v[greater_u] = 0
            v[greater_v] = 0

            maxu = max([maxu, np.amax(u)])
            minu = min([minu, np.amin(u)])

            maxv = max([maxv, np.amax(v)])
            minv = min([minv, np.amin(v)])

            rad = np.sqrt(np.multiply(u, u)+np.multiply(v, v))
            maxrad = max([maxrad, np.amax(rad)])
            # print('max flow: %.4f flow range: u = %.3f .. %.3f;'
            #       'v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))
            u = u/(maxrad+eps)
            v = v/(maxrad+eps)
        else:
            maxrad = (20*20+20*20)**.5
            flow_scaled = flow / (maxrad+eps)
            flow_scaled = np.clip(flow_scaled, -1, 1)
            u = flow_scaled[:, :, 0]
            v = flow_scaled[:, :, 1]

        img = self.compute_color(u, v)
        return img
