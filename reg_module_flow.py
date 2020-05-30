import numpy as np

from gym_grasping.flow_control.flow_module_flownet2 import FlowModule as FlowN
from gym_grasping.flow_control.servoing_fitting import solve_transform


class RegistrationModule:
    '''Compute registration via flow based matching'''

    def __init__(self, size=None):
        assert size is not None
        self.size = size
        # load flow net (needs image size)
        self.flow_module = FlowN(size=size)
        self.method_name = self.flow_module.method_name

    def register(self, demo_rec, live_rec):
        '''get a transformation from a pointcloud.

        Params:
            demo_rec
            live_rec
        '''
        # 1. compute flow
        demo_rgb = demo_rec.get_color_frame()
        demo_depth = demo_rec.get_depth_frame()
        demo_mask = demo_rec.get_seg_frame()
        live_rgb = live_rec.get_color_frame()
        live_depth = live_rec.get_depth_frame()

        assert demo_rgb.shape == live_rgb.shape
        flow = self.flow_module.step(demo_rgb, live_rgb)
        # flow_image = self.flow_module.compute_image(flow)
        # plt.imshow(flow_image)
        # plt.show()

        # 2. compute transformation (live/start -> demo/end)
        # 2.1 get end pointcloud
        end_points = np.array(np.where(demo_mask)).T
        end_pc = demo_rec.generate_pointcloud(demo_rgb, demo_depth, end_points)

        # 2.2 get start pointcloud
        masked_flow = flow[demo_mask]
        start_points = end_points + masked_flow[:, ::-1].astype('int')
        start_pc = live_rec.generate_pointcloud(live_rgb, live_depth,
                                                start_points)

        # 2.3 get points present in both start end
        mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)
        # subsample points
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]

        guess = solve_transform(start_pc[:, 0:4], end_pc[:, 0:4])
        return guess
