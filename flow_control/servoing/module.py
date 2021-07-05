"""
This is a stateful module that contains a recording, then
given a  query RGB-D image it outputs the estimated relative
pose. This module also handles incrementing along the recording.
"""
import copy
import logging
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R
from flow_control.servoing.demo import ServoingDemo
from flow_control.servoing.fitting import solve_transform
from flow_control.servoing.fitting_ransac import Ransac
from flow_control.rgbd_camera import RGBDCamera
from gym_grasping.envs.camera import PyBulletCamera
from pdb import set_trace

try:
    from flow_control.servoing.live_plot import ViewPlots, SubprocPlot
except ImportError:
    # Don't call logging here because it overwrites log level.
    SubprocPlot, ViewPlots = None, None


# magical gain values for dof, these could come from calibration
DEFAULT_CONF = dict(mode="pointcloud",
                    gain_xy=100,
                    gain_z=50,
                    gain_r=30,
                    threshold=0.20)


class ServoingTooFewPointsError(Exception):
    """Raised when we have too few points to fit"""
    pass


class ServoingModule:
    """
    This is a stateful module that contains a recording, then
    given a  query RGB-D image it outputs the estimated relative
    pose. This module also handles incrementing along the recording.
    """

    def __init__(self, recording, episode_num=0, start_index=0,
                 control_config=None, plot=False, save_dir=False):
        # Moved here because this can require caffe
        try:
            from flow_control.flow.module_raft import FlowModule
        except ModuleNotFoundError:
            from flow_control.flow.module_flownet2 import FlowModule
        # from flow_control.flow.module_IRR import FlowModule
        # from flow_control.reg.module_FGR import RegistrationModule

        self.demo = ServoingDemo(recording, episode_num, start_index)
        self.demo_cam = RGBDCamera(self.demo.env_info['camera'])

        self.calibration_checked = False
        self.live_cam = None

        self.size = self.demo.rgb_recording.shape[1:3]
        # load flow net (needs image size)
        self.flow_module = FlowModule(size=self.size)
        self.method_name = self.flow_module.method_name
        # self.reg_module = RegistrationModule()
        # self.method_name = "FGR"

        # get config dict
        if control_config is None:
            config = DEFAULT_CONF
        else:
            config = control_config
        self.config = SimpleNamespace(**config)

        # plotting
        self.cache_flow = None
        self.view_plots = False
        if plot and ViewPlots is None:
            logging.warning("Servoing Plot: ignoring plot=True, as import failed")
        elif plot:
            self.view_plots = ViewPlots(threshold=self.config.threshold,
                                        save_dir=save_dir)
        # vars set in reset
        self.counter = None
        self.action_queue = None
        self.reset()

    def set_T_cam_tcp(self, live_cam):
        # Ideally we load the values from demonstrations and live and compare
        # for this the demonstration info would need to include them

        # These values come from a) recording realsense images (flip=True)
        # b) calibrating the simulation to real views c) getting the T from sim
        T_cam_tcp_sim = np.array(
           [[9.99801453e-01, -1.81777984e-02, 8.16224931e-03, 2.77370419e-03],
            [1.99114100e-02, 9.27190979e-01, -3.74059384e-01, 1.31238638e-01],
            [-7.68387855e-04, 3.74147637e-01, 9.27368835e-01, -2.00077483e-01],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

        # These values come from a) recording realsense images (flip=False)
        # b) marker based calibration
        T_cam_tcp_calib = np.array(
           [[-0.99862, 0.03984,   0.03425,  0.00103],
            [-0.05165, -0.86398, -0.50086,  0.1244 ],
            [ 0.00964, -0.50194,  0.86485, -0.19987],
            [ 0.     ,  0.     ,  0.     ,  1.     ]])

        if isinstance(live_cam, PyBulletCamera):
            self.T_cam_tcp = T_cam_tcp_sim
            logging.warning("Using deprecated values for T_cam_tcp")
        else:
            if live_cam.flip_horizontal:
                logging.warning("Live flipping true: using T_cam_tcp from sim")
                self.T_cam_tcp = T_cam_tcp_sim
            else:
                logging.info("Live flipping false: using T_cam_tcp from calibration")
                self.T_cam_tcp = T_cam_tcp_calib
                if self.demo_cam.flip_horizontal:
                    self.demo.flip()
                    logging.warning("Demo flipping true: converting to match env.")

            # demo_trf = self.demo_cam.T_cam_tcp
            # live_trf = live_cam.T_cam_tcp
            # assert np.linalg.norm(demo_trf - live_trf) < .002
            assert self.T_cam_tcp is not None

    def set_and_check_cam(self, live_cam):
        """
        Compare demonstration calibration to live calibration
        """
        # This is needed because we use demo_cam.
        live_calib = live_cam.calibration
        demo_calib = self.demo_cam.calibration
        for key in live_calib:
            if demo_calib[key] != live_calib[key]:
                logging.warning(f"Calibration: {key} demo!=live  {demo_calib[key]} != {live_calib[key]}")

        self.set_T_cam_tcp(live_cam)
        self.calibration_checked = True

    def reset(self):
        """
        reset servoing, reset counter and index
        """
        self.counter = 0
        self.action_queue = []
        self.demo.reset()
        if self.view_plots:
            self.view_plots.reset()

    def get_trajectory_actions(self, info):
        """
        Returns:
            pre_actions: list of [(name, val), ...]
        """
        try:
            pre_actions = self.demo.keep_dict[self.demo.frame]["pre"]
        except KeyError:
            return info
        if type(pre_actions) == dict:
            pre_actions = list(pre_actions.items())

        return pre_actions

    def step(self, live_rgb, live_state, live_depth=None):
        """
        Main loop, this does sequence alignment.

        Usually what frame alignment gives, but sometimes something else.

        Arguments:
            live_rgb: live rgb image
            live_state: live state from robot
            live_depth: live depth image

        Returns:
            action: (x, y, z, r, g)
            done: binary if demo sequence is completed
            info: dict
        """
        assert np.asarray(live_state).ndim == 1

        try:
            align_transform, align_q = self.frame_align(live_rgb, live_depth)
        except ServoingTooFewPointsError:
            align_transform, align_q = np.eye(4), 999

        # this returns a relative action
        rel_action, loss = self.trf_to_act_loss(align_transform, live_state)

        if self.config.mode == "pointcloud":
            action = rel_action

        elif self.config.mode == "pointcloud-abs":
            move_g = self.demo.grip_action
            action = [align_transform, move_g]
        else:
            raise ValueError

        # debug output
        loss_str = "{:04d} loss {:4.4f}".format(self.counter, loss)
        action_str = " action: " + " ".join(['%4.2f' % a for a in rel_action])
        action_str += " "+"-".join([list(x.keys())[0] for x in self.action_queue])
        logging.debug(loss_str + action_str)

        if self.view_plots:
            series_data = (loss, self.demo.frame, align_q, live_state[0])
            self.view_plots.step(series_data, live_rgb, self.demo.rgb,
                                 self.cache_flow, self.demo.mask, rel_action)

        try:
            force_step = self.demo.keep_dict[self.demo.frame]["grip_dist"] > 1
        except TypeError:
            force_step = False

        info = {}
        done = False
        if (loss < self.config.threshold) or force_step:
            if self.demo.frame < self.demo.max_frame:
                self.demo.step()
                info = self.get_trajectory_actions(info)
                # debug
                step_str = "start: {} / {}".format(self.demo.frame, self.demo.max_frame)
                step_str += " step {} ".format(self.counter)
                logging.debug(step_str)
            elif self.demo.frame == self.demo.max_frame:
                done = True

        self.counter += 1
        return action, done, info

    def trf_to_act_loss(self, align_transform, live_state):
        """
        Arguments:
            align_transform: transform that aligns demo to live
            live_state: current live robot state

        Returns:
            rel_action: currently (x, y, z, r, g)
            loss: scalar usually between ~5 and ~0.2
        """
        T_cam_tcp = self.T_cam_tcp
        align_transform = T_cam_tcp @ align_transform @ np.linalg.inv(T_cam_tcp)

        d_x = align_transform[0, 3]
        d_y = align_transform[1, 3]
        rot_z = R.from_matrix(align_transform[:3, :3]).as_euler('xyz')[2]

        move_xy = self.config.gain_xy*d_x, -self.config.gain_xy*d_y
        move_z = -1*self.config.gain_z*(live_state[2] - self.demo.state[2])
        move_rot = -self.config.gain_r*rot_z
        move_g = self.demo.grip_action

        # This was found to work well based on a bit of experimentation
        loss_xy = np.linalg.norm(move_xy)
        loss_z = np.abs(move_z)/3
        loss_rot = np.abs(move_rot)*3
        loss = loss_xy + loss_rot + loss_z

        rel_action = [*move_xy, move_z, move_rot, move_g]

        return rel_action, loss

    def frame_align(self, live_rgb, live_depth):
        """
        Get a transformation from two pointclouds.

        Arguments:
            live_rgb: image
            live_depth: image
        Returns:
            T_in_tcp: 4x4 homogeneous transformation matrix
            fit_q: scalar fit quality, lower is better
        """
        # this should probably be (480, 640, 3)
        assert live_depth is not None
        assert self.demo.depth is not None
        assert live_rgb.shape == self.demo.rgb.shape

        # 1. compute flow
        flow = self.flow_module.step(self.demo.rgb, live_rgb)
        self.cache_flow = flow

        # 2. compute transformation
        masked_flow = flow[self.demo.mask]
        end_points = np.array(np.where(self.demo.mask)).T
        # TODO(max): add rounding before casting
        start_points = end_points + masked_flow[:, ::-1].astype('int')
        start_pc = self.demo_cam.generate_pointcloud(live_rgb, live_depth, start_points)
        end_pc = self.demo_cam.generate_pointcloud(self.demo.rgb, self.demo.depth, end_points)
        mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)

        # subsample fitting, maybe evaluate with ransac
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]

        pc_min_size = 32
        if len(start_pc) < pc_min_size or len(end_pc) < pc_min_size:
            logging.warning("Too few points, skipping fitting")
            raise ServoingTooFewPointsError

        # 3. estimate trf and transform to TCP coordinates
        # estimate T, put in non-homogenous points, get homogeneous trf.
        # trf_est = solve_transform(start_pc, end_pc)
        def eval_fit(trf_estm, start_ptc, end_ptc):
            start_m = (trf_estm @ start_ptc[:, 0:4].T).T
            fit_qe = np.linalg.norm(start_m[:, :3]-end_ptc[:, :3], axis=1)
            return fit_qe

        ransac = Ransac(start_pc, end_pc, solve_transform, eval_fit,
                        .005, 5)
        fit_qc, trf_est = ransac.run()

        # Compute fit quality
        # TODO(max): get rid of all of these Transposes...

        # Compute flow quality via color
        fit_qc = np.linalg.norm(start_pc[:, 4:7]-end_pc[:, 4:7], axis=1)

        # if self.counter > 60:
        # self.debug_show_fit(start_pc, end_pc, trf_est)

        return trf_est, fit_qc.mean()

    def debug_show_fit(self, start_pc, end_pc, trf_est):
        import open3d as o3d
        pre_q = np.linalg.norm(start_pc[:, :4]-end_pc[:, :4], axis=1)
        start_m = (trf_est @ start_pc[:, 0:4].T).T
        fit_qe = np.linalg.norm(start_m[:, :3]-end_pc[:, :3], axis=1)

        # Compute flow quality via positions
        print(pre_q.mean(), "->", fit_qe.mean())

        colors = start_pc[:, 4:7]/255.  # recorded image colors
        # import matplotlib.pyplot as plt
        # cmap = plt.get_cmap()  # color according to error
        # fit_qe_n = (fit_qe - fit_qe.min()) / fit_qe.ptp()
        # colors = fit_qe_n
        # colors = cmap(fit_qe_n)[:, :3]

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(start_pc[:, :3])
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        # pcd1.transform(trf_est)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(end_pc[:, :3])
        pcd2.colors = o3d.utility.Vector3dVector(end_pc[:, 4:7]/255.)

        o3d.visualization.draw_geometries([pcd1, pcd2])
        self.draw_registration_result(pcd1, pcd2, trf_est)

    @staticmethod
    def draw_registration_result(source, target, transformation):
        """
        plot registration results using o3d
        """
        import open3d as o3d
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    @staticmethod
    def cmd_to_action(env, name, val, prev_servo_action):
        rot = env.robot.get_tcp_angles()[2]

        if name == "grip":
            servo_control = env.robot.get_control("absolute", min_iter=24)
            pos = env.robot.get_tcp_pos()
            servo_action = [*pos, rot, val]

        elif name == "abs":
            servo_control = env.robot.get_control("absolute")
            # TODO(sergio): add rotation to abs and rel motions
            # rot = np.pi + R.from_quat(val[3:7]).as_euler("xyz")[2]
            servo_action = [*val[0:3], rot, prev_servo_action[-1]]

        elif name == "rel":
            servo_control = env.robot.get_control("absolute")
            new_pos = np.array(env.robot.get_tcp_pos()) + val[0:3]
            # rot = rot + R.from_quat(val[3:7]).as_euler("xyz")[2]
            servo_action = [*new_pos, rot, prev_servo_action[-1]]

        else:
            raise ValueError

        return servo_action, servo_control

    @staticmethod
    def abs_to_action(env, servo_action, info):
        # 1. get the robot state (ideally from when image was taken)
        t_world_tcp = tcp_pose_from_info(info)

        # 2. get desired transformation from fittings.
        t_camdemo_camlive, grip_action = servo_action

        # 3. compute desired goal in world.
        T_cam_tcp = servo_module.T_cam_tcp
        t_tcpdemo_tcplive = inv(T_cam_tcp) @ t_camdemo_camlive @ T_cam_tcp
        goal_pose = t_world_tcpnew = inv(t_tcpdemo_tcplive @ inv(t_world_tcp))

        goal_pos = goal_pose[:3, 3]
        goal_angles = R.from_matrix(goal_pose[:3, :3]).as_euler("xyz")

        direct = True
        if direct and not is_sim:
            coords = (goal_pos[0], goal_pos[1], goal_pos[2], math.pi, 0, goal_angles[2])
            env.robot.send_cartesian_coords_abs_PTP(coords)
            servo_action, servo_control = None, None
        else:
            servo_action = goal_pos.tolist() + [goal_angles[2], grip_action]
            servo_control = env.robot.get_control("absolute")

        print(t_camdemo_camlive[:3, 3].round(3), (goal_pos - t_world_tcp[:3, 3]).round(3))
        env.p.removeAllUserDebugItems()
        env.p.addUserDebugLine([0, 0, 0], T_WORLD_TCP[:3, 3], lineColorRGB=[1, 0, 0],
                               lineWidth=2, physicsClientId=0)  # red line to flange
        env.p.addUserDebugLine([0, 0, 0], goal_pos, lineColorRGB=[0, 1, 0],
                               lineWidth=2, physicsClientId=0)  # green line to object
        return servo_action, servo_control

