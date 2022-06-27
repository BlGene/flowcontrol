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

from flow_control.servoing.playback_env_servo import PlaybackEnvServo
from flow_control.servoing.fitting import solve_transform
from flow_control.servoing.fitting_ransac import Ransac
from flow_control.rgbd_camera import RGBDCamera
from flow_control.utils_coords import pos_orn_to_matrix, matrix_to_pos_orn

try:
    from gym_grasping.envs.robot_sim_env import RobotSimEnv
except ImportError:
    RobotSimEnv = None

try:
    from flow_control.servoing.live_plot import ViewPlots, SubprocPlot
except ImportError:
    # Don't call logging here because it overwrites log level.
    SubprocPlot, ViewPlots = None, None

from pdb import set_trace


# magical gain values for dof, these could come from calibration
DEFAULT_CONF = dict(mode="pointcloud",
                    gain_xy=100,
                    gain_z=50,
                    gain_r=15,
                    threshold=0.20)


class ServoingTooFewPointsError(Exception):
    """Raised when we have too few points to fit"""


class ServoingModule:
    """
    This is a stateful module that contains a recording, then
    given a  query RGB-D image it outputs the estimated relative
    pose. This module also handles incrementing along the recording.
    """

    def __init__(self, recording, start_index=0,
                 control_config=None, plot=False, save_dir=False,
                 start_paused=False):
        """
        Arguments:
            start_paused: this computes actions and losses, but returns None
                          actions
        """
        # Moved here because this requires raft
        from flow_control.flow.module_raft import FlowModule
        # from flow_control.flow.module_flownet2 import FlowModule
        # from flow_control.flow.module_IRR import FlowModule
        # from flow_control.reg.module_FGR import RegistrationModule
        self.is_live_sim = None
        if isinstance(recording, PlaybackEnvServo):
            self.demo = recording
        else:
            self.demo = PlaybackEnvServo(recording)
            logging.info("Loaded recording: %s", recording)
        self.demo_cam = RGBDCamera(self.demo.cam)
        assert isinstance(self.demo_cam.calibration, dict)

        self.live_cam = None
        self.calibration_checked = False

        # load flow net (needs image size)
        self.flow_module = FlowModule(size=self.demo.cam.resolution)
        self.method_name = self.flow_module.method_name
        # self.reg_module = RegistrationModule()
        # self.method_name = "FGR"

        config = DEFAULT_CONF
        if control_config is not None:
            config.update(control_config)
        self.config = SimpleNamespace(**config)

        # plotting
        self.cache_flow = None
        self.view_plots = False
        if plot and ViewPlots is None:
            logging.warning("Servoing Plot: ignoring plot=True, as import failed")
        elif plot:
            self.view_plots = ViewPlots(threshold=self.config.threshold,
                                        save_dir=save_dir)
        if start_paused:
            if self.view_plots is False:
                logging.warning("Servoing Module: swtiching start_paused -> False as plots not active")
                start_paused = False
            logging.info("Starting paused.")
        elif self.view_plots:
            self.view_plots.started = True
        self.paused = start_paused

        # vars set in reset
        self.counter = None
        self.action_queue = None
        self.reset()

    def set_T_tcp_cam(self, live_cam, env=None):
        """
        Set and check the T_tcp_cam variable.
        """
        # Ideally we load the values from demonstrations and live and compare
        # for this the demonstration info would need to include them

        self.T_tcp_cam = self.demo_cam.T_tcp_cam

        # TODO(sergio): check T_tcp_cam matches
        # live_T_tcp_cam = live_cam.get_extrinsic_calibration()
        # demo_T_tcp_cam = self.demo_cam.T_tcp_cam

        #assert np.linalg.norm(live_T_tcp_cam - demo_T_tcp_cam) < .002

        #try:
        #    demo_trf = self.demo_cam.T_tcp_cam
        #except AttributeError:
        #    demo_trf = live_cam.T_tcp_cam
        #live_trf = live_cam.T_tcp_cam
        #assert np.linalg.norm(demo_trf - live_trf) < .002
        #self.T_tcp_cam = live_trf
        #    logging.warning("Demo flipping true: converting to match env.")

    def set_env(self, env):
        """
        This checks to see that the env matches the demonstration.
        """
        # workaround for testing servoing between demonstration frames.

        if env == "demo":
            name = self.demo.env_info["name"]

            if name == "RobotSimEnv":
                self.is_live_sim = True
            elif name == "IIWAEnv":
                self.is_live_sim = False

            self.T_tcp_cam = self.demo.env_info["camera"]["T_tcp_cam"]
            return

        # This is needed because we use demo_cam.
        self.is_live_sim = isinstance(env, RobotSimEnv)

        # live_cam = env.camera
        # live_calib = env.camera.calibration

        live_cam = env.camera_manager.gripper_cam
        live_calib = live_cam.get_intrinsics()
        demo_calib = self.demo_cam.calibration

        # TODO(sergio): check calibration matches.
        for key in ['width', 'height', 'fx', 'fy', 'cx', 'cy']:
            if demo_calib[key] != live_calib[key]:
                logging.warning(f"Calibration: %s demo!=live %s != %s", key, demo_calib[key], live_calib[key])
        self.set_T_tcp_cam(live_cam, env)
        self.calibration_checked = True

    def reset(self):
        """
        reset servoing, reset counter and index
        """
        self.counter = 0
        self.action_queue = []
        self.demo.reset()

        demo_rgb = self.demo.cam.get_image()[0]

        if self.view_plots:
            self.view_plots.reset()

    def get_trajectory_actions(self, info):
        """
        Returns:
            pre_actions: list of [(name, val), ...]

        with:
            name: "grip"; "abs"; or "rel"
            val: 1,-1; 8-tuple, 8-tuple
        """
        try:
            pre_actions = self.demo.get_keep_dict()["pre"]

        except KeyError:
            return info
        if isinstance(pre_actions, dict):
            pre_actions = list(pre_actions.items())
        return pre_actions

    def pause(self):
        if self.view_plots:
            self.view_plots.started = False

    @staticmethod
    def process_obs(live_state, live_info):
        """
        Returns:
            live_rgb: live rgb image
            live_tcp: live tcp position, shape (6, )
            live_depth: live depth image
        """
        obs_image = live_state["rgb_gripper"]
        ee_pos = live_state["robot_state"]["tcp_pos"]
        live_depth = live_state["depth_gripper"]
        world_tcp = pos_orn_to_matrix(live_state["robot_state"]["tcp_pos"],
                                      live_state["robot_state"]["tcp_orn"])
        live_info["world_tcp"] = world_tcp
        return obs_image, ee_pos, live_depth

    @staticmethod
    def project_rot_z(goal_pos, goal_quat, t_world_tcp):
        """
        Project rotation onto Z axis
        """
        goal_xyz = R.from_matrix(t_world_tcp[:3, :3]).as_euler('xyz')
        # curr_xyz = R.from_matrix(info['world_tcp'][:3, :3]).as_euler('xyz')
        curr_xyz = R.from_quat([1, 0, 0, 0]).as_euler('xyz')
        goal_quat = R.from_euler('xyz', [curr_xyz[0], curr_xyz[1], goal_xyz[2]]).as_quat()
        return goal_pos, goal_quat

    def step(self, live_state, live_info):
        """
        Main loop, this does sequence alignment.

        Usually what frame alignment gives, but sometimes something else.

        Arguments:
            live_state: image array if is_live_sim,
            live_info: dict with keys tcp_pose, depth

        Returns:
            action: (x, y, z, r, g)
            done: binary if demo sequence is completed
            info: dict with keys:
                "align_trf"
                "grip_action"
        """
        live_rgb, live_tcp, live_depth = self.process_obs(live_state, live_info)
        assert np.asarray(live_tcp).ndim == 1

        try:
            align_transform, align_q = self.frame_align(live_rgb, live_depth)
        except ServoingTooFewPointsError:
            align_transform, align_q = np.eye(4), 999

        # this returns a relative action
        rel_action, loss = self.trf_to_act_loss(align_transform, live_tcp)

        if self.config.mode == "pointcloud":
            action = rel_action
        elif self.config.mode == "pointcloud-abs":
            # TODO(max): refactor this to make the function dictionary free.
            t_world_tcp = self.abs_to_world_tcp({"align_trf":align_transform}, live_info)

            goal_pos, goal_quat = matrix_to_pos_orn(t_world_tcp)

            # project the rotation to keep only the z component.
            # TODO(max): if servoing is unstable, try uncommenting this.
            goal_pos, goal_quat = self.project_rot_z(goal_pos, goal_quat, t_world_tcp)

            # TODO(max/abhijeet): add projection function to tilted orientation.

            # what we want to do here is add a projection of the
            # orientation to be th
            grip_action = self.demo.get_action("gripper")
            action = [*goal_pos, *goal_quat, grip_action]
        else:
            raise ValueError

        # debug output
        loss_str = f"{self.counter:04d} loss {loss:4.4f}"
        action_str = " action: " + " ".join([f'{a:4.2f}' for a in rel_action])
        action_str += " " + "-".join([list(x.keys())[0] for x in self.action_queue])
        logging.debug(loss_str + action_str)

        if self.view_plots:
            frame = self.demo.index
            demo_rgb = self.demo.cam.get_image()[0]
            demo_mask = self.demo.get_fg_mask()

            series_data = (loss, frame, align_q, live_tcp[0])
            self.view_plots.step(series_data, live_rgb, demo_rgb,
                                 self.cache_flow, demo_mask, rel_action)

            self.paused = not self.view_plots.started

        demo_info = self.demo.get_keep_dict()

        force_step = False
        try:
            if demo_info["skip"]:
                force_step = True
            if demo_info["grip_dist"] < 2:
                threshold = self.config.threshold
            else:
                threshold = self.config.threshold * 1.2
        except TypeError:
            force_step = False
            threshold = self.config.threshold

        print(f"Loss: {loss:.4f} step={int(loss < threshold or force_step)} demo_frame={self.demo.index}")

        grip_action = self.demo.get_action("gripper")

        info = {"align_trf": align_transform, "grip_action": grip_action}
        if self.paused:
            return None, False, info

        done = False
        if loss < threshold or force_step:
            demo_frame = self.demo.index
            demo_max_frame = len(self.demo) - 1

            if self.demo.index < demo_max_frame:
                self.demo.step()

                info["traj_acts"] = self.get_trajectory_actions(info)
                # debug output
                step_str = f"start: {self.demo.index} / {demo_max_frame}"
                step_str += f" step {self.counter} "
                logging.debug(step_str)

            elif self.demo.index  == demo_max_frame:
                done = True

        self.counter += 1

        pause_on_step = True
        if self.view_plots and pause_on_step:
            self.view_plots.started = False

        return action, done, info

    def trf_to_act_loss(self, align_transform, live_tcp):
        """
        Arguments:
            align_transform: transform that aligns demo to live, shape (4, 4)
            live_tcp: live tcp position, shape (6, )

        Returns:
            rel_action: currently (x, y, z, r, g)
            loss: scalar usually between ~5 and ~0.2
        """
        t_tcp_cam = self.T_tcp_cam
        demo_tcp_z = self.demo.robot.get_tcp_pos()[2]

        align_transform = t_tcp_cam @ align_transform @ np.linalg.inv(t_tcp_cam)

        d_x = align_transform[0, 3]
        d_y = align_transform[1, 3]
        rot_z = R.from_matrix(align_transform[:3, :3]).as_euler('xyz')[2]

        move_xy = self.config.gain_xy * d_x, -self.config.gain_xy * d_y
        move_z = -1 * self.config.gain_z * (live_tcp[2] - demo_tcp_z)
        move_rot = -self.config.gain_r * rot_z

        move_g = self.demo.get_action("gripper")

        # This was found to work well based on a bit of experimentation
        loss_xy = np.linalg.norm(move_xy)
        loss_z = np.abs(move_z) / 3
        loss_rot = np.abs(move_rot) * 3
        loss = loss_xy + loss_rot + loss_z


        print(f"loss_xy {loss_xy:.4f}, loss_rot {loss_rot:.4f}, loss_z {loss_z:.4f}, rot_z {rot_z:.4f}")

        rel_action = [*move_xy, move_z, move_rot, move_g]

        return rel_action, loss

    def frame_align(self, live_rgb, live_depth):
        """
        Get a transformation from two pointclouds and a demonstration mask.

        Arguments:
            live_rgb: image
            live_depth: image
        Returns:
            T_in_tcp: 4x4 homogeneous transformation matrix
            fit_q: scalar fit quality, lower is better
        """
        # this should probably be (480, 640, 3)
        assert live_depth is not None

        demo_rgb, demo_depth = self.demo.cam.get_image()
        demo_mask = self.demo.get_fg_mask()

        assert demo_depth is not None
        assert live_rgb.shape == demo_rgb.shape

        # 1. compute flow
        flow = self.flow_module.step(demo_rgb, live_rgb)
        self.cache_flow = flow

        # 2. compute transformation
        masked_flow = flow[demo_mask]
        end_points = np.array(np.where(demo_mask)).T
        # TODO(max): add rounding before casting
        start_points = end_points + masked_flow[:, ::-1].astype('int')
        start_pc = self.demo_cam.generate_pointcloud(live_rgb, live_depth, start_points)
        end_pc = self.demo_cam.generate_pointcloud(demo_rgb, demo_depth, end_points)
        mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)

        # subsample fitting, maybe evaluate with ransac
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]

        pc_min_size_t = 32
        pc_min_size = min(len(start_pc), len(end_pc))
        if pc_min_size < pc_min_size_t:
            logging.warning("Too few points %s skipping fitting, t=%s", pc_min_size, pc_min_size_t)
            raise ServoingTooFewPointsError

        # 3. estimate trf and transform to TCP coordinates
        # estimate T, put in non-homogenous points, get homogeneous trf.
        # trf_est = solve_transform(start_pc, end_pc)
        def eval_fit(trf_estm, start_ptc, end_ptc):
            start_m = (trf_estm @ start_ptc[:, 0:4].T).T
            fit_qe = np.linalg.norm(start_m[:, :3] - end_ptc[:, :3], axis=1)
            return fit_qe

        ransac = Ransac(start_pc, end_pc, solve_transform, eval_fit, .005, 5)
        fit_qc, trf_est = ransac.run()

        # Compute fit quality via color
        fit_qc = np.linalg.norm(start_pc[:, 4:7] - end_pc[:, 4:7], axis=1)

        # if self.counter > 60:
        #   self.debug_show_fit(start_pc, end_pc, trf_est)

        return trf_est, fit_qc.mean()

    def debug_show_fit(self, start_pc, end_pc, trf_est):
        """
        Plot the fitting result as two pointclouds.
        """
        import open3d as o3d
        pre_q = np.linalg.norm(start_pc[:, :4] - end_pc[:, :4], axis=1)
        start_m = (trf_est @ start_pc[:, 0:4].T).T
        fit_qe = np.linalg.norm(start_m[:, :3] - end_pc[:, :3], axis=1)

        # Compute flow quality via positions
        print(pre_q.mean(), "->", fit_qe.mean())

        colors = start_pc[:, 4:7] / 255.  # recorded image colors
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
        pcd2.colors = o3d.utility.Vector3dVector(end_pc[:, 4:7] / 255.)

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
        """
        Convert a trajectory motion into an action. Try to use
        absolute-full control.
        """
        if name == "grip":  # close gripper, don't move
            servo_control = env.robot.get_control("absolute-full", min_iter=24)
            cur_pos, cur_orn = env.robot.get_tcp_pos_orn()
            servo_action = [*cur_pos, *cur_orn, val]

        elif name == "abs":  # move to abs pos
            servo_control = env.robot.get_control("absolute-full")
            goal_pos, goal_orn = val
            servo_action = [*goal_pos, *goal_orn, prev_servo_action[-1]]

        elif name == "rel":
            servo_control = env.robot.get_control("absolute-full")
            t_rel = pos_orn_to_matrix(*val)
            new_pos, new_orn = matrix_to_pos_orn(t_rel @ env.robot.get_tcp_pose())
            servo_action = [*new_pos, *new_orn, prev_servo_action[-1]]
        else:
            raise ValueError

        return servo_action, servo_control

    def abs_to_world_tcp(self, servo_info, live_info):
        """
        The the goal tcp position: T_tcp_wold.
        """
        t_camlive_camdemo = np.linalg.inv(servo_info["align_trf"])
        cam_base_est = live_info["world_tcp"] @ self.T_tcp_cam @ t_camlive_camdemo
        tcp_base_est = cam_base_est @ np.linalg.inv(self.T_tcp_cam)
        return tcp_base_est

    def abs_to_action(self, servo_info, live_info, env=None):
        """
        Arguments:
            servo_info: dict with keys align_trf, grip_action
            live_info: dict with keys world_tcp, shape ?
            env: env handle, only needed for direct movements
        """
        t_world_tcp = self.abs_to_world_tcp(servo_info, live_info)
        goal = t_world_tcp
        goal_pos = goal[:3, 3]
        goal_angles = R.from_matrix(goal[:3, :3]).as_euler("xyz")
        goal_quat = R.from_matrix(goal[:3, :3]).as_quat()

        direct = True
        if direct and not self.is_live_sim:
            env.robot.move_async_cart_pos_abs_lin(goal_pos, goal_quat)
            servo_action, servo_control = None, None
        else:
            grip_action = servo_info["grip_action"]
            servo_action = goal_pos.tolist() + [goal_angles[2], grip_action]
            servo_control = env.robot.get_control("absolute")

        return servo_action, servo_control
