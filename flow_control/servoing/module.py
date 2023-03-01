"""
This is a stateful module that contains a recording, then
given a  query RGB-D image it outputs the estimated relative
pose. This module also handles incrementing along the recording.
"""
import copy
import logging
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R

from flow_control.servoing.playback_env_servo import PlaybackEnvServo
from flow_control.servoing.fitting import solve_transform
from flow_control.servoing.fitting_ransac import Ransac
from flow_control.rgbd_camera import RGBDCamera
from flow_control.utils_coords import pos_orn_to_matrix, matrix_to_pos_orn, rec_pprint

try:
    from flow_control.servoing.live_plot import ViewPlots
except ImportError:
    # Don't call logging here because it overwrites log level.
    ViewPlots = None

try:
    from gym_grasping.envs.robot_sim_env import RobotSimEnv
except ImportError:
    RobotSimEnv = type(None)


# A logger for this file
log = logging.getLogger(__name__)

# magical gain values for dof, these could come from calibration
DEFAULT_CONF = dict(mode="pointcloud-abs",
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

    def __init__(self, recording, control_config=None, start_paused=False,
                 load='keep', plot=False, plot_save_dir=False, flow_module='UniMatch'):
        """
        Arguments:
            recording: path to saved recording
            control_config: parameters for servoing
            start_paused: this computes actions and losses, but returns None
                          actions
            load: demonstration keyframes that should be loaded (all/keep/list)
            plot: show plot of servoing statistics
            plot_save_dir: directory in which to save plot
        """
        log.info("Loading ServoingModule...")

        if flow_module == 'UniMatch':
            log.info("Using UniMatch")
            from flow_control.flow.module_unimatch import FlowModule
        elif flow_module == 'RAFT':
            log.info("Using RAFT")
            from flow_control.flow.module_raft import FlowModule

        # we could also use flownet2, IRR, or FGR

        if isinstance(recording, PlaybackEnvServo):
            self.demo = recording
        else:
            log.info("Loading recording (make take a bit): %s", recording)
            start = time.time()
            self.demo = PlaybackEnvServo(recording, load=load)
            end = time.time()
            log.info("Loading time was %s s" % round(end - start, 3))
        self.demo_cam = RGBDCamera(self.demo.cam)
        assert isinstance(self.demo_cam.calibration, dict)

        self.live_cam = None

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
            log.warning("Servoing Plot: ignoring plot=True, as import failed")
        elif plot:
            self.view_plots = ViewPlots(threshold=self.config.threshold,
                                        save_dir=plot_save_dir)
        if start_paused:
            if self.view_plots is False:
                logging.warning("Servoing Module: switching start_paused -> False as plots not active")
                start_paused = False
            log.info("Starting paused.")
        elif self.view_plots:
            self.view_plots.started = True

        self.paused = start_paused

        # vars set in reset
        self.counter = None
        self.counter_frame = None
        self.action_queue = None
        self.reset()

    def check_calibration(self, env):
        """
        This checks to see that the env matches the demonstration.
        """
        assert env is not None

        # workaround for testing servoing between demonstration frames.
        if env == "demo":
            name = self.demo.env_info["name"]
            self.T_tcp_cam = self.demo.env_info["camera"]["T_tcp_cam"]
            return

        live_cam = env.camera_manager.gripper_cam
        live_calib = live_cam.get_intrinsics()
        demo_calib = self.demo_cam.calibration

        # check intrinsic calibration
        for key in ['width', 'height', 'fx', 'fy', 'cx', 'cy']:
            if demo_calib[key] != live_calib[key]:
                log.warning(f"Calibration: %s demo!=live %s != %s", key, demo_calib[key], live_calib[key])

        # TODO(max): T_tcp_cam should be live version.
        demo_t_tcp_cam = self.demo_cam.T_tcp_cam
        self.T_tcp_cam = demo_t_tcp_cam

        # check extrinsic calibration
        # TODO(lukas): live_cam.get_extrinsic_calibration() should work w/o robot name.
        # live_t_tcp_cam = live_cam.get_extrinsic_calibration(robot_name=env.robot.name)
        #extr_diff = np.linalg.norm(live_t_tcp_cam - demo_t_tcp_cam)
        #if extr_diff > .01:
        #    log.warning(f"Extrinsic calibration diff: %s, should be <.01", extr_diff)

    def reset(self):
        """
        reset servoing, reset counter and index
        """
        self.counter = 0
        self.counter_frame = 0
        self.action_queue = []
        self.demo.reset()
        if self.view_plots:
            self.view_plots.reset()

    def pause(self):
        if self.view_plots:
            log.info("Servoing: paused, click to resume.")
            self.view_plots.started = False

    def step(self, live_state, live_info):
        """
        Main loop, this does sequence alignment.

        Usually what frame alignment gives, but sometimes something else.

        Arguments:
            live_state: dict obs with keys: rgb_gripper, depth_gripper, robot_state
            live_info: dict with keys tcp_pose, depth

        Returns:
            action: {motion, ref}
            done: binary if demo sequence is completed
            info: dict with keys:
                "align_trf"
                "grip_action"
        """
        # this puts world_tcp in live_info as well
        live_rgb, live_depth, live_tcp = self.process_obs(live_state, live_info)

        info = {}
        # find the alignment between frames
        align_transform, align_q = self.frame_align(live_rgb, live_depth, info)

        # from the alignment find the actions
        rel_action, loss = self.trf_to_rel_act_loss(align_transform, live_tcp)
        action = self.trf_to_abs_act(align_transform, live_info)

        # find the threshold values
        threshold, force_step = self.get_threshold_or_skip()

        self.plot_live(loss, threshold, align_q, live_rgb, live_tcp, rel_action)
        self.log_step(rel_action, loss, threshold, force_step)

        info["loss"] = loss
        info["threshold"] = threshold
        info["demo_index"] = self.demo.index
        info["align_trf"] = align_transform
        info["grip_action"] = self.demo.get_action("gripper")

        if self.paused:
            return None, False, info

        done = False
        if loss < threshold or force_step:
            demo_max_frame = self.demo.get_max_frame()

            if self.demo.index < demo_max_frame:
                self.demo.step()
                self.counter_frame = 0

                info["traj_acts"] = self.get_trajectory_actions(info)
                # debug output
                step_str = f"{self.demo.index} / {demo_max_frame} start"
                #step_str += f" steps {self.counter} "
                log.info(step_str)

            elif self.demo.index == demo_max_frame:
                done = True

        self.counter += 1
        self.counter_frame += 1

        pause_on_step = False
        if self.view_plots and pause_on_step:
            self.view_plots.started = False

        return action, done, info

    @staticmethod
    def process_obs(live_state, live_info):
        """
        Returns:
            live_rgb: live rgb image
            live_tcp: live tcp position, shape (6, )
            live_depth: live depth image
        """
        live_rgb = live_state["rgb_gripper"]
        live_depth = live_state["depth_gripper"]
        live_tcp = live_state["robot_state"]["tcp_pos"]
        world_tcp = pos_orn_to_matrix(live_state["robot_state"]["tcp_pos"],
                                      live_state["robot_state"]["tcp_orn"])
        live_info["world_tcp"] = world_tcp
        assert np.asarray(live_tcp).ndim == 1
        return live_rgb, live_depth, live_tcp

    def frame_align(self, live_rgb, live_depth, info=None):
        """
        Get a transformation from two point clouds and a demonstration mask.

        Arguments:
            live_rgb: image
            live_depth: image
            info: dict, updated in place with fitting stats
        Returns:
            T_in_tcp: 4x4 homogeneous transformation matrix
            fit_q: scalar fit quality, lower is better
        """
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
        start_points = end_points + masked_flow[:, ::-1].round().astype('int')
        start_pc = self.demo_cam.generate_pointcloud(live_rgb, live_depth, start_points)
        end_pc = self.demo_cam.generate_pointcloud(demo_rgb, demo_depth, end_points)
        mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)

        # subsample fitting, maybe evaluate with ransac
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]

        pc_size_t = 32
        pc_size = len(start_pc)
        assert pc_size == len(end_pc)
        if pc_size < pc_size_t:
            log.warning("Skipping fitting, too few points %s < %s", pc_size, pc_size_t)
            trf_est, fit_qc = np.eye(4), 999
            return trf_est, fit_qc

        # 3. estimate trf and transform to TCP coordinates
        # estimate T, put in non-homogenous points, get homogeneous trf.
        # trf_est = solve_transform(start_pc, end_pc)
        def eval_fit(trf_estm, start_ptc, end_ptc):
            start_m = (trf_estm @ start_ptc[:, 0:4].T).T
            fit_qe = np.linalg.norm(start_m[:, :3] - end_ptc[:, :3], axis=1)
            return fit_qe

        ransac = Ransac(start_pc, end_pc, solve_transform, eval_fit, .002, 5)
        fit_q_pos, trf_est = ransac.run()

        # Compute fit quality via color
        fit_q_col = np.linalg.norm(start_pc[:, 4:7] - end_pc[:, 4:7], axis=1).mean()

        # if self.counter > 30:
        #   self.debug_show_fit(start_pc, end_pc, trf_est)

        if info is not None:
            info["fit_pc_size"] = pc_size
            info["fit_inliers"] = len(fit_q_pos)
            info["fit_q_pos"] = fit_q_pos.mean()
            info["fit_q_col"] = fit_q_col

        return trf_est, fit_q_col

    def trf_to_rel_act_loss(self, align_transform, live_tcp):
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

        # log.info(f"loss_xy {loss_xy:.4f}, loss_rot {loss_rot:.4f}, loss_z {loss_z:.4f}, rot_z {rot_z:.4f}")

        rot_projected_z = R.from_euler("xyz", (0, 0, rot_z)).as_quat()
        rel_action = dict(motion=((*move_xy, move_z), rot_projected_z, move_g), ref="rel")
        return rel_action, loss

    def trf_to_abs_act(self, align_transform, live_info):
        """
        Get an action from a relative transformation of the foreground object.
        """
        if self.config.mode == "pointcloud":
            raise NotImplementedError
        elif self.config.mode == "pointcloud-abs":
            t_world_tcp = self.abs_to_world_tcp(align_transform, live_info["world_tcp"])
            goal_pos, goal_quat = matrix_to_pos_orn(t_world_tcp)
            grip_action = self.demo.get_action("gripper")
            action = dict(motion=(goal_pos, goal_quat, grip_action), ref="abs")

        elif self.config.mode == "pointcloud-abs-rotz":
            t_world_tcp = self.abs_to_world_tcp(align_transform, live_info["world_tcp"])
            goal_pos, goal_quat = matrix_to_pos_orn(t_world_tcp)
            grip_action = self.demo.get_action("gripper")
            # project the rotation to keep only the z component, try if servoing is unstable
            goal_pos, goal_quat = self.project_rot_z(goal_pos, goal_quat, t_world_tcp)
            action = dict(motion=(goal_pos, goal_quat, grip_action), ref="abs")
        else:
            raise ValueError

        assert isinstance(action, dict)
        return action

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

    def get_threshold_or_skip(self):
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

        # scale threshold when there is no convergence so that we can progress
        # often we are already close enough for this to work.
        scale_threshold = True
        if scale_threshold:
            over = max(self.counter_frame - 10, 0)
            threshold *= (1+0.05)**over

        return threshold, force_step

    def get_trajectory_actions(self, info):
        """
        Returns:
            pre_actions: list of [{motion,ref}, ...]
        """
        try:
            pre_actions = self.demo.get_keep_dict()["pre"]
        except KeyError:
            return info

        if isinstance(pre_actions, dict):
            pre_actions = list(pre_actions.items())
        return pre_actions

    def plot_live(self, loss, threshold, align_q, live_rgb, live_tcp, rel_action):
        if self.view_plots:
            frame = self.demo.index
            demo_rgb = self.demo.cam.get_image()[0]
            demo_mask = self.demo.get_fg_mask()
            series_data = (loss, frame, threshold, align_q, live_tcp[0])
            self.view_plots.step(series_data, live_rgb, demo_rgb,
                                 self.cache_flow, demo_mask, rel_action)
            self.paused = not self.view_plots.started

    def log_step(self, rel_action, loss, threshold, force_step):
        # debug output
        loss_str = f"{self.counter:04d} loss {loss:4.4f}"
        action_str = " action: " + rec_pprint(rel_action["motion"])
        action_str += " " + "-".join([list(x.keys())[0] for x in self.action_queue])
        log.debug(loss_str + action_str)

        log.info(f"{self.demo.index} / {self.demo.get_max_frame()} loss: {loss:.4f}")
        # {'step' if loss < threshold or force_step else ''}")

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

    def abs_to_world_tcp(self, align_trf, live_world_tcp):
        """
        The goal tcp position: T_tcp_wold.
        """
        t_camlive_camdemo = np.linalg.inv(align_trf)
        cam_base_est = live_world_tcp @ self.T_tcp_cam @ t_camlive_camdemo
        tcp_base_est = cam_base_est @ np.linalg.inv(self.T_tcp_cam)
        return tcp_base_est