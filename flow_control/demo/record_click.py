"""Script to record demos by click on camera images."""
import math
import platform
import pickle

import hydra
import pygame
import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_io.robot_interface.base_robot_interface import GripperInterface as GI
from gym_grasping.envs.robot_sim_env import RobotSimEnv


def get_point_in_world_frame(cam, t_tcp_cam, t_world_tcp, depth, clicked_point):
    """convert a clicked point to world coordinates"""
    point_cam_frame = cam.deproject(clicked_point, depth, homogeneous=True)
    if point_cam_frame is None:
        print("No depth measurement at clicked point")
        return None
    point_world_frame = t_world_tcp @ t_tcp_cam @ point_cam_frame
    return point_world_frame[:3]


class ClickToPos:
    def __init__(self, cam, t_tcp_cam, initial_pose=None):
        """policy that acts based on mouse clicks inputs using pygame"""
        self.lock = False  # block updating with new images
        self.done = False

        self.rgb = None
        self.depth = None
        self.pos = None
        self.orn = None
        self.t_world_tcp = None

        self.cam = cam
        self.t_tcp_cam = t_tcp_cam

        self.io_log = []
        self.last_action = initial_pose
        self.gripper_state = GI.to_gripper_state("open")

        self.mode_index = None
        self.mode_functions = [self.change_z, self.change_r]
        self.mode_names = ["change_z", "change_r"]
        self.change_mode(0)

        pygame.init()
        self.screen = pygame.display.set_mode((cam.width, cam.height))
        self.screen.fill((255, 255, 255))

    def change_mode(self, new_index=None):
        """change between control modes"""
        if new_index is None:
            new_index = (self.mode_index + 1) % len(self.mode_functions)
            print("Setting mode", self.mode_names[new_index])
        self.mode_index = new_index
        self.mode_name = self.mode_names[self.mode_index]
        self.mode_function = self.mode_functions[self.mode_index]

    def step(self, rgb, depth, pos, orn, t_world_tcp):
        """step the waypoint factory"""
        # update state info (live mode)
        if self.lock:
            return

        surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        self.screen.blit(surf, (0, 0))
        pygame.display.update()

        # observe new frame
        self.rgb = rgb
        self.depth = depth
        self.pos = pos
        self.orn = orn
        self.t_world_tcp = t_world_tcp

        return self.done

    def _handle_events(self):
        """handle click events. """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.done = True
                    break

                elif event.key == pygame.K_SPACE:
                    self.change_gripper()

                else:
                    print("keydown", event.key)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.change_xy(*pygame.mouse.get_pos()[::-1])

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                self.change_gripper()

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self.change_mode()

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 4:
                self.mode_function(up=True)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 5:
                self.mode_function(up=False)

            else:
                #print("unknown", event)
                pass


    def change_gripper(self):
        """change the gripper state"""
        self.gripper_state = GI.toggle(self.gripper_state)

    def change_xy(self, x_pos, y_pos):
        """change the xy position"""
        depth_m = self.depth[x_pos, y_pos]
        point_world = get_point_in_world_frame(self.cam, self.t_tcp_cam, self.t_world_tcp, depth_m, (y_pos, x_pos))

        new_action = [list(x) for x in self.last_action]
        new_action[0][0:2] = point_world[0:2] # change xy
        self.last_action = new_action

    def change_z(self, up=True):
        """change the z position"""
        new_action = [list(x) for x in self.last_action]
        if up:
            new_action[0][2] -= .005
        else:
            new_action[0][2] += .005
        self.last_action = tuple(new_action)

    def change_r(self, up=True):
        """change the rotation"""
        new_action = [list(x) for x in self.last_action]
        old_rot = R.from_quat(self.last_action[1])
        if up:
            rot_change = R.from_euler("xyz", (0,0,5), degrees=True)
        else:
            rot_change = R.from_euler("xyz", (0,0,-5), degrees=True)
        new_action[1] = (rot_change * old_rot).as_quat()
        self.last_action = new_action

    def get_action(self):
        """get actions"""
        self._handle_events()
        return *self.last_action, self.gripper_state.value

    def save_io(self, rec_fn):
        """
        Takes filename ending in .npz -> _wfio.pkl

        Arguments:
            fn: str ending in .npz
        """
        io_fn = rec_fn.replace(".npz", "_wfio.pkl")
        with open(io_fn, "wb") as f_obj:
            pickle.dump(self.io_log, f_obj)


def run_live_continous(cfg, env, initial_pose="neutral", clicked_points=None):
    """
    This is a two-step process, first click to anchor the waypoints.
    Then execute the generated trajectory.
    """
    robot = env.robot
    cam = env.camera_manager.gripper_cam
    assert cam.resize_resolution == [640, 480]  # don't crop like default panda teleop

    # Step 1: move to neutral
    if initial_pose == "neutral":
        robot.move_to_neutral()
        raise ValueError

    env.robot.move_cart_pos_abs_lin(*initial_pose)

    # Step 2: set up recorder
    recorder = hydra.utils.instantiate(cfg.recorder, env=env)
    recorder.recording = True
    action = None
    obs, reward, done, e_info = env.step(action)
    info = {**e_info, "wp_name": "start"}
    recorder.step(None, action, obs, reward, done, info)

    # Step 3: run waypoint recorder
    t_tcp_cam = cam.get_extrinsic_calibration()
    wp_factory = ClickToPos(cam, t_tcp_cam, initial_pose)

    while 1:
        rgb, depth = cam.get_image()
        pos, orn = robot.get_tcp_pos_orn()
        t_world_tcp = robot.get_tcp_pose()

        input_done = wp_factory.step(rgb, depth, pos, orn, t_world_tcp)
        if input_done:
            print("input done")
            break

        action = wp_factory.get_action()
        if action is not None:
            print(action)
            robot.move_cart_pos_abs_lin(action[0], action[1])
            print(action[2])
            if action[2] == -1:
                robot.close_gripper(blocking=True)
            elif action[2] == 1:
                robot.open_gripper(blocking=True)
        obs, reward, done, e_info = env.step(None)
        if done:
            print("env done")
            break

    recorder.save()
    print("done executing wps!")
    return reward


def test_pick_n_place(cfg):
    new_pos = (-0.10, -0.60, 0.18)
    new_orn = tuple(R.from_euler("xyz", (math.pi, 0, math.pi/2)).as_quat())

    env = RobotSimEnv(task="pick_n_place", robot="kuka", renderer="debug", control="absolute",
                      img_size=(640, 480))

    reward = run_live_continous(cfg, env, initial_pose=(new_pos, new_orn))

    assert reward == 1.0


@hydra.main(config_path="../../../robot_io/conf/", config_name="ur3_teleop.yaml")
def main(cfg=None):
    # node = platform.uname().node
    # if node in ('knoppers',):
    #     robot = hydra.utils.instantiate(cfg.robot)
    #     env = hydra.utils.instantiate(cfg.env, robot=robot)
    test_pick_n_place(cfg)


if __name__ == "__main__":
    main()
