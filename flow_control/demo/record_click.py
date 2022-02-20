import time
import platform

import cv2
import hydra

from robot_io.recorder.simple_recorder import RecEnv
from robot_io.input_devices.keyboard_input import KeyboardInput
from gym_grasping.envs.robot_sim_env import RobotSimEnv

from flow_control.demo.record_operations import Move, ObjectSelection, NestSelection
from flow_control.demo.record_operations import WaypointFactory


def test_shape_sorter(operations):
    env = RecEnv("/home/argusm/CLUSTER/robot_recordings/flow/sick_vacuum/17-19-19/frame_000000.npz")
    # env = RecEnv("/home/argusm/CLUSTER/robot_recordings/flow/ssh_demo/orange_trapeze/frame_000000.npz")
    cam = env.cam
    robot = env.robot

    T_tcp_cam = cam.get_extrinsic_calibration()
    wf = WaypointFactory(cam, T_tcp_cam, operations=operations)

    def callback(event, x, y, flags, param):
        nonlocal wf
        wf.callback(event, x, y, flags, param)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", wf.callback)

    pretend_click_points = True
    while 1:
        # Important: Start with cv call to catch clicks that happened,
        # if we don't have those we would update the frame under the click
        cv2.waitKey(1)
        rgb, depth = cam.get_image()
        pos, orn = robot.get_tcp_pos_orn()
        T_world_tcp = robot.get_tcp_pose()
        wf.step(rgb, depth, pos, orn, T_world_tcp)
        if wf.done:
            break
        time.sleep(1)

        if pretend_click_points:
            clicked_points = [(317, 207),
                              (558, 156), (604, 149)]
            clicked_points = [(193, 135),
                              (425, 330), (544, 330)]

            for c_pts in clicked_points:
                callback(cv2.EVENT_LBUTTONDOWN, *c_pts, None, None)

    print(wf.done_waypoint_names)
    vacuum_wps = [((0.47, 0.08, 0.26), (1, 0, 0, 0), 1), ((0.5621555602802504, 0.12391772919540847, 0.22065518706705803), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 1), ((0.5721555602802504, 0.12391772919540847, 0.172655187067058), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 1), ((0.5721555602802504, 0.12391772919540847, 0.16065518706705803), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.5721555602802504, 0.12391772919540847, 0.30065518706705807), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.39030870922041905, -0.13124625763610404, 0.25444003733583315), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.39030870922041905, -0.13124625763610404, 0.18944003733583314), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.39030870922041905, -0.13124625763610404, 0.13444003733583315), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 1)]
    assert wf.done_waypoints == vacuum_wps

    wf.save_io(env.file)
    print("test passed.")


def run_live(cfg, env, operations, initial_pose="neutral", clicked_points=None):
    robot = env.robot
    cam = env.camera_manager.gripper_cam
    # don't crop like default panda teleop
    assert cam.resize_resolution == [640, 480]

    # Step 1: move to neutral
    if initial_pose == "neutral":
        robot.move_to_neutral()
    else:
        env.robot.move_cart_pos_abs_lin(*initial_pose)

    # Step 2: set up recorder
    recorder = hydra.utils.instantiate(cfg.recorder)
    recorder.recording = True
    action = None
    record_info = {"trigger_release": False, "hold_event": False,
                   "dead_man_switch_triggered": False}
    obs, _, _, e_info = env.step(action)
    info = {**e_info, **record_info, "wp_name": "start"}
    recorder.step((1, 2, 3), obs, info)

    # Step 3: run waypoint recorder
    T_tcp_cam = cam.get_extrinsic_calibration()
    wf = WaypointFactory(cam, T_tcp_cam, operations=operations)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", wf.callback)
    while 1:
        # Important: Start with cv call to catch clicks that happened,
        # if we don't have those we would update the frame under the click
        cv2.waitKey(1)
        rgb, depth = cam.get_image()
        pos, orn = robot.get_tcp_pos_orn()
        T_world_tcp = robot.get_tcp_pose()
        wf.step(rgb, depth, pos, orn, T_world_tcp)
        if wf.done:
            break
        time.sleep(1)

        if clicked_points:
            for c_pts in clicked_points:
                wf.callback(cv2.EVENT_LBUTTONDOWN, *c_pts, None, None)
    cv2.destroyWindow("image")

    # Step 4: iterate through waypoints
    obs, _, _, e_info = env.step(action)  # record view at neutral position
    info = {**e_info, **record_info, "wp_name": "start2"}
    recorder.step((1, 2, 3), obs, info)

    prev_wp = None
    for i, wp in enumerate(wf.done_waypoints):
        # move to position
        robot.move_cart_pos_abs_lin(wp.pos, wp.orn)
        if prev_wp is not None and prev_wp.grip != wp.grip:
            if wp.grip == 0:
                robot.close_gripper(blocking=True)
            elif wp.grip == 1:
                robot.open_gripper(blocking=True)
            else:
                raise ValueError
        obs, reward, done, e_info = env.step(None)

        # refine position
        print(f"moved to: {wp.name}")
        keyboard = KeyboardInput(initial_grip=env.robot.gripper.desired_action)
        while env is not None and not keyboard.get_done():
            action = keyboard.handle_keyboard_events()
            control = env.robot.get_control("relative")
            env.step(action, control)

        # record state
        action = wp.to_action()
        info = {**e_info, **record_info, "wp_name": wp.name}
        recorder.step(action, obs, record_info)

        prev_wp = wp

    recorder.__exit__()
    print("done executing wps!")
    return reward


# List of operations, these process images to generate waypoints
vacuum_ops = [Move(pos=(0.47, 0.08, 0.26), orn=(1, 0, 0, 0), name="start"),
              ObjectSelection(width=128, preg_offset=(.01, 0, 0.052), grip_offset=(.01, 0, 0.04)),
              NestSelection(inst_offset=(0, 0, 0.075))]

shape_sorting_ops = [ObjectSelection(preg_offset=(0, 0, 0.01), grip_offset=(0, 0, 0.0)),
                     NestSelection(inst_offset=(0, 0, 0.075))]

# This seems better because it's more explicit
# wp1 = move(fixed, tag="start")
# wp2 = move(tag="object-high")
# wp2 = move(tag="object-low")
# wp2 = move(tag="object-grasp")

# wp3 = move(tag="nest-high")
# wp3 = move(tag="nest-low")
# wp3 = move(tag="nest-inst")

# move1
# pic = record_picture()
# object = detect_object(pic)
# nest = detect_nest(pic)


def test_pick_n_place(cfg):
    import math
    from scipy.spatial.transform import Rotation as R
    new_pos = (-0.10, -0.60, 0.18)
    new_orn = tuple(R.from_euler("xyz", (math.pi, 0, math.pi/2)).as_quat())
    clicked_points = ((456, 235), (164, 279), (140, 278))

    env = RobotSimEnv(task="pick_n_place", robot="kuka", renderer="debug", control="absolute",
                      img_size=(640, 480))
    reward = run_live(cfg, env, shape_sorting_ops, initial_pose=(new_pos, new_orn),
                      clicked_points=clicked_points)

    assert reward == 1.0


@hydra.main(config_path="/home/argusm/lang/robot_io/robot_io/conf", config_name="panda_teleop.yaml")
def main(cfg=None):
    node = platform.uname().node
    if node in ('knoppers',):
        robot = hydra.utils.instantiate(cfg.robot)
        env = hydra.utils.instantiate(cfg.env, robot=robot)
        run_live(cfg, env, vacuum_ops)
    else:
        import math
        from scipy.spatial.transform import Rotation as R
        new_pos = (-0.10, -0.60, 0.18)
        new_orn = tuple(R.from_euler("xyz", (math.pi, 0, math.pi/2)).as_quat())
        clicked_points = ((456, 235), (164, 279), (140, 278))

        env = RobotSimEnv(task="pick_n_place", robot="kuka", renderer="debug", control="absolute",
                          img_size=(640, 480))
        run_live(cfg, env, shape_sorting_ops, initial_pose=(new_pos, new_orn),
                 clicked_points=clicked_points)

        # test_pick_n_place(cfg)

        # test_shape_sorter(vacuum_ops)


if __name__ == "__main__":
    main()
