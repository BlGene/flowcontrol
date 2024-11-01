"""
Module for testing condtional servoing.
"""
import logging

import numpy as np
import hydra.utils

from flow_control.servoing.module import ServoingModule
from flow_control.servoing.runner import evaluate_control


def test_cam(cam):
    """
    have a look at the camera images to see if they are working
    """
    import cv2
    rgb, _ = cam.get_image()
    print(cam.get_intrinsics())
    while 1:
        rgb, _ = cam.get_image()
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)


def test_robot(robot):
    """
    move the robot around a bit to see if it's working.
    """
    robot.move_to_neutral()
    pos, orn = robot.get_tcp_pos_orn()

    for dy in (.02, -.02):
        new_pos = pos.copy()
        new_pos[1] += dy
        robot.move_cart_pos_abs_ptp(new_pos, orn)

    robot.move_to_neutral()
    print("done!")


def select_demo(control_config, tasks, live_rgb):
    '''
    Selects the demonstration with the minimum reprojection error

    Args:
        tasks: List of strings with the path of the tasks
        live_rgb: Array with the live view

    Returns:
        task: String with the path of the selected demonstration

    '''
    servo_modules = [
        (t, ServoingModule(t, control_config=control_config, plot=False, save_dir=None)) for t in tasks
    ]

    best_task = None
    best_error = np.inf
    for t, s in servo_modules:
        flow = s.flow_module.step(s.demo.rgb, live_rgb)
        warped = s.flow_module.warp_image(live_rgb / 255.0, flow)
        error = ((warped - s.demo.rgb / 255.0) ** 2.0).sum(axis=2) * s.demo.mask
        error = error.sum() / s.demo.mask.sum()
        if error < best_error:
            best_error = error
            best_task = t

    return best_task


@hydra.main(config_path="/home/argusm/lang/robot_io/robot_io/conf", config_name="panda_teleop.yaml")
def main(cfg):
    """
    Try running conditional servoing.
    """
    logging.basicConfig(level=logging.DEBUG, format="")
    # recording = "/home/argusm/kuka_recordings/flow/simple_motions"
    # recording = "/home/argusm/kuka_recordings/flow/shape_sorting"
    # recording = '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2'

    tasks = [
        '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2',
        '/home/argusm/kuka_recordings/flow/ssh_demo/orange_trapeze_2',
        '/home/argusm/kuka_recordings/flow/ssh_demo/green_oval_2',
    ]

    tasks = ['/home/argusm/kuka_recordings/flow/sick_vacuum/17-19-19', ]

    control_config = dict(mode="pointcloud-abs", threshold=0.25)

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    for _ in range(1):
        move_to_first_frame = True
        if move_to_first_frame:
            # cur_pos, cur_orn = robot.get_tcp_pos_orn()
            # world_tcp = servo_module.demo.world_tcp
            # goal_pos = world_tcp[:3, 3]
            # goal_quat = R.from_matrix(world_tcp[:3, :3]).as_quat()

            # input("Enter to move.")
            goal_pos = np.array((0.56, 0.0, 0.24))
            cur_orn = np.array((0.99964865, 0.01395868, -0.02089317, -0.00843854))
            env.robot.move_cart_pos_abs_lin(goal_pos, cur_orn)

        else:
            robot.move_to_neutral()

        state, _, _, _ = env.step(None)
        live_rgb = state['rgb_gripper']

        if len(tasks) > 1:
            task = select_demo(control_config, tasks, live_rgb)
            tasks.remove(task)
        else:
            task = tasks[0]

        servo_module = ServoingModule(task, control_config=control_config, plot=True, save_dir=f'{task}/plots')

        # TODO(max): gripper cam not closing properly,maybe this helps
        try:
            state, _, _, _ = evaluate_control(env, servo_module, start_paused=True)
        except KeyboardInterrupt:
            del env.camera_manager.gripper_cam


if __name__ == "__main__":
    main()

    # cam = hydra.utils.instantiate(cfg.cams.gripper_cam)
    # test_cam(cam)

    # robot = hydra.utils.instantiate(cfg.robot)
    # test_robot(robot)
