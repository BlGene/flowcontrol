import time
import numpy as np
import hydra.utils
from pdb import set_trace
from flow_control.servoing.module import ServoingModule
from flow_control.flow_control_main import evaluate_control
from scipy.spatial.transform import Rotation as R
import logging
import os
import shutil

def test_cam(cam):
    import cv2
    rgb, depth = cam.get_image()
    print(cam.get_intrinsics())
    while 1:
        rgb, depth = cam.get_image()
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)
 
def test_robot(robot):
    robot.move_to_neutral()
    pos, orn = robot.get_tcp_pos_orn()

    for dy in (.02, -.02):
        new_pos = pos.copy()
        new_pos[1] += dy
        robot.move_cart_pos_abs_ptp(new_pos, orn)
        #robot.visualize_joint_states()
    robot.move_to_neutral()
    print("done!")
 

@hydra.main(config_path="/home/argusm/lang/robot_io/robot_io/conf", config_name="panda_teleop.yaml")
def main(cfg):
    logging.basicConfig(level=logging.DEBUG, format="")
    recording = "/home/argusm/kuka_recordings/flow/simple_motions"
    recording = "/home/argusm/kuka_recordings/flow/shape_sorting"
    plot_dir = '/home/argusm/kuka_recordings/flow/shape_sorting/plots'

    recording = '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2'
    plot_dir = '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2/plots'

    #plot_dir = None
    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(recording,
                                  episode_num=0,
                                  control_config=control_config,
                                  plot=True, save_dir=plot_dir)

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    move_to_first_frame = True
    if move_to_first_frame:
        #cur_pos, cur_orn = robot.get_tcp_pos_orn()
        #world_tcp = servo_module.demo.world_tcp
        #goal_pos = world_tcp[:3, 3]
        #goal_quat = R.from_matrix(world_tcp[:3, :3]).as_quat()

        #input("Enter to move.")
        goal_pos = np.array((0.56, 0.0, 0.24))
        cur_orn = np.array((0.99964865, 0.01395868, -0.02089317, -0.00843854))
        env.robot.move_cart_pos_abs_lin(goal_pos, cur_orn)

    else:
        robot.move_to_neutral()

    # TODO(max): gripper cam not closing properly,maybe this helps
    try:
        state, reward, done, info = evaluate_control(env, servo_module, start_paused=True)
    except KeyboardInterrupt:
        del env.camera_manager.gripper_cam


if __name__ == "__main__":
    main()

    # cam = hydra.utils.instantiate(cfg.cams.gripper_cam)
    # test_cam(cam)

    # robot = hydra.utils.instantiate(cfg.robot)
    # test_robot(robot)

