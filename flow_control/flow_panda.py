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
import torch

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


def warp(x, flow):
    """
    Warp an image/tensor (im2) back to im1, according to the optical flow

    Args:
        x: [H, W, C] (im2)
        flow: [H, W, 2] flow

    Returns:
        warped: [H, W, 2]
    """
    x = torch.from_numpy(x)[None].float().permute(0, 3, 1, 2).cuda()
    flow = torch.from_numpy(flow)[None].float().permute(0, 3, 1, 2).cuda()
    
    B, _, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().cuda()

    vgrid = torch.autograd.Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W-1, 1)-1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H-1, 1)-1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return (output*mask)[0].permute(1, 2, 0).cpu().numpy()

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
        (t, ServoingModule(
            t,
            episode_num=0,
            control_config=control_config,
            plot=False, save_dir=None
        )) for t in tasks
    ]

    best_task = None
    best_error = np.inf
    for t, s in servo_modules:
        flow = s.flow_module.step(s.demo.rgb, live_rgb)
        warped = warp(live_rgb / 255.0, flow)
        error = ((warped - s.demo.rgb / 255.0) ** 2.0).sum(axis=2) * s.demo.mask
        error = error.sum() / s.demo.mask.sum()
        if error < best_error:
            best_error = error
            best_task = t

    return best_task
 

@hydra.main(config_path="/home/argusm/lang/robot_io/robot_io/conf", config_name="panda_teleop.yaml")
def main(cfg):
    logging.basicConfig(level=logging.DEBUG, format="")
    recording = "/home/argusm/kuka_recordings/flow/simple_motions"
    recording = "/home/argusm/kuka_recordings/flow/shape_sorting"
    plot_dir = '/home/argusm/kuka_recordings/flow/shape_sorting/plots'

    recording = '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2'
    plot_dir = '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2/plots'


    tasks = [
        '/home/argusm/kuka_recordings/flow/ssh_demo/yellow_half_2',
        '/home/argusm/kuka_recordings/flow/ssh_demo/orange_trapeze_2',
        '/home/argusm/kuka_recordings/flow/ssh_demo/green_oval_2',
    ]

    control_config = dict(mode="pointcloud-abs", threshold=0.25)

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)


    for attemp in range(1):

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


        state, _, _, _ = env.step(None)
        live_rgb = state['rgb_gripper']

        task = select_demo(control_config, tasks, live_rgb)
        tasks.remove(task)
        
        servo_module = ServoingModule(task,
                                      episode_num=0,
                                      control_config=control_config,
                                      plot=True, save_dir=f'{task}/plots', start_index=0)

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

