"""
Records demo episodes from sim or real robot.
"""
import math
import time
import cv2
from robot_io.recorder.playback_recorder import PlaybackRecorder
from gym_grasping.envs.iiwa_env import IIWAEnv

KUKA_DIR = '/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/default'


def get_target_poses(tcp_base):
    delta = 0.04
    for i in (0, 1, 2):
        for j in (1, -1):
            target_pose = list(tcp_base[:3, 3])
            target_pose[i] += j * delta
            yield target_pose


def start_recording(save_dir=KUKA_DIR, max_steps=1e6):
    """
    record from real robot
    """
    max_steps = int(max_steps)
    env = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced',
                   dv=0.01, drot=0.04, joint_vel=0.05,
                   gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True,
                   reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2))

    # print(iiwa.cam.get_extrinsic_calibration("iiwa"))

    rec = PlaybackRecorder(env=env, save_dir=save_dir)
    env.reset()
    tcp_base = env.robot.get_tcp_pose()
    base_orn = env.robot.get_state()["tcp_pose"][3:6]

    for target_pos in get_target_poses(tcp_base):
        env.robot.move_cart_pos_abs_ptp(target_pos, base_orn)
        time.sleep(.2)

        _, _, _, info = env.step((0, 0, 0, 0, 1))

        cv2.imshow('win', info['rgb_unscaled'][:, :, ::-1])
        if cv2.waitKey(1) == ord('s'):
            print("Stopping recording")
            break
    env.reset()


if __name__ == "__main__":
    SAVE_DIR = '/home/argusm/kuka_recordings/flow/tmp'
    start_recording(SAVE_DIR)
