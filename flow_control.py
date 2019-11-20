"""
Testing file for development, to experiment with evironments.
"""
import json
from math import pi

from gym_grasping.envs.grasping_env import GraspingEnv
from gym_grasping.flow_control.servoing_module import ServoingModule
from pdb import set_trace


def evaluate_control(env, recording, max_steps=600, mouse=False):
    if mouse:
        from gym_grasping.robot_io.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')

    done = False
    for counter in range(max_steps):
        # Compute controls (reverse order)
        action = [0, 0, 0, 0, 1]
        if mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        elif servo_module.base_frame == servo_module.max_demo_frame:
            # for end move up if episode is done
            action = [0, 0, 1, 0, 0]
        elif counter > 0:
            action = servo_action
        elif counter == 0:
            # inital frame dosent have servo action
            pass
        else:
            pass

        # Environment Stepping
        state, reward, done, info = env.step(action)

        #state extraction
        link_state = env._p.getLinkState(env.robot.robot_uid,
                                         env.robot.flange_index)
        ee_pos = list(link_state[0])
        ee_pos[2] += 0.02
        servo_action = servo_module.step(state, ee_pos)

        # logging
        state_dict = dict(state=state,
                          reward=reward,
                          done=done,
                          ee_pos=ee_pos)

        if done:
            print("done. ", reward, counter)
            break

    if 'ep_length' not in info:
        info['ep_length'] = counter

    return None


from collections import defaultdict
import numpy as np

def save_imitation_trajectory(save_id, collect):
    assert(isinstance(collect[0],dict))

    episode = defaultdict(list)

    for key in collect[0]:
        for step in collect:
            episode[key].append(step[key])

        episode[key] = np.array(episode[key])

    save_fn = f"./eval_t30/run_{save_id:03}.npz"
    np.savez(save_fn, **episode)


if __name__ == "__main__":
    import itertools

    task_name = "stack"
    recording = "stack_recordings/episode_118"
    episode_num = 1
    base_index = 20
    threshold = .30  # .40 for not fitting_control

    #task_name = "block"
    #recording = "block_recordings/episode_41"
    #threshold = 0.35 # .40 for not fitting_control

    #task_name = "block"
    #recording = "block_yellow_recordings/episode_1"
    #threshold = 1.8 # .40 for not fitting_control

    #samples = sorted(list(itertools.product([-1, 1, -.5, .5, 0], repeat=2)))[:7]
    # load env (needs

    img_size =  (256, 256)
    env = GraspingEnv(task=task_name, renderer='tiny', act_type='continuous',
                      max_steps=600, img_size=img_size)

    servo_module = ServoingModule(recording, episode_num=episode_num,
                                  base_index=base_index,
                                  threshold=threshold, plot=True)

    num_samples = 10
    for i, s in enumerate(range(num_samples)):
        print("starting", i, "/", num_samples)
        collect = evaluate_control(env, recording, max_steps=600)

        save_imitation_trajectory(i, collect)

        env.reset()
        servo_module.reset()

