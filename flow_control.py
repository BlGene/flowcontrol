"""
Testing file for development, to experiment with evironments.
"""
import json
from math import pi

from gym_grasping.envs.grasping_env import GraspingEnv
from gym_grasping.flow_control.servoing_module import ServoingModule

from pdb import set_trace


def evaluate_control(recording, perturbation, env=None, task_name="stack",
                     threshold=0.4, max_steps=1000, mouse=False, plot=True):
    # Input:
    #   perturbation goes from -1,1

    # internal parameters
    perturb_actions = False  # perturb gripper position or perturb object pose
    to_shade = 0.6 # displace objects into shade

    # load the flow module
    servo_module = ServoingModule(recording, plot=plot)

    # load env (needs
    if env is None:
        object_pose = [-0.01428776+to_shade, -0.52183914,  0.15, pi]
        object_pose[0] += perturbation[0]*0
        object_pose[3] += perturbation[1]*pi/2
        print("perturbation", perturbation)
        #object_pose = (.071+perturbation[0]*0.02, -.486+perturbation[1]*0.02, 0.15, 0)
        env = GraspingEnv(task=task_name, renderer='tiny', act_type='continuous',
                          max_steps=1e9, object_pose=object_pose,
                          img_size=servo_module.size)

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
            action = [0,0,1,0,0]
        elif counter > 50:
            action = servo_action
        elif counter == 0:
            # inital frame dosent have action
            pass
        else:
            pass

        # Environment Stepping
        state, reward, done, info = env.step(action)
        if done:
            print("done. ", reward)
            break

        ee_pos = list(env._p.getLinkState(env.robot.robot_uid, env.robot.flange_index)[0])
        ee_pos[2] += 0.02
        servo_action = servo_module.step(state, ee_pos)


    if 'ep_length' not in info:
        info['ep_length'] = counter
    return state, reward, done, info

if __name__ == "__main__":
    import itertools

    task_name = "stack"
    recording = "stack_recordings/episode_118"
    threshold = 0.35 # .40 for not fitting_control

    #task_name = "block"
    #recording = "block_recordings/episode_41"
    #threshold = 0.35 # .40 for not fitting_control

    #task_name = "block"
    #recording = "block_yellow_recordings/episode_1"
    #threshold = 1.8 # .40 for not fitting_control

    samples = sorted(list(itertools.product([-1, 1, -.5, .5, 0], repeat=2)))[:7]

    if len(samples) > 10:  # statistics mode
        save = True
        plot = False
    else:  # dev mode
        save = False
        plot = True
        plot_cv = False

    num_samples = len(samples)
    results = []
    for i, s in enumerate(samples):
        print("starting",i,"/",num_samples)
        state, reward, done, info = evaluate_control(recording,
                                                     list(s),
                                                     task_name=task_name,
                                                     threshold=threshold,
                                                     plot=plot)
        res = dict(offset=s,
                   angle=0,
                   threshold=threshold,
                   reward=reward,
                   ep_length=info['ep_length'])

        results.append(res)
        if save:
            with open('./translation_backward.json',"w") as fo:
                json.dump(results, fo)

        break
