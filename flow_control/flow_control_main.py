"""
Testing file for development, to experiment with environments.
"""
import copy
import time
import logging

from flow_control.utils_coords import get_action_dist, rec_pprint, action_to_current_state
from flow_control.utils_coords import pos_orn_to_matrix, matrix_to_pos_orn


def flatten(xss):
    return *xss[0][0], *xss[0][1], xss[1]


def action_to_abs(env, action):
    if action["ref"] == "rel":
        new_action = copy.copy(action)
        t_rel = pos_orn_to_matrix(*action["motion"][0:2])
        new_pos, new_orn = matrix_to_pos_orn(t_rel @ env.robot.get_tcp_pose())
        new_action["motion"] = (new_pos, new_orn, action["motion"][2])
        new_action["ref"] = "abs"
        return new_action
    return action


def dispatch_action_panda(env, trj_act):
    assert trj_act["ref"] == "abs"
    goal_pos, goal_quat, goal_g = trj_act["motion"]
    env.robot.move_cart_pos_abs_lin(goal_pos, goal_quat)
    if goal_g == 1:
        env.robot.open_gripper()
    elif goal_g == -1:
        env.robot.close_gripper(blocking=True)
        time.sleep(.3)
    else:
        raise ValueError(f"Bad gripper action: {goal_g} must be 1, -1")


def evaluate_control(env, servo_module, max_steps=1000, initial_align=True, use_trajectory=True, done_cooldown=5):
    """
    Function that runs the policy.
    Arguments:
        env: the environment
        servo_module: the servoing module
        max_steps: the number of steps to run servoing for
        initial_align: align with the initial absolute position of the demo
        use_trajectory: use dead-reckoning actions based on the demonstration trajectory
        done_cooldown: episode steps to wait for after servoing is completed (allows simulations to finish).
    """
    assert env is not None
    servo_module.set_env(env)
    safe_move = dict(path="ptp", blocking=True)

    if initial_align:
        initial_act = action_to_current_state(servo_module.demo, grip_action=1)
        initial_act.update(safe_move)
        action_dist_t = 0.05
        for i in range(25):
            _, _, _, _ = env.step(initial_act)
            dist = get_action_dist(env, initial_act)
            if dist < action_dist_t:
                break
        if dist > action_dist_t:
            logging.warning("Bad absolute move, dist = %s, t = %s", dist, action_dist_t)
        # TODO(max): remove this, but test simulation
        servo_action = initial_act
    else:
        servo_action = None

    state, reward, done, info, counter = None, 0, False, {}, 0
    logging.info("Servoing starting.")
    for counter in range(max_steps):
        state, reward, done, info = env.step(servo_action)
        if done or done_cooldown == 0:
            break

        # Normal servoing, based on correspondences
        servo_action, servo_done, servo_info = servo_module.step(state, info)
        if servo_done:
            done_cooldown -= 1

        # Trajectory actions, based on the trajectory of the demo; dead reckoning.
        # These are the big actions.
        servo_queue = servo_info["traj_acts"] if "traj_acts" in servo_info else None
        if use_trajectory and servo_queue:
            for _ in range(len(servo_queue)):
                trj_act = servo_queue.pop(0)
                print(f"Trajectory action: {trj_act['name']} motion={rec_pprint(trj_act['motion'])}")
                #servo_module.pause()
                trj_act = action_to_abs(env, trj_act)
                trj_act.update(safe_move)
                action_dist_t = 0.01
                for i in range(25):
                    state, reward, done, info = env.step(trj_act)
                    dist = get_action_dist(env, trj_act)
                    if dist < action_dist_t:
                        break
                if dist > action_dist_t:
                    logging.warning("Bad absolute move, dist = %s, t = %s", dist, action_dist_t)
                #servo_module.pause()
            servo_action = None
            continue

        # TODO(max): this should probably be removed, I think this was added for the panda robot.
        if servo_module.config.mode == "pointcloud-abs" and servo_action is not None:
            # do a direct application of action, bypass the env
            assert servo_action["ref"] == "abs"
            servo_action["path"] = "ptp"
            servo_action["blocking"] = True
            #env.robot.move_cart_pos_abs_ptp(servo_action["motion"][0], servo_action["motion"][1])
            #servo_action = None

    if servo_module.view_plots:
        del servo_module.view_plots
    info['ep_length'] = counter

    print(f"\nServoing completed with reward: {reward}, ran for {counter} steps.\n")

    return state, reward, done, info
