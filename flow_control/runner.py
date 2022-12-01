"""
Testing file for development, to experiment with environments.
"""
import copy
import logging

from robot_io.actions.actions import Action
from robot_io.recorder.simple_recorder import SimpleRecorder, DummyRecorder

from flow_control.utils_coords import get_action_dist, rec_pprint, action_to_current_state
from flow_control.utils_coords import pos_orn_to_matrix, matrix_to_pos_orn


class FastUREnv:
    def __init__(self, env, cartesian_speed=0.25):  # tested up to 0.5, works well
        self.cartesian_speed = cartesian_speed
        self.env = env
        self.cartesian_speed_save = None

    def __enter__(self):
        self.cartesian_speed_save = self.env.robot._cartesian_speed
        self.env.robot._cartesian_speed = self.cartesian_speed

    def __exit__(self, type, value, traceback):
        self.env.robot._cartesian_speed = self.cartesian_speed_save


def action_to_abs(env, action):
    if action["ref"] == "rel":
        new_action = copy.copy(action)
        t_rel = pos_orn_to_matrix(*action["motion"][0:2])
        new_pos, new_orn = matrix_to_pos_orn(env.robot.get_tcp_pose() @ t_rel)
        new_action["motion"] = (new_pos, new_orn, action["motion"][2])
        new_action["ref"] = "abs"
        return new_action
    return action


def act2inst(dict_action, path=None, blocking=None):
    if dict_action is None:
        return None

    act_inst = Action(target_pos=dict_action["motion"][0],
                      target_orn=dict_action["motion"][1],
                      gripper_action=dict_action["motion"][2],
                      ref=dict_action["ref"],
                      path=path if path is not None else dict_action["path"],
                      blocking=blocking if blocking is not None else dict_action["blocking"])
    return act_inst


def run_trajectory_action(env, trj_act):
    trj_act = action_to_abs(env, trj_act)
    trj_act_robot_io = act2inst(trj_act, path="lin", blocking=True)

    dist = 1e9
    action_dist_t = 0.01  # distance threshold for trajectory actions
    state, reward, done, info, counter = None, 0, False, {}, 0
    for i in range(25):
        state, reward, done, info = env.step(trj_act_robot_io)
        dist = get_action_dist(env, trj_act)
        if dist < action_dist_t:
            logging.info("Good absolute move, dist = %.4f, t = %.4f", dist, action_dist_t)
            # servo_module.pause()
            break

    if dist > action_dist_t:
        cur_pos, _ = env.robot.get_tcp_pos_orn()
        logging.warning("Bad absolute move, dist = %.4f, t = %.4f", dist, action_dist_t)
        logging.warning("Goal: %s, current %s", trj_act["motion"][0], cur_pos)

    return state, reward, done, info


def evaluate_control(env, servo_module, max_steps=1000, initial_align=True, done_cooldown=5, recorder=None):
    """
    Function that runs the policy.
    Arguments:
        env: the environment
        servo_module: the servoing module
        max_steps: the number of steps to run servoing for
        initial_align: align with the initial absolute position of the demo
        done_cooldown: episode steps to wait for after servoing is completed (allows simulations to finish).
        recorder: module used to record run: None or BaseRecorder instance
    """
    servo_module.check_calibration(env)
    rec = recorder if recorder is not None else DummyRecorder()
    servo_action = None

    if initial_align:
        initial_act = action_to_current_state(servo_module.demo, grip_action=1)
        run_trajectory_action(env, initial_act)
        servo_action = initial_act

    state, reward, done, info, counter, cmb_info = None, 0, False, {}, 0, {}
    logging.info("Servoing starting.")
    for counter in range(max_steps):
        servo_action_robot_io = act2inst(servo_action, path="lin", blocking=False)
        state, reward, done, info = env.step(servo_action_robot_io)
        if done or done_cooldown == 0:
            rec.step(state, servo_action, None, reward, done, cmb_info)
            break

        # Normal servoing, based on correspondences
        servo_action, servo_done, servo_info = servo_module.step(state, info)
        if servo_done:
            done_cooldown -= 1

        if servo_action is not None:
            assert servo_action["ref"] == "abs"

        cmb_info = {**info, **servo_info}
        rec.step(state, servo_action, None, reward, done, cmb_info)

        # Trajectory actions, are big actions based on the trajectory of the demo; dead reckoning.
        if "traj_acts" in servo_info:
            servo_queue = servo_info["traj_acts"]
            with FastUREnv(env):  # increase speed of linear motions
                for _ in range(len(servo_queue)):
                    trj_act = servo_queue.pop(0)
                    logging.info("Trajectory action: %s motion=%s", trj_act['name'], rec_pprint(trj_act['motion']))
                    # servo_module.pause()
                    state, reward, done, info = run_trajectory_action(env, trj_act)
                    # servo_module.pause()
            servo_action = None
            continue

    logging.info(f"\nServoing completed with reward: {reward}, ran for {counter} steps.\n")
    info['ep_length'] = counter
    rec.save()

    if servo_module.view_plots:
        del servo_module.view_plots

    return state, reward, done, info
