import hydra

from flow_control.utils_real_robot import move_to_neutral_safe


@hydra.main(config_path="/conf")
def main(cfg):
    """
    Teleoperate the robot with different input devices.
    Depending on the recorder, either record the whole interaction or only if the recording is triggered by the input
    device.

    Args:
        cfg: Hydra config
    """

    robot = hydra.utils.instantiate(cfg.robot)
    move_to_neutral_safe(robot)

    env = hydra.utils.instantiate(cfg.env, robot=robot)
    obs = env.reset()
    env.render()  # call render once, so that input windows appear on top.
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)

    done = False
    with hydra.utils.instantiate(cfg.recorder, env=env) as recorder:
        while not done:
            action, record_info = input_device.get_action()
            print(action.target_orn)

            next_obs, rew, done, info = env.step(action)
            recorder.step(obs, action, next_obs, rew, done, info, record_info)
            env.render()
            obs = next_obs
            done = record_info["done"]

            if done:
                pos, orn = robot.get_tcp_pos_orn()
                move_to_neutral_safe(robot)
                # print(pos.round(3), orn.round(3))
            # done = False
            # input_device.done = False

        recorder.save()


if __name__ == "__main__":
    main()
