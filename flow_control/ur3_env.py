import time

from robot_io.envs.robot_env import RobotEnv
from robot_io.utils.utils import restrict_workspace


class UR3RobotEnv(RobotEnv):
    """
    Example env class that can be used for teleoperation.
    Should be adapted to handle specific tasks.
    """
    def step(self, action):
        """
        Execute one action on the robot.

        Args:
            action (dict): {"motion": (position, orientation, gripper_action), "ref": "rel"/"abs"}
                a dict with the key 'motion' which is a cartesian motion tuple
                and the key 'ref' which specifies if the motion is absolute or relative
        Returns:
            obs (dict): agent's observation of the current environment.
            reward (float): Currently always 0.
            done (bool): whether the episode has ended, currently always False.
            info (dict): contains auxiliary diagnostic information, currently empty.

        obs should not contain the keys: action, done, **rew**, or info as this will
        cause call problems when recording.
        """
        if action is None:
            return self._get_obs(), 0, False, {}
        assert isinstance(action, dict) and len(action["motion"]) == 3

        target_pos, target_orn, gripper_action = action["motion"]
        ref = action["ref"]

        if ref == "abs":
            target_pos = restrict_workspace(self.workspace_limits, target_pos)
            self.robot.move_async_cart_pos_abs_ptp(target_pos, target_orn)
        elif ref == "rel":
            self.robot.move_async_cart_pos_rel_ptp(target_pos, target_orn)
        else:
            raise ValueError

        if gripper_action == 1:
            self.robot.open_gripper()
        elif gripper_action == -1:
            self.robot.close_gripper()
        else:
            raise ValueError

        self.fps_controller.step()
        if self.show_fps:
            print(f"FPS: {1 / (time.time() - self.t1)}")
        self.t1 = time.time()

        obs = self._get_obs()

        reward = self.get_reward(obs, action)

        termination = self.get_termination(obs)

        info = self.get_info(obs, action)

        return obs, reward, termination, info

    def render(self, mode="human"):
        """
        Renders the environment.
        If mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.

        Args:
            mode (str): the mode to render with
        """
        if mode == "human":
            self.camera_manager.render()
            self.robot.visualize_joint_states()
            self.robot.visualize_external_forces()
