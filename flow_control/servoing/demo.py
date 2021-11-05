import json
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R


from robot_io.recorder.simple_recorder import load_rec_list


def state2matrix(state):
    """
    helper function to return matrix form of bullet object state.

    Args:
        state: position, orientation(euler) as vector
    Returns:
        matrix: 4x4 transformation matrix
    """
    position, orientation = state[0:3], state[3:6]
    assert len(position), len(orientation) == (3, 3)
    matrix = np.eye(4)
    matrix[:3, 3] = position
    matrix[:3, :3] = R.from_euler("xyz", orientation).as_matrix()
    return matrix


class ServoingDemo:
    """
    Servoing Demonstration

    This file contains all of the information required for servoing.
    """
    def __init__(self, recording, episode_num=0, start_index=0):
        self.start_index = start_index

        # set in reset and set_frame
        self.cur_index = None

        # frame data
        self.frame = None
        self.rgb = None
        self.depth = None
        self.mask = None
        self.state = None
        self.grip_action = None

        # set in load_demo (don't move down)
        self.rgb_recording = None
        self.depth_recording = None
        self.mask_recording = None
        self.keep_indexes = None
        self.ee_positions = None
        self.gr_actions = None
        self.env_info = None

        if isinstance(recording, str):
            demo_dict = self.load_from_file(recording, episode_num)
            self.load_demo(demo_dict)
        else:
            # force to load something because of FlowNet size etc.
            demo_dict = recording
            self.load_demo(demo_dict)
        self.max_frame = self.rgb_recording.shape[0] - 1

    def reset(self):
        self.cur_index = self.start_index
        self.set_frame(self.cur_index)

    def step(self):
        self.cur_index += 1
        self.set_frame(self.cur_index)

    def set_frame(self, demo_index):
        """
        set a frame from which to do the servoing
        """
        self.frame = self.keep_indexes[np.clip(demo_index, 0, len(self.keep_indexes) - 1)]
        assert not self.frame > self.max_frame

        self.rgb = self.rgb_recording[self.frame]
        self.depth = self.depth_recording[self.frame]
        self.mask = self.mask_recording[self.frame]
        self.state = self.ee_positions[self.frame]
        self.world_tcp = self.world_tcps[self.frame]
        self.grip_action = float(self.gr_actions[self.frame])

    def get_state(self, frame):
        """ Get virtual state, should be same as env.step """
        frame_dict = dict(
            rgb_unscaled=self.rgb_recording[frame],
            depth=self.depth_recording[frame],
            tcp_pose=self.ee_positions[frame],  # live tcp_pose is (6, )
            world_tcp=self.world_tcps[frame],
            #mask = self.mask_recording[frame]
            #grip_action = float(self.gr_actions[frame])
        )
        return self.rgb_recording[frame], frame_dict

    def flip(self):
        self.rgb_recording = self.rgb_recording[:, ::-1, ::-1, :].copy()
        self.depth_recording = self.depth_recording[:, ::-1, ::-1].copy()
        self.mask_recording = self.mask_recording[:, ::-1, ::-1].copy()
        self.rgb = self.rgb_recording[self.frame]
        self.depth = self.depth_recording[self.frame]
        self.mask = self.mask_recording[self.frame]

    def load_demo(self, demo_dict):
        """
        set a demo that is given as a dictionary, not file
        """
        self.env_info = demo_dict['env_info']
        self.rgb_recording = demo_dict['rgb']
        self.depth_recording = demo_dict["depth"]
        self.mask_recording = demo_dict["mask"]

        self.keep_dict = demo_dict["keep_dict"]
        self.keep_indexes = sorted(demo_dict["keep_dict"].keys())

        self.ee_positions = np.array([[*rs["tcp_pos"], *rs["tcp_orn"]] for rs in demo_dict["state"]])
        self.world_tcps = np.apply_along_axis(state2matrix, 1, self.ee_positions)

        # self.gr_actions = (state_recording[:, -2] > 0.068).astype('float')
        # self.gr_actions = (state_recording[:, -2] > 0.070).astype('float')
        self.gr_actions = demo_dict["actions"][:, 2].astype('float')

    @staticmethod
    def load_from_file(recording, episode_num=None):
        """
        load a demo from files.

        Arguments:
            recording: path to recording containing episode_0.npz
            episode_num: integer to select episode
        """
        logging.info("Loading demo: {} ...".format(recording))
        rec_info_fn = "{}/env_info.json".format(recording)
        keep_dict_fn = "{}/servo_keep.json".format(recording)
        mask_recording_fn = "{}/servo_mask.npz".format(recording)

        # load data
        rec = load_rec_list(recording)
        state = np.array([renv.get_robot_state() for renv in rec])
        rgb = np.array([renv.cam.get_image()[0] for renv in rec])
        depth = np.array([renv.cam.get_image()[1] for renv in rec])
        actions = np.array([renv.get_action()["motion"] for renv in rec], dtype=object)

        try:
            with open(rec_info_fn) as f_obj:
                env_info = json.load(f_obj)
        except FileNotFoundError:
            logging.warning(f"Couldn't find {rec_info_fn}, can't check robot")
            env_info = None

        try:
            with open(keep_dict_fn) as f_obj:
                keep_dict = json.load(f_obj)
                # undo json mangling
                keep_dict = {int(key): val for key, val in keep_dict.items()}
        except FileNotFoundError:
            logging.warning(f"Couldn't find {keep_dict_fn}, servoing will take ages")
            raise FileNotFoundError

        try:
            mask_recording = np.load(mask_recording_fn)["mask"]
        except FileNotFoundError:
            logging.warning(f"Couldn't find {mask_recording_fn}, servoing will fail")
            return NotImplementedError
            mask_recording = np.ones(rgb.shape[0:3], dtype=bool)

        logging.info("Loading completed.")

        return dict(rgb=rgb,
                    depth=depth,
                    state=state,
                    actions=actions,
                    mask=mask_recording,
                    keep_dict=keep_dict,
                    env_info=env_info)
