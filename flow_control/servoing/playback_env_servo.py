import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from robot_io.envs.playback_env import PlaybackEnv
from robot_io.recorder.simple_recorder import SimpleRecorder

class PlaybackEnvServo(PlaybackEnv):
    """
    Loads several pre-recorded demonstration frames.

    Returns:
        list of PlaybackEnvSteps
    """
    def __init__(self, recording_dir, keep_dict="file", load="all", fg_masks=None):
        # first load the keep_dict, to double check files
        self.keep_dict = self.load_keep_dict(recording_dir, keep_dict)
        if load == "all":
            # load all frames, first have to find which ones there are.
            load_base == "all"
        elif load == "keep":
            # load keyframes, first have to find which ones they are.
            load_base = sorted(self.keep_dict.keys())
        else:
            raise ValueError

        super().__init__(recording_dir, load=load_base)

        mask_recording_fn = Path(recording_dir) / "servo_mask.npz"
        try:
            mask_file = np.load(mask_recording_fn)
            m_masks = mask_file["mask"]
            fg_obj = mask_file["fg"]
            fg_masks_from_file = np.array([m == f for m, f in zip(m_masks, fg_obj)])
        except FileNotFoundError:
            if fg_masks is None:
                logging.warning(f"Couldn't find {mask_recording_fn}, servoing will fail")

        if fg_masks is None:
            fg_masks = fg_masks_from_file

        assert fg_masks is None or  len(fg_masks) > len(self.keep_indexes)

        self.fg_masks = fg_masks

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __getattr__(self, name):
        index = self.__getattribute__("index")
        return getattr(self.steps[index], name)

    def get_keep_dict(self):
        return self.keep_dict[self.index]

    def get_fg_mask(self):
        return self.fg_masks[self.index]

    def to_list(self):
        return self.steps


    def load_keep_dict(self, recording_dir, keep_dict):
        # first check if we find these things on files
        if keep_dict == "file":
            keep_dict_fn = Path(recording_dir) / "servo_keep.json"
            try:
                with open(keep_dict_fn) as f_obj:
                    keep_dict_from_file = json.load(f_obj)
                # undo json mangling
                keep_dict_e = {int(key): val for key, val in keep_dict_from_file.items()}

            except FileNotFoundError:
                logging.warning(f"Couldn't find {keep_dict_fn}, servoing will take ages")
                keep_dict_e = {k: None for k in range(len(self.steps))}

        elif keep_dict is None:
            keep_dict_e = {k: None for k in range(len(self.steps))}
        else:
            assert isinstance(keep_dict, dict)
            keep_dict_e = keep_dict
        return keep_dict_e

    @staticmethod
    def freeze(env, reward=0, done=False, fg_mask=None):
        """
        Create a static view of a single env step w/ extra info for servoing.
        """
        obs, info = env._get_obs()
        keep_dict = {0: None}
        assert fg_mask.shape == env.camera.resolution
        fg_masks = fg_mask[np.newaxis, :]

        with TemporaryDirectory() as tmp_dir_name:
            simp_rec = SimpleRecorder(env, tmp_dir_name)
            action = dict(motion=(None, None, 1))
            simp_rec.step(action, obs, reward=reward, done=done, info=info)
            simp_rec.save()
            demo_pb = PlaybackEnvServo(tmp_dir_name, keep_dict=keep_dict,
                                       fg_masks=fg_masks)
        return demo_pb
