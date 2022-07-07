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
    def __init__(self, recording_dir, keep_dict="file", fg_masks=None):
        super().__init__(recording_dir, keep_dict=keep_dict)

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
        assert fg_masks is None or len(self.steps) == len(fg_masks)
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