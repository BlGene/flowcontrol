import subprocess
import numpy as np
import os.path as osp
import os

seeds = range(100, 120, 1)
reward_path = './recombination/rewards_debug_new_1'

for seed in seeds:
    seed_dir = osp.join(reward_path, str(seed))
    steps = [osp.join(seed_dir, d) for d in os.listdir(seed_dir)]

    for step in steps:
        subproc_cmd = f'ffmpeg -framerate 8 -i {step}/frame_%06d.jpg -r 25 -pix_fmt yuv420p ' \
                      f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {step}/video.mp4'

        # Run subprocess using the command
        subprocess.run(subproc_cmd, check=True, shell=True)
