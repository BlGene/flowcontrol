import os
import json
import shutil
import unittest
import subprocess
import ipdb

from flow_control.servoing.module import ServoingModule

def eval_scores():

    # Convert notebook to script
    convert_cmd = "jupyter nbconvert --to script ./Eval_scores_Online.ipynb"
    convert_cmd = convert_cmd.split()
    subprocess.run(convert_cmd, check=True)

    # Run generated script
    segment_cmd = "python ./Eval_scores_Online.py"
    segment_cmd = segment_cmd.split()
    subprocess.run(segment_cmd, check=True)

    # Cleanup, don't leave file lying around because e.g. github PEP check
    os.remove("./Eval_scores_Online.py")

if __name__ == "__main__":
    eval_scores()