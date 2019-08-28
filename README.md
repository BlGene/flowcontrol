# Installation

This is best done after installing gym-grasping.

```
source activate bullet
cd ../../..  # go to directory containing gym-grasping
git clone git@github.com:lmb-freiburg/flownet2.git
cd flownet2
wget https://lmb.informatik.uni-freiburg.de/people/argusm/flowcontrol/Makefile.config
vim Makefile.config # adjust paths
# for compliling caffe change
export LD_LIBRARY_PATH="" # or at least not the conda stuff
make -j 5 all tools pycaffe
```

Next download the flownet models, this takes a while.
```
cd ./models
head -n 5 download-models.sh | bash
```

# Downloading Demonstrations

Downaload a demonstration sequence.
```
wget https://lmb.informatik.uni-freiburg.de/people/argusm/flowcontrol/bolt_recordings.tar
tar -xvf bolt_recordings.tar
```


## Recording Demonstrations
To record deomonstration use this file, a 3D mouse is nearly always required.
```
cd ../recorders
python curriculum_episode_recorder.py -r --task bolt
```

# Running Code

To run the demo:
```
# to run caffe set environment variables
export PYTHONPATH=${PYTHONPATH}:/home/argusm/lang/flownet2/python
export LD_LIBRARY_PATH=/home/argusm/local/miniconda3/envs/bullet/lib
cd ../gym_grasping/gym_grasping/scripts
python grasping_env_test.py
```


