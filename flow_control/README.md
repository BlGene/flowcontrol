# FlowControl

FlowControl is a visual servoing method to copy single demonstration sequences.

This is the implementation for the [FlowControl paper](https://lmb.informatik.uni-freiburg.de/projects/flowcontrol/):

```
@inproceedings{argus20iros,
  author = {Maximilian Argus and Lukas Hermann and Jon Long and Thomas Brox},
  title = {FlowControl: Optical Flow Based Visual Servoing},
  booktitle = {Proceedings of the International Conference on Intelligent Robots and System (IROS)},
  year = 2020,
  address = {Las Vegas, Arizona},
  url = {https://ras.papercept.net/proceedings/IROS20/634.pdf},
}
```


# Installation

## 1. Install gym-grasping

FlowControl can be run in simulation, this requires installing gym-grasping.

## 2. Installing FlowNet2 (Optional but recommended)

FlowControl requires something to find correspondences, the current default
is FlowNet2, this will be installed next.

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

Download the flownet models, this takes a while.
```
cd ./models
head -n 5 download-models.sh | bash
```

Add flownet to the system paths, so that it can be found by flowcontrol.
This needs to be done every time flownet is called (or added to `.bashrc`)

```
# to run caffe set environment variables
export PYTHONPATH=${PYTHONPATH}:/home/argusm/lang/flownet2/python
export LD_LIBRARY_PATH=/home/argusm/local/miniconda3/envs/bullet/lib
```

Test if flownet is working:
```
python flow_module_flownet2.py
```


## 3. Test

Next we test if flowcontrol is working
```
python test_flow.py
```



# Usage

## Recording Demonstrations

To record deomonstration use this file, a 3D mouse is nearly always required.
```
cd ../recorders
python curriculum_episode_recorder.py -r --task bolt
```
