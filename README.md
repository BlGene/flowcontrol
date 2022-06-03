<p align="center">
<img src="docs/flow_logo.svg" width="100"/>
</p>

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

1. Install gym-grasping. See repo's README.
2. Install FlowControl. ```pip install -e .```
3. Install an optical flow algorithm. See below.
4. Test ```python test_flow.py```


## Installing an Optical Flow Algorithm

FlowControl requires  at least one optical flow algorithm, so choose between RAFT and FlowNet2.


### Installing RAFT (Recommended)

RAFT is a bit easier to install as it does not need to be compiled.
```
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT; ./download_models.sh
conda install pytorch torchvision cudatoolkit matplotlib tensorboard scipy -c pytorch
export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/RAFT/core

python flow_control/flow/module_raft.py  # to test.
```

Note: I'm working with python3.8, and the RAFT specified versions don't
install. It's working for me with pytorch 1.8.1 torchvision 0.9.1 and
cudatoolkit 10.2.89.


### Installing FlowNet2 (Optional)

FlowNet2 is a bit older and still uses caffe, but it works quite well.

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

Download the FlowNet models, this takes a while.
```
cd ./models
head -n 5 download-models.sh | bash
```

Add FlowNet to the system paths, so that it can be found by FlowControl.
This needs to be done every time FlowNet is called (or added to `.bashrc`)

```
# to run caffe set environment variables
export PYTHONPATH=$PYTHONPATH:/home/argusm/lang/flownet2/python
export LD_LIBRARY_PATH=/home/argusm/local/miniconda3/envs/bullet/lib
```

Test if FlowNet is working:
```
python flow_module_flownet2.py
```


# Recording Demonstrations

To record demonstrations use this file, this is almost always done using the 3D mouse.

```
cd ../recorders
python curriculum_episode_recorder.py -r --task bolt
```


# Development:

## Geometric Conventions

Coordinate conventions:
1. Right-hand coordinate system
2. Multiply from left: `T_2 @ T_1`
3. Variable naming: `T_tcp_flange` goes from TPC <- flange.
4. Quaternion order: `(x, y, z, w)`

All computatoins as matrix multiplications.
