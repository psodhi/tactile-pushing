tactile-pushing
===================================================

This repository contains the source code of the paper [Learning Tactile Models for Factor Graph-Based Estimation](https://arxiv.org/pdf/2012.03768.pdf).

# Installation

## Setup python environment and packages

Create a virtual python environment using [Anaconda](https://www.anaconda.com/products/individual):
```
conda create -n pushest python=3.7
conda activate pushest
```

Install the `pushestpy` python package. In the `push_estimation/python` dir execute:
```
pip install -e .
```

## Install gtsam library

Clone [gtsam](https://github.com/borglab/gtsam) repository into a directory of your choice:
```
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.1rc
```

In the `gtsam/` dir execute:
```
mkdir build/ && cd build/
cmake -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION=3.7 ..
make -j
sudo make install

# install python pkg to anaconda environment
python python/setup.py install --force 
```

## Install python wrapper for custom factors

In the `push_estimation/` dir execute:
```
cd wrap
mkdir build && cd build
cmake ..
make -j
sudo make install 
```

## Install pushest library

In the `push_estimation/` dir execute:
```
mkdir build/ && cd build/
cmake .. 
make -j
sudo make install 
make python-install 
```

# Usage 

# Citing
If you find this repository helpful in your publications, please consider citing the following:

```
@inproceedings{sodhi2021tactile,
    title={Learning Tactile Models for Factor Graph-based Estimation},
    author={Sodhi, Paloma and Kaess, Michael and Mukadam, Mustafa and Anderson, Stuart},
    booktitle=IEEE Intl. Conf. on Robotics and Automation (ICRA),
    year={2021},
}
```

# License
This repository is licensed under the [BSD License](LICENSE.md).