tactile-pushing
===================================================

This repository contains the source code of the paper [Learning Tactile Models for Factor Graph-Based Estimation](https://arxiv.org/pdf/2012.03768.pdf).

# Installation

Install the `pushestpy` python package locally. In the `push_estimation/pushestpy` dir execute:
```
pip install -e .
```

Create a virtual python environment using [Anaconda](https://www.anaconda.com/products/individual):
```
conda create -n env_push python=3.7
conda activate env_push
```

Install [gtsam](https://github.com/borglab/gtsam). Start by cloning the gtsam repository:
```
git clone https://github.com/borglab/gtsam.git
git checkout tags/4.0.0
```

Build and install the gtsam library:
```
cmake -DGTSAM_INSTALL_CYTHON_TOOLBOX=ON -DGTSAM_PYTHON_VERSION=3.7 ..
make -j
make install
```
If doing a local install, additionally pass in the install path `-DCMAKE_INSTALL_PREFIX=../install`.

Build and install the pushestcpp library for custom factors. In the `push_estimation/` dir execute:
```
mkdir build/ && cd build/
cmake ..
make -j
make install
```
# Usage 

# Citing
If you find this repository helpful in your publications, please cite the following:

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