# Improved instant-NGP model
This repo is an improved version of another [Instant-NGP repo](https://github.com/kwea123/ngp_pl), and bases on pytorch implementation. 

# Dependencies

So far this has only been tested on Roger's machine. Any feedback is welcomed.

## Python libraries

* Install `pytorch>=1.11.0` by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
* Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
* Install core requirements by `pip install -r requirements.txt`

## TinyCudaNN

This repo relies on a particular version of TinyCudaNN, and has been tested to be *NOT* compatible with the latest TinyCUDANN release.

Installation steps are as follow:

1. git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
2. cd tiny-cuda-nn
3. git checkout cb06141d71ddec7c1478c7d4c88099397168306a
4. Use your favorite editor to edit `include/tiny-cuda-nn/common.h` and set `TCNN_HALF_PRECISION` to `0` (see https://github.com/NVlabs/tiny-cuda-nn/issues/51 for details)
5. cmake . -B build
6. cmake --build build --config RelWithDebInfo -j
7. cd bindings/torch
8. python setup.py install

## Compile CUDA extension of this project

Run `pip install models/csrc/` (please run this each time you `pull` the code)

# Preparing Data

- download preprocessed data at https://drive.google.com/uc?id=10Tj-0uh_zIIXf0FZ6vT7_te90VsDnfCU
- put under datasets/TanksAndTempleBG/

# Run examples!

```
bash scripts/tanks/playground.sh
```
