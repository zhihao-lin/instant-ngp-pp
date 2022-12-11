# Improved instant-NGP model
This repo is an improved version of another [Instant-NGP repo](https://github.com/kwea123/ngp_pl), and bases on pytorch implementation. 

# Dependencies

So far this has only been tested on Roger's machine. Any feedback is welcomed. Following installations are tested on a CUDA11.3 machine.

## Python libraries

* Install `pytorch>=1.11.0` by `pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
* Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
* Install core requirements by `pip install -r requirements.txt`

## TinyCudaNN

This repo relies on TinyCudaNN.

It is recommended to build tTinyCudaNN on a CUDA-11.3 machine.

Installation steps are as follows:

1. git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
2. cd tiny-cuda-nn
3. Use your favorite editor to edit `include/tiny-cuda-nn/common.h` and set `TCNN_HALF_PRECISION` to `0` (see https://github.com/NVlabs/tiny-cuda-nn/issues/51 for details)
4. cd bindings/torch
5. python setup.py install

## Compile CUDA extension of this project

Run `pip install models/csrc/` (please run this each time you `pull` the code)

# Preparing Data

- download preprocessed data at https://drive.google.com/uc?id=10Tj-0uh_zIIXf0FZ6vT7_te90VsDnfCU
- put under datasets/TanksAndTempleBG/

# Run examples!

```
python train.py --config configs/Playground.txt
```

This code will validate your model when training procedure finishes.

# Resume training!

```
python train.py --config configs/Playground.txt --ckpt_path PATH/TO/CHECKPOINT/DIR/epoch={n}.ckpt
```

There is a bug of pytorch lightning regarding to progress bar(see https://github.com/Lightning-AI/lightning/issues/13124 for details). 

# Validate your model!

```
python train.py --config configs/Playground.txt --ckpt_path PATH/TO/CHECKPOINT/DIR/epoch={n}.ckpt --val_only
```

# Renderings

```
python render.py --config configs/Playground.txt --weight_path PATH/TO/SLIM/CHECKPOINT/DIR/epoch={n}_slim.ckpt
```