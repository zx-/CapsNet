# CapsNet

Implementation of convolutional capsules network described in [paper](https://arxiv.org/pdf/1710.09829.pdf) by 
Sara Sabour, Nicholas Frosst & Geoffrey E Hinton. Contains runnable files that build and train network on MNIST dataset.
Most of the code is documented so feel free to check files for docstrings.

## Model

Check [layers](https://github.com/zx-/CapsNet/blob/master/layers/) for more detailed model description.
![CapsNet](images/graph.PNG)


## Training

Network is trained using MNIST dataset which will be downloaded automatically to `data/mnist`.
Every implemented training outputs results as summaries for tensorboard.
This repository contains 3 runnable files:

#### [simple_model.py](simple_model.py)

This file was mostly used for testing. Contains capsnet without reconstruction. 
Runs 20 iterations over 64 examples and minimizes [margin_loss](https://github.com/zx-/CapsNet/blob/master/loss/loss.py#L13).


#### [capsnet.py](capsnet.py)

Run `capsnet.py -h` to see all parameters.

With default params runs capsnet with reconstruction for 5 epochs over 10000 train (1000 test) 
examples with batch 64. Uses [dataset](data/mnist_dataset.py#L23) and iterators.

#### [capsnet_estimator.py](capsnet_estimator.py)

Run `capsnet_estimator.py -h` to see all parameters.

Estimator implementation of capsnet. Runs over whole dataset and with default parameters uses
batch 64 and 5 epochs.


## Tests

To run [tests](tests/) use:

`python -m unittest discover tests`
