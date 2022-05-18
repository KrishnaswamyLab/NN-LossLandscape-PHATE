This folder contains code for the paper "Exploring the geometry and topology of neural network loss landscapes", published at the 20th Symposium on Intelligent Data Analysis (IDA) 2022

## Origin of the code
Code has been adapted from the following GitHub repositories made available online through the MIT license:

https://github.com/pluskid/fitting-random-labels
Which contains simple demo code for the following paper:
> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *Understanding deep learning requires rethinking generalization*. International Conference on Learning Representations (ICLR), 2017. [[arXiv:1611.03530](https://arxiv.org/abs/1611.03530)].

and

https://github.com/tomgoldstein/loss-landscape
Which contains code for the following paper:
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913). NIPS, 2018.


## Setup
The following libraries were used with a Nvidia Titan RTX GPU:
- python 3.7.5
- numpy 1.19.4
- torch 1.6.0
- torchvision 0.7.0
- phate 1.0.6
- h5py 2.10.0
- python-igraph 0.8.3
- scikit-learn 0.24.2
- umap-learn 0.5.1

## Code organization
- `train.py`: main command line interface and training loops;
- `model_wideresnet.py` and `model_mlp.py`: model definition for Wide Resnets and MLPs;
- `cifar10_data.py`: a thin wrapper of torchvision's CIFAR-10 dataset to support (full or partial) random label corruption;
- `cmd_args.py`: command line argument parsing;
- `net_plotter_seed.py`: code for generating random vectors in parameter space, filter-normalizing them and updating a model's parameters by taking a step in parameter space in a given direction;
- `jump_retrain.py`: loads a trained network, generates a random direction specific to a given seed, updated the network's parameters by taking a step in that direction of specified step size and retrains the network;
- `phate_tda_analysis.py`: loads the sampled data (loss and parameter values), computes the cosine PHATE 2D embeddings and saves them, using the PHATE diffusion potentials it computes the loss-level based filtration and the respective persistence diagrams and saves them.

## Command examples
Initialize a WideResNet 28-2 with a random number generator seed of 0 and train it for 200 epochs:
`python train.py --rand_seed=0 --wrn-depth=28 --wrn-widen-factor=2 --data-augmentation --epochs=200 --batch-size=128 --weight-decay=0.0001`

Sample the optimum found through training (previous command) with the jump and retrain procedure and save the parameters at each epoch of retraining:
```
for retrain_seed in 0 1 2 3; do
  for step_size in 0.25 0.5 0.75 1.0; do
    python3 jump_retrain.py --rand_seed=0 --wrn-depth=28--wrn-widen-factor=2 --batch-size=128 --weight-decay=0.0001 --data-augmentation --load_epoch=200 --retrain_rand_seed=$retrain_seed --step_size=$step_size --retrain_epochs=40 --retrain_saves_per_epoch=1 --retrain_learning-rate=0.001 --retrain_batch-size=128 --retrain_weight-decay=0.0001 --retrain_data-augmentation
  done
done
```

Load the sampled data, run PHATE on the parameters, save the 2D embeddings, compute the respective persistence diagrams of dimensions 0 and 1 and save them:
`python3 phate_tda_analysis.py --rand_seed=0 --wrn-depth=28 --wrn-widen-factor=2 --batch-size=128 --weight-decay=0.0001 --load_epoch=200 --retrain_saves_per_epoch=1 --retrain_learning-rate=0.001 --retrain_batch-size=128 --retrain_weight-decay=0.0001 --retrain_data-augmentation`
