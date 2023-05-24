# Federated Deep Equilibrium Learning: A Compact Shared Representation for Personalization
This repository implements all experiments for the paper **Federated Deep Equilibrium Learning: A Compact Shared Representation for Personalization**.

Authors:

Paper Link:

---

## Network Models

This project contains explicit models such as ResNet, Transformer and implicit DEQ models such as DEQ-Resnet and DEQ-Transformer.
- **ResNet**, **DEQ-Resnet**
> Working with datasets: FEMNIST, CIFAR-10, CIFAR-100

- **Transformer**, **DEQ-Transformer**
> Working with datasets: Shakespeare

## Datasets

- FEMNIST: 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28x28 pixels, 200 clients (nature non-IID).
- CIFAR-10: Consist of 60000 32x32 color images in 10 classes including 50000 training images and 10000 test images, 100 users (non-IID by labels)
- CIFAR-100: Consist of 60000 32x32 color images in 100 classes including 50000 training images and 10000 test images, 100 users (non-IID by labels)
- Shakespeare: Text Dataset of Shakespeare Dialogues, 200 users (nature non-IID).

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

Task Name | Dataset        | Model                             | Task Summary              |
----------|----------------|-----------------------------------|---------------------------|
cifar10_image | CIFAR-10      | ResNet34, DEQ-ResNet-M | Image Classification      |
cifar100_image | CIFAR-100      | ResNet34, DEQ-ResNet-M  | Image Classification      |
femnist_image | FEMNIST        | ResNet20, DEQ-ResNet-S     | Character Recognition         |
shakespeare_character | Shakespeare    | Transformer-8, DEQ-Transformer        | Next-Character prediction |

<!-- mdformat on -->

All datasets will be saved at ``./data/``.

Link download the dataset folder: [Link](https://drive.google.com/file/d/1-15jPcxBcQy5okgX-9ddws4Ib0MIPEXq/view?usp=sharing)

Please extract the zip file and replace the original ``./data/`` folder by the extraced ``./data/`` folder.

***The models and the datasets must match!!! Otherwise an error will occur.*** ❗️

## Directory structure

* `main.py`: the main driver script
* `utils/data_utils.py`: helper functions for reading datasets, data loading, logging, etc.
* `utils/jax_utils.py`: helper functions for loss, prediction, structs, etc.
* `utils/model_utils.py`: model definitions in JAX/Haiku
* `deq/`: implementations of DEQ
* `trainers/`: implementations of the algorithms
* `runners/`: scripts for starting an experiment
* `data/`: directory of datasets

## Environment

- OS: Ubuntu 20.04
- Python == 3.9
- JAX == 0.3.25
- Jaxlib == 0.3.25+cuda11.cudnn82
- jaxopt == 0.5.5
- optax == 0.1.4
- chex == 0.1.5
- dm-haiku == 0.0.9

Please follow the installation guidelines in [http://github.com/google/jax](https://github.com/google/jax#pip-installation-gpu-cuda) to install compatible version Jax and Jaxlib version for your machine. The version of packages related to Jax (jaxopt, optax, chex, dm-haiku) may also need to be adjusted for compatible with Jax and Jaxlib.

To install other dependencies: `pip3 install -r requirements.txt`

## Experiments

The general template commands for running an experiment are:

```bash
bash runners/<dataset>/run_<algorithm>.sh [other flags]
```

### Flags

| params | full params        | description                                               | default value | options |
|--------|--------------------|-----------------------------------------------------------|---------------|---------|
|        | --trainer          | Algorithm to run                                          | fedeq_vision  |         |
| -m     | --model            | Local Models                                              | deq_resnet_s  |         |
| -c     | --num_clients      | the total number of clients                               | 100           |         |
| -lpc   | --labels_per_client| the number of labels per clients for CIFAR-10, CIFAR-100  | 5             |         |
| -d     | --dataset          | Dataset Name                                              | femnist       |         |
| -t     | --num_rounds       | the number of global rounds                               | 100           |         |
| -lr    | --learning_rate    | the learning rate of client optimizers                    | 0.01          |         |
| -cr    | --client_rate      | the propotion of clients selected for training each round | 0.1           |         |
| -b     | --batch_size       | batch size                                                | 10            |         |
| -le    | --local_epochs     | the number of epochs training representation              | 5             |         |
| -pe    | --personalized_epochs     | the number of epochs training personalized params  | 3             |         |
| -fs    | --fwd_solver       | Root-finding solver for DEQ models                        | anderson      |         |
| -bs    | --bwd_solver       | Backward solver for DEQ models                            | normal_cg     |         |
|        | --rho              | The value of rho for ADMM consensus optimization          | 0.01          |         |
| -fu    | --frac_unseen      | the propotion of unseen clients                           | 0.0           |         |
| -r     | --repeat           | Number of times to repeat the experiment                  | 1             |         |
| -g     | --gpu              | The ID of GPU used to run experiments                     | 0             |         |

### Example:

Run algorithms to evaluate personalization

```bash
bash runners/femnist/run_fedeq.sh -r 1 -g 0
bash runners/cifar10/run_fedeq.sh -r 1 -g 0
bash runners/cifar100/run_fedeq.sh -r 1 -g 0
bash runners/shakespeare/run_fedeq.sh -r 1 -g 0
```

Run algorithms to evaluate the generalization to unseen clients

```bash
bash runners/femnist/run_fedeq.sh -fu 0.1 -r 1 -g 0
bash runners/cifar10/run_fedeq.sh -fu 0.1 -r 1 -g 0
bash runners/cifar100/run_fedeq.sh -fu 0.1 -r 1 -g 0
bash runners/shakespeare/run_fedeq.sh -fu 0.1 -r 1 -g 0
```

---

## References

### Motley
* Shanshan Wu, Tian Li, Zachary Charles, Yu Xiao, Ziyu Liu, Zheng Xu, and Virginia Smith. Motley: Benchmarking Heterogeneity and Personalization in Federated Learning, 2022 [https://github.com/google-research/federated/tree/master/personalization_benchmark](https://github.com/google-research/federated/tree/master/personalization_benchmark)

### Jax
* James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. [http://github.com/google/jax](http://github.com/google/jax)

### Haiku
* Tom Hennigan, Trevor Cai, Tamara Norman, and Igor Babuschkin. Haiku: Sonnet for JAX, 2020. [http://github.com/deepmind/dm-haiku](http://github.com/deepmind/dm-haiku).

### Jaxopt
* Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares Lopez, Fabian Pedregosa, and Jean-Philippe Vert. Efficient and modular implicit differentiation Advances in Neural Information Processing Systems, 2022. [https://github.com/google/jaxopt](https://github.com/google/jaxopt) 

### FedJax
* Guanhua Wang, Haibo Yu, Shuang Wu, Wei Dai, Jun Feng, Shuai Li, Han Yu, Tian Li, and Jakub Konecny. Fedjax: Federated learning simulation with jax, 2021. [https://fedjax.readthedocs.io/](https://fedjax.readthedocs.io/en/latest/)
