# Codebase for "GANITE"

This code is the algorithm of GANITE by PyTorch. The code is inspired by the official repository (https://github.com/jsyoon0823/GANITE).

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

This directory contains implementations of GANITE framework for individualized treatment effect estimations using a real-world dataset.
-   Twin data: http://data.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html

## Requirements

See requirements.txt

## Different parts from the original code.

-   Enable to use PyTorch
-   Enable to use tensorboard for logging.
-   Enable to use experimental name in results dir
-   Different optimization algorithm from Adam to AdamW
-   Enable to use GPU
-   Enable to store the model parameters and results as text under `results` dir

### Command inputs:

-   name: experiment name
-   data_name: twin
-   train_rate: the proportion spliting data into training
-   h_dim: hidden dimensions
-   iterations: number of training iterations
-   batch_size: the number of samples in each batch
-   lr: step size for optimization in networks
-   alpha: hyper-parameter to adjust the loss importance

Note that network parameters should be optimized.

### Example command

1. Run the experiments
```shell
python3 main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 30 --iteration 10000 --batch_size 256 --alpha 1 --name test01
```

2. See log
```shell
tensorboard --logdir .results/test01
```

### Outputs

-   test_y_hat: estimated potential outcomes
-   metric_results: PEHE and ATE
-   model parameters for generator, discriminator, and inference_net
