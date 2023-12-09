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
python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 50 --iteration 10000 --batch_size 4096 --alpha 1 --name no_drop_adam01 --lr 1e-5
python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 200 --iteration 10000 --batch_size 4096 --alpha 1 --name no_drop_adam02 --lr 1e-5
python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 50 --iteration 10000 --batch_size 4096 --alpha 1 --name test_separate_train01 --lr 1e-5
python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 50 --iteration 100000 --batch_size 4096 --alpha 2 --beta 2 --name addinference_loss01 --lr 1e-5
python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 8 --iteration 5000 --batch_size 128 --alpha 2 --beta 2 --name addinference_loss02_arch_5 --lr 1e-5

```

2. See log
```shell
tensorboard --logdir .results/test01
```

3. Test by trained model
```shell
python test_ganite_torch.py --name no_drop_adam01 --epoch 10000 --h_dim 50
```

### Outputs

-   test_y_hat: estimated potential outcomes
-   metric_results: PEHE and ATE
-   model parameters for generator, discriminator, and inference_net

### Citation
```
@software{GANITE_pytorch,
  author = {Yohei Nishimura},
  url = {https://github.com/YorkNishi999/ganite_pytorch},
  version = {1.0.0},
  year = {2023}
}
```

