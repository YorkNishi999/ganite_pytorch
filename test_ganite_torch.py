import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch

from ganite_torch import ganite_torch
from data_loading import data_loading_twin
from metrics_all import *
from utils import create_result_dir
from model import InferenceNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main (args):
    """Main function for GANITE experiments.

    Args:
    - data_name: twin
    - train_rate: ratio of training data
    - Network parameters (should be optimized for different datasets)
        - h_dim: hidden dimensions
        - iteration: number of training iterations
        - batch_size: the number of samples in each batch
        - alpha: hyper-parameter to adjust the loss importance

    Returns:
    - test_y_hat: estimated potential outcomes
    - metric_results: performance on testing data
    """
    ## Data loading
    train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = \
    data_loading_twin(args.train_rate)

    print(args.data_name + ' dataset is ready.')

    ## Potential outcome estimations by GANITE
    # Set newtork parameters
    parameters = dict()
    parameters['h_dim'] = args.h_dim
    parameters['iteration'] = args.iteration
    parameters['batch_size'] = args.batch_size
    parameters['alpha'] = args.alpha
    parameters['lr'] = args.lr
    name = args.name

    results_path = os.path.join("results", name)
    model_path = os.path.join(results_path, "models")

    # load model
    num_epoch = args.epoch - 1
    inference_net = InferenceNet(train_x.shape[1], parameters['h_dim'])
    inference_net.load_state_dict(torch.load(os.path.join(model_path, f"epoch_{num_epoch}_i.pth")))
    inference_net = inference_net.to(device)


    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)

    test_y_hat = inference_net(test_x).cpu().detach().numpy()

    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. PEHE
    test_PEHE, interval = PEHE(test_potential_y, test_y_hat)
    print(test_PEHE, interval)

    metric_results['PEHE'] = test_PEHE
    metric_results['PEHE_interval'] = interval

    # 2. ATE
    test_ATE, interval = ATE(test_potential_y, test_y_hat)
    metric_results['ATE'] = test_ATE
    metric_results['ATE_interval'] = interval

    ## Print performance metrics on testing data
    print(metric_results)
    with open(f"results/{name}/results_from_testcode.txt", "w") as f:
        f.write(str(metric_results))

    return test_y_hat, metric_results


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['twin'],
        default='twin',
        type=str)
    parser.add_argument(
        '--train_rate',
        help='the ratio of training data',
        default=0.8,
        type=float)
    parser.add_argument(
        '--h_dim',
        help='hidden state dimensions (should be optimized)',
        default=30,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=100,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=256,
        type=int)
    parser.add_argument(
        '--alpha',
        help='hyper-parameter to adjust the loss importance (should be optimized)',
        default=1,
        type=int)
    parser.add_argument(
        '--name',
        help='name of the experiment',
        default='test01',
        type=str)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=5e-4,
        type=float)
    parser.add_argument(
        '--epoch',
        help='epoch',
        default=10000,
        type=int)

    args = parser.parse_args()

    # Calls main function
    test_y_hat, metrics = main(args)
