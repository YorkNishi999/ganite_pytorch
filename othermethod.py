import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ganite_torch import ganite_torch
from data_loading import data_loading_twin
from metrics_all import *
from utils import create_result_dir

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestCentroid
from bartpy.sklearnmodel import SklearnModel

def main(args):

    models = [LinearRegression(), NearestCentroid()]
    # models = [SklearnModel()]
    for model in models:
    # model = () # regression
    # # model = () # knn
    # # model = SklearnModel() # bart
        pehe_train_list = []
        pehe_test_list = []
        ate_train_list = []
        ate_test_list = []

        if model == SklearnModel():
            iteration = 1
        else:
            iteration = 1000
        for _ in range(iteration):
            ## Data loading
            train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = data_loading_twin(0.8)
            X = np.column_stack((train_x, train_t))

            model.fit(X, train_y)

            x_train_t1 = np.column_stack((train_x, np.ones(train_x.shape[0])))
            x_train_t0 = np.column_stack((train_x, np.zeros(train_x.shape[0])))
            y_hat_train_1 = model.predict(x_train_t1)
            y_hat_train_0 = model.predict(x_train_t0)
            y_hat_train = np.column_stack((y_hat_train_0, y_hat_train_1))

            p_train, _ = PEHE(train_potential_y, y_hat_train)
            a_train, _ = ATE(train_potential_y, y_hat_train)
            pehe_train_list.append(p_train)
            ate_train_list.append(a_train)

            x_test_t1 = np.column_stack((test_x, np.ones(test_x.shape[0])))
            x_test_t0 = np.column_stack((test_x, np.zeros(test_x.shape[0])))

            y_hat_test_1 = model.predict(x_test_t1)
            y_hat_test_0 = model.predict(x_test_t0)
            y_hat_test = np.column_stack((y_hat_test_0, y_hat_test_1))

            p_test, _  = PEHE(test_potential_y, y_hat_test)
            a_test, _  = ATE(test_potential_y, y_hat_test)
            pehe_test_list.append(p_test)
            ate_test_list.append(a_test)

        avg_pehe_train = np.mean(pehe_train_list)
        std_pehe_train = np.std(pehe_train_list)
        avg_pehe_test = np.mean(pehe_test_list)
        std_pehe_test = np.std(pehe_test_list)

        avg_ate_train = np.mean(ate_train_list)
        std_ate_train = np.std(ate_train_list)
        avg_ate_test = np.mean(ate_test_list)
        std_ate_test = np.std(ate_test_list)

        with open(f"results/other_method_by_montecarlo.txt", "a") as f:
            f.write(f"--------------model: {model}, iteration: {iteration}-------------\n")
            f.write(f"avg_pehe_train: {avg_pehe_train}, std_pehe_train: {std_pehe_train}\n")
            f.write(f"avg_pehe_test: {avg_pehe_test}, std_pehe_test: {std_pehe_test}\n")
            f.write(f"avg_ate_train: {avg_ate_train}, std_ate_train: {std_ate_train}\n")
            f.write(f"avg_ate_test: {avg_ate_test}, std_ate_test: {std_ate_test}\n")
            f.write("-------------------------------------------\n")
        print(f"--------------model: {model}--------------")
        print(f"avg_pehe_train: {avg_pehe_train}, std_pehe_train: {std_pehe_train}")
        print(f"avg_pehe_test: {avg_pehe_test}, std_pehe_test: {std_pehe_test}")
        print(f"avg_ate_train: {avg_ate_train}, std_ate_train: {std_ate_train}")
        print(f"avg_ate_test: {avg_ate_test}, std_ate_test: {std_ate_test}")
        print("-------------------------------------------")


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['twin'],
        default='twin',
        type=str)


    args = parser.parse_args()

    # Calls main function
    main(args)
