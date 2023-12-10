import subprocess
import csv
import os

# 引数を定義する
# names = ['5layer03', '5layer04', '5layer05', '5layer06', '5layer07', '5layer08', '5layer09', '5layer10']
# names = ['new_5layer03_5000_lrsmal', 'new_5layer04_5000']
names = ['Deep01_8', 'Deep02_8', 'Deep03_8', 'Deep04_8']
hdims = [8, 8, 8, 8]
# hdims = [8, 4]
# dropouts = [False, False, False, False, False, False, True, True]
dropouts = [False, False, True, True]
# adams = [False, False, True, True, True, True, False, True]
adams = [True, True, True, True]
batchs = [64, 128, 64, 128]
# python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 200 --iteration 10000
# --batch_size 4096 --alpha 1 --name no_drop_adam02 --lr 1e-5


for (n, h, d, a, b) in zip(names, hdims, dropouts, adams, batchs):
    command = ['python', 'main_ganite_torch.py', '--data_name', 'twin',
               '--train_rate', '0.8', '--h_dim', str(h), '--iteration', '5000',
                '--batch_size', str(b), '--alpha', '2', '--beta', '2', '--name', str(n),
                '--lr', '1e-5', '--dropout', str(d), '--adamw', str(a)]

    print(command)
    # コマンドを実行する
    subprocess.run(command)
