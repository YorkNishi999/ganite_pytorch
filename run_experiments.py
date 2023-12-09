import subprocess
import csv
import os

# 引数を定義する
names = ['5layer01', '5layer02', '5layer03',
         '5layer04', '5layer05',
         '5layer06', '5layer07', '5layer08', '5layer09', '5layer10']
hdims = [32, 16, 8, 4, 32, 16, 8, 4, 8, 8]
dropouts = [False, False, False, False, False, False, False, False, True, True]
adams = [False, False, False, False, True, True, True, True, False, True]
# python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 200 --iteration 10000
# --batch_size 4096 --alpha 1 --name no_drop_adam02 --lr 1e-5


for (n, h, d, a) in zip(names, hdims, dropouts, adams):
    command = ['python', 'main_ganite_torch.py', '--data_name', 'twin',
               '--train_rate', '0.8', '--h_dim', str(h), '--iteration', '10000',
                '--batch_size', '128', '--alpha', '2', '--beta', '2', '--name', str(n),
                '--lr', '1e-5', '--dropout', str(d), '--adamw', str(a)]

    print(command)
    # コマンドを実行する
    subprocess.run(command)
