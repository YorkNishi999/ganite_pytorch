import subprocess
import csv
import os

# 引数を定義する
names = ['sep_gminus_no_drop_adam01', 'sep_gminus_no_drop_adam02', 'sep_gminus_no_drop_adam03',
         'sep_gminus_no_drop_adam04', 'sep_gminus_no_drop_adam05', 'sep_gminus_drop_adam06',
         'sep_gminus_drop_adam', 'sep_gminus_no_drop_adamw', 'sep_gminus_drop_adamw']
hdims = [50, 100, 150, 200, 250, 300, 300, 300, 300]
dropouts = [False, False, False, False, False, False, True, False, True]
adams = [False, False, False, False, False, False, False, True, True]
# python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 200 --iteration 10000
# --batch_size 4096 --alpha 1 --name no_drop_adam02 --lr 1e-5


for (n, h, d, a) in zip(names, hdims, dropouts, adams):
    command = ['python', 'main_ganite_torch.py', '--data_name', 'twin',
               '--train_rate', '0.8', '--h_dim', str(h), '--iteration', '10000',
                '--batch_size', '4096', '--alpha', '1', '--name', str(n),
                '--lr', '1e-5', '--dropout', str(d), '--adamw', str(a)]

    print(command)
    # コマンドを実行する
    subprocess.run(command)
