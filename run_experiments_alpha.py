import subprocess
import csv
import os

# 引数を定義する
names = ['sep_gminus_no_drop_adamw_alpha01', 'sep_gminus_no_drop_adamw_alpha02',
         'sep_gminus_no_drop_adamw_alpha03', 'sep_gminus_no_drop_adamw_alpha04', 'sep_gminus_no_drop_adamw_alpha05',
         'sep_gminus_no_drop_adamw_alpha06', 'sep_gminus_no_drop_adamw_alpha07', 'sep_gminus_no_drop_adamw_alpha08',
         'sep_gminus_no_drop_adamw_alpha09']
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# python main_ganite_torch.py --data_name twin --train_rate 0.8 --h_dim 200 --iteration 10000
# --batch_size 4096 --alpha 1 --name no_drop_adam02 --lr 1e-5


for (n, a) in zip(names, alphas):
    command = ['python', 'main_ganite_torch.py', '--data_name', 'twin',
               '--train_rate', '0.8', '--h_dim', '300', '--iteration', '10000',
                '--batch_size', '4096', '--alpha', str(a), '--name', str(n),
                '--lr', '1e-5', '--dropout', 'False', '--adamw', 'True']

    print(command)
    # コマンドを実行する
    subprocess.run(command)
