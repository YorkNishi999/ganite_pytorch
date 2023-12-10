import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from metrics_all import *

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GANITE function
def ganite_torch(train_x, train_t, train_y, test_x, train_potential_y, test_potential_y, parameters, name, flags):

    # Unpack parameters
    h_dim = parameters['h_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iteration']
    alpha = parameters['alpha']
    beta = parameters['beta']

    # Convert numpy arrays to PyTorch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_t = torch.tensor(train_t, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(train_x, train_t, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    # generator = Generator(train_x.shape[1], h_dim, flags['dropout']).to(device)
    generator = GeneratorDeep(train_x.shape[1], h_dim, flags['dropout'], 15).to(device)
    discriminator = Discriminator(train_x.shape[1], h_dim, flags['dropout']).to(device)
    # inference_net = InferenceNet(train_x.shape[1], h_dim, flags['dropout']).to(device)
    inference_net = InferenceNetDeep(train_x.shape[1], h_dim, flags['dropout'], 15).to(device)

    # Optimizers
    if flags['adamw']:
        G_optimizer = optim.AdamW(generator.parameters(), lr=parameters['lr'], betas=(0.9, 0.999))
        D_optimizer = optim.AdamW(discriminator.parameters(), lr=parameters['lr'], betas=(0.9, 0.999))
        I_optimizer = optim.AdamW(inference_net.parameters(), lr=parameters['lr'], betas=(0.9, 0.999))
    else:
        G_optimizer = optim.Adam(generator.parameters(), lr=parameters['lr'], betas=(0.9, 0.999))
        D_optimizer = optim.Adam(discriminator.parameters(), lr=parameters['lr'], betas=(0.9, 0.999))
        I_optimizer = optim.Adam(inference_net.parameters(), lr=parameters['lr'], betas=(0.9, 0.999))

    # logs
    writer = SummaryWriter(log_dir=os.path.join(f"results/{name}/logs", "tensorboard_log"))
    model_dir = os.path.join(f"results/{name}/models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Training loop
    with tqdm(range(0, iterations)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description("[Epoch %d]" % (epoch))
            g_loss_list = []
            d_loss_list = []
            i_loss_list = []
            for x, t, y in train_loader:
                t = t.unsqueeze(1)
                y = y.unsqueeze(1)

                # Compute losses
                # 1. Discriminator loss
                parameter_setting_discriminator(generator, discriminator, inference_net)

                for _ in range(2):
                    y_tilde = generator(x, t, y)
                    d_logit = discriminator(x, t, y, y_tilde)
                    D_loss = nn.BCEWithLogitsLoss()(d_logit, t)

                    D_optimizer.zero_grad()
                    D_loss.backward(retain_graph=True)
                    D_optimizer.step()

                # 2. Generator loss
                parameter_setting_generator(generator, discriminator, inference_net)
                y_tilde = generator(x, t, y)
                d_logit = discriminator(x, t, y, y_tilde)
                D_loss = nn.BCEWithLogitsLoss()(d_logit, t)
                # G_loss_GAN = D_loss
                G_loss_GAN = -D_loss # minus のほうが良い結果が出る
                y_est = t * y_tilde[:, 1].view(-1, 1) + (1 - t) * y_tilde[:, 0].view(-1, 1)
                G_loss_factual = nn.BCEWithLogitsLoss()(y_est, y)
                G_loss = G_loss_factual + alpha * G_loss_GAN

                G_optimizer.zero_grad()
                G_loss.backward(retain_graph=True)
                G_optimizer.step()

                # 3. Inference loss
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # for x, t, y in train_loader:
                # t = t.unsqueeze(1)
                # y = y.unsqueeze(1)
                parameter_setting_inference_net(generator, discriminator, inference_net)

                y_hat = inference_net(x)
                y_tilde = generator(x, t, y)

                y_t0 = t * y + (1 - t) * y_tilde[:, 1].view(-1, 1)
                I_loss1 = nn.BCEWithLogitsLoss()(y_hat[:, 1].view(-1, 1), y_t0)
                y_t1 = (1 - t) * y + t * y_tilde[:, 0].view(-1, 1)
                I_loss2 = nn.BCEWithLogitsLoss()(y_hat[:, 0].view(-1, 1), y_t1)

                y_ate = torch.sum(t * y - (1 - t) * y)
                y_hat_ate = torch.sum(y_hat[:, 1] - y_hat[:, 0])
                supervised_loss = torch.nn.MSELoss()(y_hat_ate, y_ate)

                I_loss = I_loss1 + I_loss2 + beta * supervised_loss
                # Backward pass and optimize
                I_optimizer.zero_grad()
                I_loss.backward()
                I_optimizer.step()

                # logs
                g_loss_list.append(G_loss.item())
                d_loss_list.append(D_loss.item())
                i_loss_list.append(I_loss.item())
            writer.add_scalar('loss/D_loss_epoch', sum(d_loss_list) / len(d_loss_list), epoch)
            writer.add_scalar('loss/G_loss_epoch', sum(g_loss_list) / len(g_loss_list), epoch)
            writer.add_scalar('loss/I_loss_epoch', sum(i_loss_list) / len(i_loss_list), epoch)

            # calc metric
            parameter_setting_test(generator, discriminator, inference_net)
            train_y_hat = inference_net(train_x).cpu().detach().numpy()
            test_y_hat = inference_net(test_x).cpu().detach().numpy()

            # 1. PEHE
            train_PEHE, interval = PEHE(train_potential_y, train_y_hat)
            writer.add_scalar('metrics/train_PEHE', train_PEHE, epoch)
            writer.add_scalar('metrics/train_PEHE_interval', interval[1], epoch)

            test_PEHE, interval = PEHE(test_potential_y, test_y_hat)
            writer.add_scalar('metrics/test_PEHE', test_PEHE, epoch)
            writer.add_scalar('metrics/test_PEHE_interval', interval[1], epoch)

            # 2. ATE
            train_ATE, interval = ATE(train_potential_y, train_y_hat)
            writer.add_scalar('metrics/train_ATE', train_ATE, epoch)
            writer.add_scalar('metrics/train_ATE_interval', interval[1], epoch)

            test_ATE, interval = ATE(test_potential_y, test_y_hat)
            writer.add_scalar('metrics/test_ATE', test_ATE, epoch)
            writer.add_scalar('metrics/test_ATE_interval', interval[1], epoch)


            # # 3. sqrt_PEHE # comment out for twin
            # test_sqrt_PEHE, interval = sqrt_PEHE(test_potential_y, test_y_hat)
            # writer.add_scalar('test_sqrt_PEHE', test_sqrt_PEHE, epoch)
            # print(f"test_sqrt_PEHE: {test_sqrt_PEHE} ({interval})")


            # Optionally print training progress
            if epoch % 1000 == 0 and epoch != 0:
                g_save_path = os.path.join(model_dir, f"epoch_{epoch}_g.pth")
                d_save_path = os.path.join(model_dir, f"epoch_{epoch}_d.pth")
                i_save_path = os.path.join(model_dir, f"epoch_{epoch}_i.pth")
                torch.save(generator.state_dict(), g_save_path)
                torch.save(discriminator.state_dict(), d_save_path)
                torch.save(inference_net.state_dict(), i_save_path)

    # Generate the potential outcomes for the test set
    g_save_path = os.path.join(model_dir, f"epoch_{epoch}_g.pth")
    d_save_path = os.path.join(model_dir, f"epoch_{epoch}_d.pth")
    i_save_path = os.path.join(model_dir, f"epoch_{epoch}_i.pth")
    torch.save(generator.state_dict(), g_save_path)
    torch.save(discriminator.state_dict(), d_save_path)
    torch.save(inference_net.state_dict(), i_save_path)

    test_y_hat = inference_net(test_x).cpu().detach().numpy()
    return test_y_hat
