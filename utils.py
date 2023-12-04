"""GANITE Codebase.
Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar,
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets",
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-
"""
import os

def parameter_setting_discriminator(generator, discriminator, inference_net):
    for params in generator.parameters():
        params.requires_grad = False
    for params in discriminator.parameters():
        params.requires_grad = True
    for params in inference_net.parameters():
        params.requires_grad = False
    generator.eval()
    discriminator.train()
    inference_net.eval()

def parameter_setting_generator(generator, discriminator, inference_net):
    for params in generator.parameters():
        params.requires_grad = True
    for params in discriminator.parameters():
        params.requires_grad = False
    for params in inference_net.parameters():
        params.requires_grad = False
    generator.train()
    discriminator.eval()
    inference_net.eval()

def parameter_setting_inference_net(generator, discriminator, inference_net):
    for params in generator.parameters():
        params.requires_grad = False
    for params in discriminator.parameters():
        params.requires_grad = False
    for params in inference_net.parameters():
        params.requires_grad = True
    generator.eval()
    discriminator.eval()
    inference_net.train()

def create_result_dir(name, parameters):
    if not os.path.exists(f"results/{name}"):
        os.makedirs(f"results/{name}")

    with open(f"results/{name}/parameters.txt", "w") as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")
