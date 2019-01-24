import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import *
from functions import *
from collections import defaultdict
from sys import exit
import sys
sys.path.append("/home/xqding/apps/FastMBAR/FastMBAR/")
from FastMBAR import *

checkpoint = torch.load('./output/model_epoch_499.pt')
vae = VAE(10, 28*28)
vae.load_state_dict(checkpoint['model_state_dict'])
vae = vae.double()
vae = vae.cuda()

transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST("./data/",
                                           transform = transform,
                                           download = True)
batch_size = 128
mnist_dataloader = DataLoader(mnist_dataset,
                              batch_size = batch_size)

idx, (x, label) = next(enumerate(mnist_dataloader))
x = x.squeeze()
x = x.reshape(-1, 28*28)
x = x.cuda()
tmp = torch.rand_like(x)
x = tmp < x
x = x.double()
L = 10

beta = np.linspace(0,1,2000)
num_beta = len(beta)
epsilon = np.linspace(0.3, 0.11, num_beta)
L = 10

z = torch.randn((batch_size, 10), dtype = x.dtype, device = x.device)
log_w = 0.0
for i in range(1, num_beta):
    print(i)
    _, _, _, energy_pxgz = vae.calc_energy(x, z, 1.0)
    log_w = log_w + (beta[i] - beta[i-1])*energy_pxgz.data
    flag, z = vae.HMC(x, epsilon[i], L, z, beta[i])
