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
z = torch.randn((batch_size, 10), dtype = x.dtype, device = x.device)
L = 10

beta = np.concatenate(
    (np.array([0.0]),
     np.linspace(0.01, 0.1, 5, endpoint = False),
     np.linspace(0.1, 0.5, 10, endpoint = False),
     np.linspace(0.5, 0.9, 10, endpoint = False),
     np.linspace(0.9, 1.0, 5),
     ))
num_beta = len(beta)
epsilon = np.concatenate(
    (np.array([1.0]),
     np.linspace(0.4, 0.3, 5),
     np.linspace(0.20, 0.15, 10),
     np.linspace(0.15, 0.11, 10),
     np.linspace(0.11, 0.11, 5),
     ))

num_steps = 60
burn_in_num_steps = 10
energy = TRE_HMC(vae, L, epsilon, beta, num_steps, burn_in_num_steps, x)

num_conf = np.array([energy.shape[-1]/len(beta) for i in range(num_beta)])
NLL = []
for i in range(batch_size):
    mbar = FastMBAR(energy[i].data.cpu().numpy(), num_conf)
    F_cpu, _ = mbar.calculate_free_energies(verbose = False)
    print(F_cpu[-1])
    NLL.append(F_cpu[-1])
exit()
