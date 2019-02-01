import numpy as np
import torch
import matplotlib.pyplot as plt
from functions import *
from sys import exit

torch.set_default_tensor_type(torch.DoubleTensor)

dim_z = 10
batch_size = 10

parameters = {}
parameters['mu_0'] = torch.ones(dim_z).double()
parameters['sigma_0'] = 0.5*torch.ones(dim_z)

parameters['mu_1'] = -1*torch.ones(dim_z)
parameters['sigma_1'] = 0.5*torch.ones(dim_z)

parameters['mu_r'] = torch.zeros(dim_z)
parameters['sigma_r'] = 2.0*torch.ones(dim_z)

parameters['beta'] = torch.ones(batch_size)

beta = np.linspace(0,1,2000)
num_beta = len(beta)
epsilon = np.linspace(0.1, 0.1, num_beta)
L = 10

log_w = 0.0
z = parameters['mu_r'] + parameters['sigma_r']*torch.randn(batch_size, dim_z)

for i in range(1, num_beta):
    parameters['beta'][:] = beta[i]
    _, _, energy_target, energy_ref = calc_energy(z, parameters)
    log_w = log_w + (beta[i] - beta[i-1])*(energy_ref - energy_target)
    flag, z = HMC(calc_energy, torch.tensor(epsilon[i]).expand(batch_size), L, z, parameters)

    print(i)
    
