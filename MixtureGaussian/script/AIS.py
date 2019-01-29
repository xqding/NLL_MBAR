import numpy as np
import torch
import matplotlib.pyplot as plt
from functions import *

parameters = {}
parameters['mu_0'] = torch.tensor([1.0, 1.0])
parameters['sigma_0'] = torch.diag(torch.tensor([0.5, 0.5]))

parameters['mu_1'] = torch.tensor([-1.0, -1.0])
parameters['sigma_1'] = torch.diag(torch.tensor([0.5, 0.5]))

parameters['mu_r'] = torch.tensor([0.0, 0.0])
parameters['sigma_r'] = torch.diag(torch.tensor([2.0, 2.0]))

parameters['beta'] = 1.0

beta = np.linspace(0,1,1000)
num_beta = len(beta)
epsilon = np.linspace(0.2, 0.2, num_beta)
L = 10

log_w = 0.0
z = parameters['mu_r'] + parameters['sigma_r']*torch.randn(2)
z = torch.randn(2)
for i in range(1, num_beta):
    parameters['beta'] = beta[i]
    _, _, energy_target, energy_ref = calc_energy(z, parameters)
    log_w = log_w + (beta[i] - beta[i-1])*(energy_ref - energy_target)
    flag, z = HMC(calc_energy, epsilon[i], L, z, parameters)

    print(i, flag)
    
