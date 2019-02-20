import numpy as np
import torch
import matplotlib.pyplot as plt
from functions import *
from sys import exit

torch.set_default_tensor_type(torch.DoubleTensor)

## dimension of z
dim_z = 10
batch_size = 10000

## parameters for target distribution and reference distribution
parameters = {}
parameters['mu_0'] = torch.ones(dim_z).double()
parameters['sigma_0'] = 0.5*torch.ones(dim_z)

parameters['mu_1'] = -1*torch.ones(dim_z)
parameters['sigma_1'] = 0.5*torch.ones(dim_z)

parameters['mu_r'] = torch.zeros(dim_z)
parameters['sigma_r'] = 2.0*torch.ones(dim_z)

parameters['beta'] = torch.ones(batch_size)

## intemediate temperatures
num_beta = 1000
beta = np.linspace(0,1,num_beta)

## step size is online adjusted such that
## the accept rate is around 0.65
epsilon_min = 0.05
epsilon_max = 1.00
epsilon_incr = 1.03
epsilon_decr = 0.97
epsilon = torch.tensor([0.5 for i in range(batch_size)])
#epsilon = torch.tensor([0.1 for i in range(batch_size)])

## num of steps in each HMC
L = 10

## accumulate results
log_w = 0.0
z = parameters['mu_r'] + parameters['sigma_r']*torch.randn(batch_size, dim_z)

accept_rate = 0.0
accept_rate_beta = 0.95

for i in range(1, num_beta):
    parameters['beta'][:] = beta[i]
    _, _, energy_target, energy_ref = calc_energy(z, parameters)
    log_w = log_w + (beta[i] - beta[i-1])*(energy_ref.cpu() - energy_target.cpu())

    ## moving variables to GPUs
    z = z.cuda()
    for k in parameters.keys():
        parameters[k] = parameters[k].cuda()
    epsilon = epsilon.cuda()
    
    flag, z = HMC(calc_energy, epsilon, L, z, parameters)

    accept_rate = accept_rate_beta*accept_rate + (1-accept_rate_beta)*flag.float()

    flag = accept_rate <= 0.65
    epsilon[flag] = epsilon[flag] * epsilon_decr

    flag = accept_rate > 0.65
    epsilon[flag] = epsilon[flag] * epsilon_incr    

    flag = epsilon > epsilon_max
    epsilon[flag] = epsilon_max

    flag = epsilon < epsilon_min
    epsilon[flag] = epsilon_min
    
    if i % 100 == 0:
        print(i)
        
w = np.exp(log_w)
mini_batch_size = 10

log_w = [torch.log(torch.mean(w[10*i:10*(i+1)])).item() for i in range(batch_size//mini_batch_size)]

print("result: {:.3f} +- {:.3f}".format(np.mean(log_w), np.std(log_w)))
