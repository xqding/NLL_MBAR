import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import *
from collections import defaultdict
from sys import exit
import sys
sys.path.append("/home/xqding/apps/FastMBAR/FastMBAR/")
from FastMBAR import *

def TRE_HMC(vae, L, epsilon, beta, num_steps, burn_in_num_steps, x):
    num_beta = len(beta)
    assert(len(beta) == len(epsilon))

    samples = []
    energy = []

    batch_size = x.shape[0]
    
    ## initialize conformations
    for i in range(len(beta)):
        samples.append([])
        energy.append([])
        q = torch.randn((batch_size, 10),
                        dtype = x.dtype,
                        device = x.device)
        samples[i].append(q)

    for k in range(num_steps):
        print("Steps: {}".format(k))
        for i in range(len(beta)):
            if beta[i] == 0:
                q = torch.randn((batch_size, 10),
                                dtype = x.dtype,
                                device = x.device)
                samples[i].append(q)
                _, _, energy_pz, energy_pxgz = vae.calc_energy(x, q, 1.0)
                energy[i].append(energy_pxgz.data)

            else:
                flag, q = vae.HMC(x, epsilon[i], L, samples[i][-1], beta[i])
                samples[i].append(q)
                _, _, energy_pz, energy_pxgz = vae.calc_energy(x, q, 1.0)
                energy[i].append(energy_pxgz.data)

                # print("beta: {:.2f}, epsilon: {:.2f}, acceptance ratio: {:.3f}".format(
                #     beta[i],epsilon[i], torch.mean(flag).item()))

        ## exchange
        step_range = 5
        for i in range(len(beta)-step_range+1):
            for j in range(i+1, i+step_range):
                if j == i:
                    continue
                # print((i,j))            
                accept_p = torch.exp((energy[j][-1] - energy[i][-1])*(beta[j] - beta[i]))
                flag = torch.rand_like(accept_p) <= accept_p

                tmp = samples[j][-1].clone().detach()
                samples[j][-1][flag] = samples[i][-1][flag]
                samples[i][-1][flag] = tmp[flag]
            
    for i in range(len(beta)):
        energy[i] = torch.stack(energy[i][burn_in_num_steps:], dim = -1)

    energy = torch.cat(energy, dim = -1)
    num_conformations = energy.shape[-1]
    energy = energy.view([batch_size, 1, num_conformations])
    beta = energy.new_tensor(beta)
    beta = beta.view([1,num_beta, 1])

    energy = energy * beta

    return energy
