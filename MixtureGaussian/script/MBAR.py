import torch
from functions import *
from sys import exit
import sys
sys.path.append("/home/xqding/apps/FastMBAR/FastMBAR/")
from FastMBAR import *
import argparse
import pickle

## command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type = int)
args = parser.parse_args()

torch.set_default_tensor_type(torch.DoubleTensor)

## temperatures
num_beta = 10
beta = np.linspace(0, 1, num_beta)

## step size
#epsilon = np.linspace(0.15, 0.15, num_beta)
epsilon = np.linspace(0.2, 0.1, num_beta)
epsilon = torch.from_numpy(epsilon)

## parameters of target distribution and reference distribution
dim_z = 10
parameters = {}
parameters['mu_0'] = torch.ones(dim_z).double()
parameters['sigma_0'] = 0.5*torch.ones(dim_z)

parameters['mu_1'] = -1*torch.ones(dim_z)
parameters['sigma_1'] = 0.5*torch.ones(dim_z)

parameters['mu_r'] = torch.zeros(dim_z)
parameters['sigma_r'] = 2.0*torch.ones(dim_z)

parameters['beta'] = torch.from_numpy(beta)

## parameters for TRE_HMC sampling
batch_size = num_beta
L = 10
num_steps = 1000
burn_in_num_steps = 100

## sample
energy, exchange_pair = TRE_HMC(calc_energy, L, epsilon,
                                num_steps, burn_in_num_steps,
                                parameters)
num_conf = np.array([energy.shape[-1]/len(beta) for i in range(num_beta)])

mbar = FastMBAR(energy.data.cpu().numpy(), num_conf)
F, _ = mbar.calculate_free_energies(verbose = False)

with open("./output/MBAR_F_{}.pkl".format(args.idx), 'wb') as file_handle:
    pickle.dump(F, file_handle)
