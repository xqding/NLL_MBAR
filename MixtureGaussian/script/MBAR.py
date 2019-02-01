import torch
from functions import *
from sys import exit
import sys
sys.path.append("/home/xqding/apps/FastMBAR/FastMBAR/")
from FastMBAR import *

torch.set_default_tensor_type(torch.DoubleTensor)

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
     np.linspace(0.1, 0.1, 5),
     np.linspace(0.1, 0.1, 10),
     np.linspace(0.1, 0.1, 10),
     np.linspace(0.1, 0.1, 5),
     ))
epsilon = torch.from_numpy(epsilon)

dim_z = 10
batch_size = num_beta

parameters = {}
parameters['mu_0'] = torch.ones(dim_z).double()
parameters['sigma_0'] = 0.5*torch.ones(dim_z)

parameters['mu_1'] = -1*torch.ones(dim_z)
parameters['sigma_1'] = 0.5*torch.ones(dim_z)

parameters['mu_r'] = torch.zeros(dim_z)
parameters['sigma_r'] = 2.0*torch.ones(dim_z)

parameters['beta'] = torch.from_numpy(beta)

L = 10
num_steps = 65
burn_in_num_steps = 10

NLL = []
for i in range(10):
    energy = TRE_HMC(L, epsilon, num_steps, burn_in_num_steps, parameters)
    num_conf = np.array([energy.shape[-1]/len(beta) for i in range(num_beta)])

    mbar = FastMBAR(energy.data.cpu().numpy(), num_conf)
    F_cpu, _ = mbar.calculate_free_energies(verbose = True)
    
    NLL.append(F_cpu[-1])

exit()
