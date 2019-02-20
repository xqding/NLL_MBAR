import numpy as np
import torch

def calc_energy(z, parameters):
    """
    Calculate energy U and gradient of U with respect to z

    Args:
        z: tensor of size batch_size x d, where d is the dimension of z
    """

    ## make a copy of z
    z = z.clone().detach().requires_grad_(True)
    d = z.shape[-1]     ## dimension of z
    
    pi = z.new_tensor(np.pi, requires_grad = False)

    ## parameters for the Gaussian mixture distribution and reference
    ## Gaussian distribution
    mu_0 = parameters['mu_0']
    sigma_0 = parameters['sigma_0']
    mu_1 = parameters['mu_1']
    sigma_1 = parameters['sigma_1']
    mu_r = parameters['mu_r']
    sigma_r = parameters['sigma_r']

    ## inverse temperature
    beta = parameters['beta']    

    
    sigma_0_product = torch.exp(torch.sum(torch.log(sigma_0)))
    sigma_1_product = torch.exp(torch.sum(torch.log(sigma_1)))    

    ## energy of the target distribution: -logP(z)
    energy_target = -torch.log(
        0.3*1.0/((2*pi)**(d/2.0)*sigma_0_product) * \
        torch.exp(torch.sum(-0.5*((z - mu_0)/sigma_0)**2, -1)) + \
        0.7*1.0/((2*pi)**(d/2.0)*sigma_1_product) * \
        torch.exp(torch.sum(-0.5*((z - mu_1)/sigma_1)**2, -1))
    )

    ## energy of the reference distribution: -logQ(z)
    energy_ref = 0.5*d*torch.log(2*pi) + torch.sum(torch.log(sigma_r)) + \
                 torch.sum(0.5*((z - mu_r)/sigma_r)**2, -1)

    ## energy of the intermediate distribution:
    energy = beta*energy_target + (1-beta)*energy_ref

    ## use backpropgation to calculate force on z
    energy.backward(torch.ones_like(energy))

    return energy.data, z.grad, energy_target.data, energy_ref.data
    
def HMC(calc_energy, epsilon, L, current_q, parameters):
    ''' Hamiltonian Monte Carlo (HMC) Algorithm
    
    '''
    ## sample a new momentum
    current_q.requires_grad_(False)
    current_p = torch.randn_like(current_q)
    

    #### update momentum and position ####
    ## proposed (q,p)
    q = current_q.clone().detach()
    p = current_p.clone().detach()        

    ## step size
    epsilon = epsilon.reshape(-1, 1)

    ## propogate state (q,p) using Hamiltonian equation with
    ## leapfrog integration
    
    ## propagate momentum by a half step at the beginning
    U, grad_U, _, _ = calc_energy(q, parameters)
    p = p - 0.5*epsilon*grad_U

    ## propagate position and momentum alternatively by a full step
    ## L is the number of steps
    for i in range(L):
        q = q + epsilon * p
        U, grad_U, _, _ = calc_energy(q, parameters)
        if i != L-1:
            p = p - epsilon*grad_U

    ## propagate momentum by a half step at the end
    p = p - 0.5*epsilon*grad_U

    ## calculate Hamiltonian of current state and proposed state
    current_U, _, _, _ = calc_energy(current_q, parameters)
    current_K = torch.sum(0.5*current_p**2, -1)
    current_E = current_U + current_K
    
    proposed_U, _, _, _ = calc_energy(q, parameters)
    proposed_K = torch.sum(0.5*p**2, -1)
    proposed_E = proposed_U + proposed_K

    ## accept proposed state using Metropolis criterion
    flag_accept = torch.rand_like(proposed_E) <= torch.exp(-(proposed_E - current_E))
    current_q[flag_accept] = q[flag_accept]

    ## if beta == 0, sample from the Gaussian reference distribution directly
    flag_beta_0 = parameters['beta'] == 0
    dim_z = q.shape[-1]    
    current_q[flag_beta_0] = parameters['mu_r'] + \
                             parameters['sigma_r'] * \
                             torch.randn(torch.sum(flag_beta_0).item(),dim_z,
                                         device = q.device)
    
    return flag_accept, current_q

def TRE_HMC(calc_energy, L, epsilon, num_steps, burn_in_num_steps, parameters):
    ''' Temperature Replica Exchange Hamiltonian Monte Carlo (TRE_HMC)

    '''
    
    # inverse temperatures
    beta = parameters['beta'] 
    num_beta = len(beta)
    assert(len(beta) == len(epsilon))

    ## dimenstion of z
    dim_z = parameters['mu_0'].shape[0]

    ## save samples and corresponding energies for each temperature
    samples = []
    energy = []

    ## for each temperature, use the samples from beta = 0, i.e., from
    ## the reference distribution as initial samples.
    z = parameters['mu_r'] + \
        parameters['sigma_r'] * torch.randn(num_beta, dim_z)
    samples.append(z)

    ## parameters for adjusting step size epsilon
    epsilon_min = 0.05
    epsilon_max = 1.00
    epsilon_incr = 1.03
    epsilon_decr = 0.97

    ## keep track of accept rate
    accept_rate = 0.0
    accept_rate_beta = 0.95

    ## keep track of exchange
    exchange_idx_pair = []

    ## start TRE_HMC
    for k in range(num_steps):
        ## run Hamiltonian Monte Carlo and save samples and energies
        ## In HMC, states are propgrated independently between different
        ## temperature
        flag, z = HMC(calc_energy, epsilon, L, samples[-1], parameters)                
        samples.append(z)
        _, _, energy_target, energy_ref = calc_energy(z, parameters)
        energy.append((energy_target, energy_ref))

        ## exponential average of accept rate
        accept_rate = accept_rate_beta*accept_rate + (1-accept_rate_beta)*flag.float()

        # ## adjust step size epsilon based on accept rate
        # flag = accept_rate <= 0.65
        # epsilon[flag] = epsilon[flag] * epsilon_decr

        # flag = accept_rate > 0.65
        # epsilon[flag] = epsilon[flag] * epsilon_incr    

        # flag = epsilon > epsilon_max
        # epsilon[flag] = epsilon_max

        # flag = epsilon < epsilon_min
        # epsilon[flag] = epsilon_min        
        
        # ## do the temperature replica exchange for the samples at the
        # ## current step.
        # ## samples from each temperature are attempted to exchange
        # ## with samples from other nearby temperatures.
        # exchange_idx_pair.append([])        
        # beta_range = 1
        # for i in range(len(beta) - beta_range):
        #     for j in range(i+1, i + beta_range + 1):
        #         accept_prop = torch.exp(
        #             (energy_target[j] - energy_target[i] +
        #              energy_ref[i] - energy_ref[j]) *
        #             (beta[j] - beta[i])
        #         )
                
        #         flag = torch.rand_like(accept_prop) <= accept_prop
                
        #         if flag.item():                                        
        #             tmp = samples[-1][i].clone().detach()
        #             samples[-1][i] = samples[-1][j]
        #             samples[-1][j] = tmp

        #             tmp = energy_target[i]
        #             energy_target[i] = energy_target[j]
        #             energy_target[j] = tmp

        #             tmp = energy_ref[i]
        #             energy_ref[i] = energy_ref[j]
        #             energy_ref[j] = tmp

        #             exchange_idx_pair[-1].append((i,j))
                    
        # print(k)
        # print("accept rate", accept_rate)
        # print("epsilon", epsilon)
                
    energy_target = [e_target for e_target, e_ref in energy[burn_in_num_steps:]]
    energy_ref = [e_ref for e_target, e_ref in energy[burn_in_num_steps:]]

    energy_target = torch.cat(energy_target)
    energy_ref = torch.cat(energy_ref)

    energy = torch.matmul(beta.reshape(-1,1), energy_target.reshape(1,-1)) + \
             torch.matmul(1-beta.reshape(-1,1), energy_ref.reshape(1,-1))
        
    return energy, exchange_idx_pair
