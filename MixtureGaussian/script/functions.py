import numpy as np
import torch

def calc_energy(z, parameters):
    """
    calculate energy U and gradient of U with respect to z

    Args:
        z: tensor of size batch_size x d, where d is the dimension of z
    """
    
    z = z.clone().detach().requires_grad_(True)        
    pi = z.new_tensor(np.pi, requires_grad = False)
    d = z.shape[-1]

    mu_0 = parameters['mu_0']
    sigma_0 = parameters['sigma_0']
    mu_1 = parameters['mu_1']
    sigma_1 = parameters['sigma_1']
    mu_r = parameters['mu_r']
    sigma_r = parameters['sigma_r']
    beta = parameters['beta']    
    
    sigma_0_product = torch.exp(torch.sum(torch.log(sigma_0)))
    sigma_1_product = torch.exp(torch.sum(torch.log(sigma_1)))    
    
    energy_target = -torch.log(
        0.5*1.0/((2*pi)**(d/2.0)*sigma_0_product) * \
        torch.exp(torch.sum(-0.5*((z - mu_0)/sigma_0)**2, -1)) + \
        0.5*1.0/((2*pi)**(d/2.0)*sigma_1_product) * \
        torch.exp(torch.sum(-0.5*((z - mu_1)/sigma_1)**2, -1))
    )

    energy_ref = 0.5*d*torch.log(2*pi) + torch.sum(torch.log(sigma_r)) + \
                 torch.sum(0.5*((z - mu_r)/sigma_r)**2, -1)
    
    energy = beta*energy_target + (1-beta)*energy_ref
    energy.backward(torch.ones_like(energy))

    return energy.data, z.grad, energy_target.data, energy_ref.data
    
def HMC(calc_energy, epsilon, L, current_q, parameters):
    ## sample a new momentum
    current_p = torch.randn_like(current_q)

    #### update momentum and position
    q = current_q.clone().detach()
    p = current_p.clone().detach()        

    epsilon = epsilon.reshape(-1, 1)
    
    ## propagate momentum by a half step at the beginning
    U, grad_U, _, _ = calc_energy(q, parameters)
    p = p - 0.5*epsilon*grad_U

    ## propagate position and momentum alternatively
    ## by a full step
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
    flag = torch.rand_like(proposed_E) <= torch.exp(-(proposed_E - current_E))
    flag = flag.double().reshape(-1,1)
    
    q = flag*q + (1 - flag)*current_q

    flag = parameters['beta'] == 0
    dim_z = q.shape[-1]
    q[flag] = parameters['mu_r'] + parameters['sigma_r']*torch.randn(torch.sum(flag).item(), dim_z)
    return flag, q

def TRE_HMC(L, epsilon, num_steps, burn_in_num_steps, parameters):
    beta = parameters['beta']
    num_beta = len(beta)
    assert(len(beta) == len(epsilon))
    dim_z = parameters['mu_0'].shape[0]
    
    samples = []
    energy = []
    
    z = parameters['mu_r'] + parameters['sigma_r']*torch.randn(num_beta, dim_z)
    samples.append(z)

    for k in range(num_steps):        
        print("Steps: {}".format(k))
        flag, z = HMC(calc_energy, epsilon, L, samples[-1], parameters)
        samples.append(z)
        _, _, energy_target, energy_ref = calc_energy(z, parameters)        
        energy.append((energy_target, energy_ref))

        ## exchange
        step_range = 5
        for i in range(len(beta)-step_range+1):
            for j in range(i+1, i+step_range):
                if j == i:
                    continue
                accept_p = torch.exp(
                    (energy_target[j] - energy_target[i] +
                     energy_ref[i] - energy_ref[j]) *
                    (beta[j] - beta[i])
                )
                
                flag = torch.rand_like(accept_p) <= accept_p

                tmp = samples[-1].clone().detach()
                samples[-1][flag] = samples[-1][flag]
                samples[-1][flag] = tmp[flag]

    energy_target = [e_target for e_target, e_ref in energy[burn_in_num_steps:]]
    energy_ref = [e_ref for e_target, e_ref in energy[burn_in_num_steps:]]

    energy_target = torch.cat(energy_target)
    energy_ref = torch.cat(energy_ref)

    energy = torch.matmul(beta.reshape(-1,1), energy_target.reshape(1,-1)) + \
             torch.matmul(1-beta.reshape(-1,1), energy_ref.reshape(1,-1))
        
    return energy







# def calc_energy(z, parameters):
#     mu_0 = parameters['mu_0']
#     sigma_0 = parameters['sigma_0']
#     mu_1 = parameters['mu_1']
#     sigma_1 = parameters['sigma_1']
#     mu_r = parameters['mu_r']
#     sigma_r = parameters['sigma_r']
#     beta = parameters['beta']    

#     z = z.clone().detach().requires_grad_(True)        
#     pi = z.new_tensor(np.pi, requires_grad = False)

#     d = z.shape[0]
    
#     energy_target = - torch.log(
#         0.5*1.0/((2*pi)**(d/2.0)*torch.sqrt(torch.abs(torch.det(sigma_0)))) * \
#         torch.exp(
#             -0.5*torch.matmul(torch.matmul(torch.inverse(sigma_0), z - mu_0), z - mu_0)
#         ) + \
#         0.5*1.0/((2*pi)**(d/2.0)*torch.sqrt(torch.abs(torch.det(sigma_1)))) * \
#         torch.exp(
#             -0.5*torch.matmul(torch.matmul(torch.inverse(sigma_1), z - mu_1), z - mu_1)
#         )
#         )
    
#     energy_ref = 0.5*torch.matmul(torch.matmul(torch.inverse(sigma_r), z - mu_r), z - mu_r)+\
#                  d/2.0*torch.log(2*pi) + 0.5*torch.log(torch.abs(torch.det(sigma_r)))


#     energy = beta * energy_target + (1.0 - beta)*energy_ref    

#     energy.backward()

#     return energy, z.grad, energy_target.data, energy_ref.data
