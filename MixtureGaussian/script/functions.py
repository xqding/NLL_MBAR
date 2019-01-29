import numpy as np
import torch

def calc_energy(z, parameters):
    mu_0 = parameters['mu_0']
    sigma_0 = parameters['sigma_0']
    mu_1 = parameters['mu_1']
    sigma_1 = parameters['sigma_1']
    mu_r = parameters['mu_r']
    sigma_r = parameters['sigma_r']
    beta = parameters['beta']    

    z = z.clone().detach().requires_grad_(True)        
    pi = z.new_tensor(np.pi, requires_grad = False)

    d = z.shape[0]
    
    energy_target = - torch.log(
        0.5*1.0/((2*pi)**(d/2.0)*torch.sqrt(torch.abs(torch.det(sigma_0)))) * \
        torch.exp(
            -0.5*torch.matmul(torch.matmul(torch.inverse(sigma_0), z - mu_0), z - mu_0)
        ) + \
        0.5*1.0/((2*pi)**(d/2.0)*torch.sqrt(torch.abs(torch.det(sigma_1)))) * \
        torch.exp(
            -0.5*torch.matmul(torch.matmul(torch.inverse(sigma_1), z - mu_1), z - mu_1)
        )
        )
    
    energy_ref = 0.5*torch.matmul(torch.matmul(torch.inverse(sigma_r), z - mu_r), z - mu_r)+\
                 d/2.0*torch.log(2*pi) + 0.5*torch.log(torch.abs(torch.det(sigma_r)))


    energy = beta * energy_target + (1.0 - beta)*energy_ref    

    energy.backward()

    return energy, z.grad, energy_target.data, energy_ref.data

def HMC(calc_energy, epsilon, L, current_q, parameters):
    beta = parameters['beta']
    ## sample a new momentum
    current_p = torch.randn_like(current_q)

    #### update momentum and position
    q = current_q.clone().detach()
    p = current_p.clone().detach()        

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
    #flag = flag.float().reshape(-1,1)
    
    q = flag*q + (1 - flag)*current_q

    return flag, q

def TRE_HMC(L, epsilon, beta, num_steps, burn_in_num_steps, parameters):
    num_beta = len(beta)
    assert(len(beta) == len(epsilon))

    samples = []
    energy = []

    mu_r = parameters['mu_r']
    sigma_r = parameters['sigma_r']
    
    ## initialize conformations
    for i in range(len(beta)):
        samples.append([])
        energy.append([])
        q = torch.randn()
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
