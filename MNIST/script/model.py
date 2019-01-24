import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.transform(x)
        mu = self.fc_mu(out)
        logsigma = self.fc_logsigma(out)
        sigma = torch.exp(logsigma)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, output_dim),
                                       nn.Sigmoid())
    def forward(self, h):
        return self.transform(h)


class VAE(nn.Module):
    def __init__(self, dim_z, dim_image_vars):
        super(VAE, self).__init__()
        self.dim_z = dim_z
        self.dim_image_vars = dim_image_vars

        ## encoder
        self.encoder = Encoder(dim_image_vars, 200, dim_z)
        
        ## decoder
        self.decoder = Decoder(dim_z, 200, dim_image_vars)
        
    def encode(self, x):
        mu_z, sigma_z = self.encoder(x)
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps
        return z, mu_z, sigma_z, eps
    
    def decode(self, z):
        p = self.decoder(z)
        return p
    
    def train_loss(self, x):
        z, mu_z, sigma_z, eps = self.encode(x)
        p = self.decode(z)

        log_PxGz = torch.sum(x*torch.log(p) + (1-x)*torch.log(1-p), -1)
        DKL = torch.sum(0.5*(mu_z**2 + sigma_z**2 - torch.log(sigma_z**2) - 1), -1)
        loss = -torch.mean(log_PxGz - DKL, 0)
        return loss

    def calc_energy(self, x, z, beta):
        z = z.clone().detach().requires_grad_(True)
        p = self.decode(z)
        log_PxGz = torch.sum(x*torch.log(p) + (1-x)*torch.log(1-p), -1)
        log_Pz = torch.sum(-0.5*(z**2 + torch.log(z.new_tensor(2*np.pi))), -1)
        energy = - (log_Pz + beta*log_PxGz)
        energy.backward(torch.ones_like(energy))        
        return energy, z.grad, -log_Pz, -log_PxGz

    def HMC(self, x, epsilon, L, current_q, beta):
        ## sample a new momentum
        current_p = torch.randn_like(current_q)

        #### update momentum and position
        q = current_q.clone().detach()
        p = current_p.clone().detach()        

        ## propagate momentum by a half step at the beginning
        U, grad_U, _, _ = self.calc_energy(x, q, beta)
        p = p - 0.5*epsilon*grad_U

        ## propagate position and momentum alternatively
        ## by a full step
        for i in range(L):
            q = q + epsilon * p
            U, grad_U, _, _ = self.calc_energy(x, q, beta)
            if i != L-1:
                p = p - epsilon*grad_U

        ## propagate momentum by a half step at the end
        p = p - 0.5*epsilon*grad_U

        ## calculate Hamiltonian of current state and proposed state
        current_U, _, _, _ = self.calc_energy(x, current_q, beta)
        current_K = torch.sum(0.5*current_p**2, -1)
        current_E = current_U + current_K
        
        proposed_U, _, _, _ = self.calc_energy(x, q, beta)
        proposed_K = torch.sum(0.5*p**2, -1)
        proposed_E = proposed_U + proposed_K

        ## accept proposed state using Metropolis criterion
        flag = torch.rand_like(proposed_E) <= torch.exp(-(proposed_E - current_E))
        flag = flag.double().reshape(-1,1)
        
        q = flag*q + (1 - flag)*current_q

        return flag, q

    
