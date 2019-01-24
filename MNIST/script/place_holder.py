assert(len(beta) == len(epsilon))

samples = []
energy = []
    
num_steps = 100

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

            
burn_in_num_steps = 50
for i in range(len(beta)):
    energy[i] = torch.stack(energy[i][burn_in_num_steps:], dim = -1)
    
energy = torch.cat(energy, dim = -1)
num_conformations = energy.shape[-1]
energy = energy.view([batch_size, 1, num_conformations])
beta = energy.new_tensor(beta)
beta = beta.view([1,num_beta, 1])

energy = energy * beta

num_conf = np.array([num_conformations/num_beta for i in range(num_beta)])
NLL = []
for i in range(batch_size):
    mbar = FastMBAR(energy[i].data.cpu().numpy(), num_conf)
    F_cpu, _ = mbar.calculate_free_energies(verbose = False)
    print(F_cpu[-1])
    NLL.append(F_cpu[-1])
