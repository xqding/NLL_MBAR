import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import *

checkpoint = torch.load('./output/model_epoch_499.pt')
vae = VAE(10, 28*28)
vae.load_state_dict(checkpoint['model_state_dict'])

z = torch.randn(25,10)
x = vae.decode(z)

fig = plt.figure(0)
fig.clf()
gs = gridspec.GridSpec(5,5)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(25):
    axes = plt.subplot(gs[i])
    axes.imshow(1-x[i,:].cpu().data.numpy().reshape(28,28),
                cmap = 'binary')
    axes.axis('off')

#fig.savefig("./output/rollout_result.eps")
plt.show()
#sys.exit()





