import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from model import *


transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST("./data/",
                                           transform = transform,
                                           download = True)
batch_size = 128
mnist_dataloader = DataLoader(mnist_dataset, batch_size = batch_size,
                              shuffle = True)
vae = VAE(10, 28*28)
vae = vae.cuda()

optimizer = optim.Adam(vae.parameters())
num_epoch = 500

log_file = open("./output/log_file.txt", 'w')

for epoch in range(num_epoch):
    for idx, data in enumerate(mnist_dataloader):
        x, label = data
        x = x.squeeze()
        x = x.reshape(-1, 28*28)
        x = x.cuda()        
        tmp = torch.rand_like(x)
        x = tmp < x
        x = x.float()        
        loss = vae.train_loss(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: {:>5d}, idx: {:>5d}, loss: {:>5.2f}".format(epoch, idx, loss.item()))
        print("Epoch: {:>5d}, idx: {:>5d}, loss: {:>5.2f}".format(epoch, idx, loss.item()),
              file = log_file, flush = True)

torch.save({
    'epoch': epoch,
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}, "./output/model_epoch_{}.pt".format(epoch))
