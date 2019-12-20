import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from models.generator import DCGenerator
from models.discriminator import DCDiscriminator
from models.init import normal_weights_init

import argparse
import os
from mmcv import Config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN model")
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--save_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from', default="")
    parser.add_argument(
        '--images_dir',
        type=str,
        help='the dir to load true images')
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()
    return args


def plot_loss(G_losses, D_losses, save_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))



def plot_results(img_list, real_batch, save_dir, show=False):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(os.path.join(save_dir, "fake.gif"), writer='pillow')

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    # plt.show()
    plt.savefig(os.path.join(save_dir, "RvsF.png"))


def main():
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    cfg = Config.fromfile(args.config)
    cfg.lr = cfg.lr * args.ngpus

    # Dataset
    dataset = datasets.ImageFolder(root=args.images_dir,
                                   transform=transforms.Compose([
                                       transforms.Resize(cfg.image_size),
                                       transforms.CenterCrop(cfg.image_size),
                                       transforms.ToTensor(),
                                       #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                   ]))
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpus > 0) else "cpu")

    # model
    G = DCGenerator(cfg.G['input_dim'], cfg.G['num_filters'], cfg.G['output_dim']).to(device)
    G.apply(normal_weights_init)
    D = DCDiscriminator(cfg.D['input_dim'], cfg.D['num_filters'], cfg.D['output_dim']).to(device)
    D.apply(normal_weights_init)
    if (device.type == 'cuda') and (args.ngpus > 1):
        G = nn.DataParallel(G, list(range(args.ngpus)))
        D = nn.DataParallel(D, list(range(args.ngpus)))

    fixed_noise = torch.randn(64, cfg.G['input_dim'], 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # loss and optimizer
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    optimizerD = optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)

    # log
    img_list = []
    G_losses = []
    D_losses = []

    # Train
    num_epochs = cfg.num_epochs
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            D.zero_grad()
            real = data[0].to(device)
            bs = real.size(0)
            label = torch.full((bs, ), real_label, device=device)

            # train D
            output = D(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(bs, cfg.G['input_dim'], 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_fake + errD_real
            optimizerD.step()

            # train G
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            # iters += 1

        torch.save(G.state_dict(), os.path.join(args.save_dir, "epoch_%d_G.pth" % epoch))
        torch.save(D.state_dict(), os.path.join(args.save_dir, "epoch_%d_D.pth" % epoch))
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    plot_loss(G_losses, D_losses, args.save_dir)
    plot_results(img_list, next(iter(dataloader))[0].to(device), args.save_dir)

if __name__ == '__main__':
    main()

