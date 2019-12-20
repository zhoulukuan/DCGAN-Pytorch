dataroot = "data/faces"

# Generator
G = dict(input_dim=100, output_dim=3, num_filters=[512, 256, 128, 64])

# Discriminator
D = dict(input_dim=3, output_dim=1, num_filters=[64, 128, 256, 512])

# train setting
num_epochs = 10
lr = 2e-4
betas = (0.5, 0.999)
batch_size = 128
image_size = 64