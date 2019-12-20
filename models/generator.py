import torch.nn as nn

class DCGenerator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(DCGenerator, self).__init__()

        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Deconv layer
            if i == 0:
                deconv = nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0, bias=False)
            else:
                deconv = nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1, bias=False)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # BN layer
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, nn.ReLU(inplace=True))

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out

