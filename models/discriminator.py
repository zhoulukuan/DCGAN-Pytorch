import torch.nn as nn

class DCDiscriminator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(DCDiscriminator, self).__init__()

        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            if i == 0:
                conv = nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1, bias=False)
            else:
                conv = nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1, bias=False)

            conv_name = 'conv' + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            # Batch normalization
            if i != 0:
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, nn.LeakyReLU(0.2, inplace=True))

        self.output_layer = nn.Sequential(
            nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out