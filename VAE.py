import torch
from torch import nn

# 使用全连接层搭建的VAE
class VAE(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(encoder_dim, latent_dim)
        self.decoder = Decoder(decoder_dim, latent_dim)

    def forward(self, x, device="cpu"):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # 注意，encoder输出的并不是方差\sigma^2，而是log（\sigma）
        mean, logvar = self.encoder(x)

        z = torch.randn_like(mean)
        z = mean + z * torch.exp(logvar)
        z = z.to(device)

        x_new = self.decoder(z)

        return mean, logvar, x_new, z

    def generate(self, z):
        x = self.decoder(z)

        return x

class Encoder(nn.Module):
    def __init__(self, encoder_dims, latent_dim):
        super().__init__()
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_dims)-1):
            self.encoder.append(
                nn.Linear(encoder_dims[i], encoder_dims[i+1], bias=True)
            )
            self.encoder.append(
                nn.ReLU(inplace=True)
            )

        self.fc_mean = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

    def forward(self, x):
        for cnt in range(len(self.encoder)):
            x = self.encoder[cnt](x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, decoder_dim, latent_dim):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_dim) - 1):
            self.decoder.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1], bias=True)
            )
            if i == len(decoder_dim) - 2:
                self.decoder.append(
                    nn.Sigmoid()
                )
            else:
                self.decoder.append(
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        for cnt in range(len(self.decoder)):
            x = self.decoder[cnt](x)
        x = x.reshape(x.shape[0], 1, 28, 28)
        return x


import torch.functional as F

# 使用CNN搭建的VAE
class VAE_CNN(torch.nn.Module):

    def __init__(self, encoder_dim, decoder_dim, latent_dim):
        super(VAE_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 2, 1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()

        self.fc_pre = torch.nn.Linear(16 * 7 * 7, 256)
        self.fc_mean = torch.nn.Linear(256, latent_dim)  # 6*6 from image dimension
        self.fc_logvar = torch.nn.Linear(256, latent_dim)  # 6*6 from image dimension
        

        self.fc_transpose1 = torch.nn.Linear(latent_dim, 256)  # 6*6 from image dimension
        self.fc_transpose2 = torch.nn.Linear(256, 16*7*7)  # 6*6 from image dimension
        self.conv_transpose1 = torch.nn.ConvTranspose2d(16, 6, 3, 2, 1, 1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(6, 1, 3, 2, 1, 1)
        self.final = torch.nn.Sigmoid()


    def forward(self, x, device):
        # Max pooling over a (2, 2) window  [N, 1, 28, 28] --> [N, 6, 14, 14]
        x = self.relu1(self.conv1(x))
        # If the size is a square you can only specify a single number  [N, 6, 14, 14] --> [N, 16, 7, 7]
        x = self.relu2(self.conv2(x))
        x = x.view(-1, 16*7*7)
        x = self.relu3(self.fc_pre(x))

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        z = torch.randn_like(mean)
        z = mean + z * torch.exp(logvar)
        z = z.to(device)

        x = self.relu4(self.fc_transpose1(z))
        x = self.fc_transpose2(x)
        x = x.view(x.shape[0], 16, 7, 7)
        x = self.relu3(self.conv_transpose1(x))
        x = self.final(self.conv_transpose2(x))

        return mean, logvar, x, z

    def generate(self, z):
        x = self.relu4(self.fc_transpose1(z))
        x = self.fc_transpose2(x)
        x = x.view(x.shape[0], 16, 7, 7)
        x = self.relu3(self.conv_transpose1(x))
        x = self.final(self.conv_transpose2(x))

        return x
