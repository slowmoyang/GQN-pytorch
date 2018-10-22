import torch
import torch.nn as nn
import torch.nn.functional as F

class Core(nn.Module):
    def __init__(self, channels_chz=64, channels_u=128, input_channel=64+7+256+64):
        super().__init__()
        self.lstm_tanh = nn.Conv2d(input_channel, channels_chz, kernel_size=5,stride=1, padding=2)
        self.lstm_i = nn.Conv2d(input_channel, channels_chz, kernel_size=5, stride=1, padding=2)
        self.lstm_f = nn.Conv2d(input_channel, channels_chz, kernel_size=5, stride=1, padding=2)
        self.lstm_o = nn.Conv2d(input_channel, channels_chz, kernel_size=5, stride=1, padding=2)
        self.deconv_h = nn.ConvTranspose2d(channels_chz, channels_u, kernel_size=4, stride=4, padding=0)

    def forward(self, prev_hg, prev_cg, prev_u, prev_z, v, r):
        v = torch.reshape(v, v.shape + (1, 1))
        v = v.repeat((1,1,)+ prev_hg.shape[2:])

        lstm_in = torch.cat((prev_hg, v, r, prev_z), dim=1)
        forget_gate = torch.sigmoid(self.lstm_f(lstm_in))
        input_gate = torch.sigmoid(self.lstm_i(lstm_in))
        next_c = forget_gate * prev_cg + input_gate * torch.tanh(self.lstm_tanh(lstm_in))
        next_h = torch.sigmoid(self.lstm_o(lstm_in)) * torch.tanh(next_c)
        next_u = self.deconv_h(next_h) + prev_u

        return next_h, next_c, next_u


class Prior(nn.Module):
    def __init__(self, channels_z=64):
        super().__init__()
        self.mean_z = nn.Conv2d(channels_z, channels_z, kernel_size=5, stride=1, padding=2)
        self.ln_var_z = nn.Conv2d(channels_z, channels_z, kernel_size=5, stride=1, padding=2)

    def compute_mean_z(self, h):
        return self.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.ln_var_z(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        latent_z = self.reparameterize(mean, ln_var)
        return mean, ln_var, latent_z


class ObservationDistribution(nn.Module):
    def __init__(self, channel=128):
        super().__init__()
        self.mean_x = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0)

    def compute_mean_x(self, u):
        return self.mean_x(u)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, u, ln_var):
        mean = self.compute_mean_x(u)
        latent_z = self.reparameterize(mean, ln_var)
        return latent_z