import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import math
import sys
import os

from gqn_nn import representation, inference, generator, functions


class GQN(nn.Module):
    def __init__(self, gpu_enable=True):
        super(GQN, self).__init__()
        # Device configuration
        self.device = torch.device('cuda' if (torch.cuda.is_available() and gpu_enable) else 'cpu')
        
        # hyperparameter
        self.image_size = (64, 64)
        self.chrz_size = (16, 16)  # needs to be 1/4 of image_size
        self.channels_r = 256
        self.channels_chz = 64
        self.inference_channels_map_x = 32
        self.inference_share_core = False
        self.inference_share_posterior = False
        self.generator_generation_steps = 12
        self.generator_channels_u = 128
        self.generator_share_core = False
        self.generator_share_prior = False
        self.pixel_sigma_i = 2.0
        self.pixel_sigma_f = 0.7
        self.pixel_n = 2 * 1e5


        # define network 
        # === representation network ===
        self.representation_network = representation.TowerNetwork(channels_r=self.channels_r)


        # === inference network ===
        cores = []
        num_cores = 1 if self.inference_share_core else self.generator_generation_steps
        for t in range(num_cores):
            # LSTM core
            core = inference.Core(channels_chz=self.channels_chz, 
                                  input_channel=self.channels_chz*2 + self.inference_channels_map_x + 7 + self.channels_r)
            cores.append(core)
        self.inference_cores = nn.Sequential(*cores)

        # z posterior sampler
        posteriors = []
        num_posteriors = 1 if self.inference_share_posterior else self.generator_generation_steps
        for t in range(num_posteriors):
            posterior = inference.Posterior(channels_z=self.channels_chz)
            posteriors.append(posterior)
        self.inference_posteriors = nn.Sequential(*posteriors)

        # x downsampler
        self.inference_downsampler = inference.Downsampler(channels=self.inference_channels_map_x)


        # === generator network ===
        # LSTM core
        cores = []
        num_cores = 1 if self.generator_share_core else self.generator_generation_steps
        for t in range(num_cores):
            core = generator.Core(channels_chz=self.channels_chz, channels_u=self.generator_channels_u,
                                  input_channel=self.channels_chz*2 + 7 + self.channels_r)
            cores.append(core)
        self.generation_cores = nn.Sequential(*cores)

        # z prior sampler
        priors = []
        num_priors = 1 if self.generator_share_prior else self.generator_generation_steps
        for t in range(num_priors):
            prior = generator.Prior(channels_z=self.channels_chz)
            priors.append(prior)
        self.generation_priors = nn.Sequential(*priors)

        # observation sampler
        self.generation_observation = generator.ObservationDistribution(channel=self.generator_channels_u)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, 10.0)
                if m.bias is not None:
                    m.bias.data.zero_()


    def generate_initial_state(self, batch_size):
        h0_g = torch.zeros((batch_size, self.channels_chz,) + self.chrz_size).to(self.device)
        c0_g = torch.zeros((batch_size, self.channels_chz,) + self.chrz_size).to(self.device)
        u0   = torch.zeros((batch_size, self.generator_channels_u,) + self.image_size).to(self.device)
        h0_e = torch.zeros((batch_size, self.channels_chz,) + self.chrz_size).to(self.device)
        c0_e = torch.zeros((batch_size, self.channels_chz,) + self.chrz_size).to(self.device)
        return h0_g, c0_g, u0, h0_e, c0_e

    def compute_observation_representation(self, images, viewpoints):
        # specify batch_size, num_views
        batch_size = images.shape[0]
        num_views = images.shape[1]

        # (batch, views, channels, height, width) -> (batch * views, channels, height, width)
        images = images.reshape((-1,) + images.shape[2:])
        viewpoints = viewpoints.reshape((-1,) + viewpoints.shape[2:])

        # representation
        r = self.representation_network(images, viewpoints)

        # (batch * views, channels, height, width) -> (batch, views, channels, height, width)
        r = r.reshape((batch_size, num_views,) + r.shape[1:])

        # sum element-wise across views
        r = torch.sum(r, dim=1)

        return r

    def get_generation_core(self, l):
        if self.generator_share_core:
            return self.generation_cores[0]
        return self.generation_cores[l]

    def get_generation_prior(self, l):
        if self.generator_share_prior:
            return self.generation_priors[0]
        return self.generation_priors[l]

    def get_inference_core(self, l):
        if self.inference_share_core:
            return self.inference_cores[0]
        return self.inference_cores[l]

    def get_inference_posterior(self, l):
        if self.inference_share_posterior:
            return self.inference_posteriors[0]
        return self.inference_posteriors[l]

    def generate_image(self, query_viewpoints, r):
        batch_size = query_viewpoints.shape[0]
        h0_g, c0_g, u0, _, _ = self.generate_initial_state(batch_size)
        hl_g = h0_g
        cl_g = c0_g
        ul_g = u0
        for l in range(self.generator_generation_steps):
            core = self.get_generation_core(l)
            prior = self.get_generation_prior(l)
            _, _, zg_l = prior(hl_g)
            next_h_g, next_c_g, next_u_g = core(hl_g, cl_g, ul_g, zg_l, query_viewpoints, r)

            hl_g = next_h_g
            cl_g = next_c_g
            ul_g = next_u_g

        x = self.generation_observation.compute_mean_x(ul_g)
        return x


    def forward(self, images, viewpoints, num_views, query_index, pixel_var, pixel_ln_var):
        # batch_size
        batch_size = images.shape[0]

        # query
        query_images = images[:, query_index]
        query_viewpoints = viewpoints[:, query_index]

        # representation
        #if num_views > 0:
        r = self.compute_observation_representation(images[:, :num_views], viewpoints[:, :num_views])
        #else:
        #    r = torch.zeros((batch_size, self.channels_r) + self.chrz_size).to(self.device)

        # define initial state
        h0_gen, c0_gen, u_0, h0_enc, c0_enc = self.generate_initial_state(batch_size)

        # loss
        loss_kld = 0

        # state
        hl_enc = h0_enc
        cl_enc = c0_enc
        hl_gen = h0_gen
        cl_gen = c0_gen
        ul_enc = u_0

        xq = self.inference_downsampler(query_images)

        for l in range(self.generator_generation_steps):
            # prepare l-th layer
            inference_core = self.get_inference_core(l)
            inference_posterior = self.get_inference_posterior(l)
            generation_core = self.get_generation_core(l)
            generation_prior = self.get_generation_prior(l)

            # inference
            h_next_enc, c_next_enc = inference_core(hl_gen, hl_enc, cl_enc, xq, query_viewpoints, r)
            mean_z_q, ln_var_z_q, ze_l = inference_posterior(hl_enc)

            # generator
            mean_z_p, ln_var_z_p, _ = generation_prior(hl_gen)
            h_next_gen, c_next_gen, u_next_enc = generation_core(hl_gen, cl_gen, ul_enc, ze_l, query_viewpoints, r)

            # compute kl_divergence
            kld = functions.gaussian_kl_divergence(mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)
            loss_kld += torch.sum(kld)

            # update state
            hl_gen = h_next_gen
            cl_gen = c_next_gen
            ul_enc = u_next_enc
            hl_enc = h_next_enc
            cl_enc = c_next_enc

        # generate + calculate nll
        mean_x = self.generation_observation.compute_mean_x(ul_enc)
        negative_log_likelihood = functions.gaussian_negative_log_likelihood(query_images, mean_x, pixel_var, pixel_ln_var)
        loss_nll = torch.sum(negative_log_likelihood)
        # mean_square?

        # loss
        loss_nll /= batch_size
        loss_kld /= batch_size

        return loss_nll, loss_kld, mean_x
        
