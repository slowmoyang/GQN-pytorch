import argparse
import sys
import os
import random
import numpy as np
import math
import cv2

import torch
import torch.nn as nn

sys.path.append("generative_query_network")
sys.path.append(os.path.join("..", ".."))
import gqn

from model import GQN
from optimizer import AnnealingStepLR


def make_uint8(array):
    return np.uint8(np.clip((array.transpose(1, 2, 0) + 1) * 0.5 * 255, 0, 255))


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr[0]


def main():
    # Device configuration
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu_enable) else 'cpu')

    # seed
    torch.manual_seed(args.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)
    
    # Load Data
    dataset = gqn.data.Dataset(args.dataset_path)

    # define model
    model = GQN(gpu_enable=args.gpu_enable).to(device)
    model.init_weights() # initialization by HeNorm (Kaiming_normal(0.1))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0*1e-4, betas=(0.9, 0.99), eps=1e-8)
    scheduler = AnnealingStepLR(optimizer, mu_i=5.0*1e-4, mu_f=5.0*1e-5, n=1.6*1e6)

    # visualization
    if args.with_visualization:
        figure = gqn.imgplot.figure()
        axis1 = gqn.imgplot.image()
        axis2 = gqn.imgplot.image()
        axis3 = gqn.imgplot.image()
        figure.add(axis1, 0, 0, 1 / 3, 1)
        figure.add(axis2, 1 / 3, 0, 1 / 3, 1)
        figure.add(axis3, 2 / 3, 0, 1 / 3, 1)
        plot = gqn.imgplot.window(
            figure, (500 * 3, 500),
            "Query image / Reconstructed image / Generated image")
        plot.show()
    
    # define sigma_t
    sigma_t = model.pixel_sigma_i

    print("start training")
    model.train()
    current_training_step = 0
    for iteration in range(args.training_iterations):
        mean_kld = 0
        mean_nll = 0
        total_batch = 0

        for subset_index, subset in enumerate(dataset):
            iterator = gqn.data.Iterator(subset, batch_size=args.batch_size)

            for batch_index, data_indices in enumerate(iterator):
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # (batch, views, height, width, channels) ->  (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3))

                # ToTensor
                images = torch.from_numpy(images).to(device)
                viewpoints = torch.from_numpy(viewpoints).to(device)

                # information
                total_views = images.shape[1]
                batch_size = images.shape[0]

                # prepare pixel_var/pixel_ln_var from sigma_t
                pixel_var = torch.full((batch_size, 3,) + model.image_size, sigma_t**2).to(device)
                pixel_ln_var = torch.full((batch_size, 3,) + model.image_size, math.log(sigma_t**2)).to(device)

                # sample number of views and create query_set
                num_views = random.choice(range(total_views-1)) + 1  # num_views should be chosen from [1, 15]
                query_index = random.choice(range(total_views)) #num_views # to avoid duplication
                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # GQN network
                loss_nll, loss_kld, mean_x = model(images, viewpoints, num_views, query_index, pixel_var, pixel_ln_var)
                loss = loss_nll + loss_kld

                # optimizer update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # visualization
                if args.with_visualization and plot.closed() is False:
                    axis1.update(make_uint8(query_images.data.cpu().numpy()[0]))
                    axis2.update(make_uint8(mean_x.data.cpu().numpy()[0]))
                    # generate_x
                    with torch.no_grad():
                        # representation
                        #if num_views > 0:
                        r = model.compute_observation_representation(images[:, :num_views], viewpoints[:, :num_views])
                        #else:
                        #    r = torch.zeros((args.batch_size, model.channels_r) + model.chrz_size).to(device)
                        # generator
                        generated_x = model.generate_image(query_viewpoints[None, 0], r[None, 0])
                        axis3.update(make_uint8(generated_x.data.cpu().numpy()[0]))
                elif batch_index % 5 == 0 and args.with_visualization == False:
                    source    = make_uint8(query_images.data.cpu().numpy()[0])
                    inference = make_uint8(mean_x.data.cpu().numpy()[0])
                    with torch.no_grad():
                        # representation
                        #if num_views > 0:
                        r = model.compute_observation_representation(images[:, :num_views], viewpoints[:, :num_views])
                        #else:
                        #    r = torch.zeros((args.batch_size, model.channels_r) + model.chrz_size).to(device)
                        # generator
                        generated_x = model.generate_image(query_viewpoints[None, 0], r[None, 0])
                        generate  = make_uint8(generated_x.data.cpu().numpy()[0])
                    result = np.concatenate((source, inference, generate),axis=1)
                    cv2.imwrite(os.path.join(args.snapshot_path,"current_state.png"),result)


                printr("Iteration {}: Subset {} / {}: Batch {} / {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f}".
                    format(iteration + 1, subset_index + 1, len(dataset), batch_index + 1,
                           len(iterator), float(loss_nll.data),
                           float(loss_kld.data), get_learning_rate(optimizer),
                           sigma_t))

                # simg_t/pixel_var/pixel_ln_var update
                sf = model.pixel_sigma_f
                si = model.pixel_sigma_i
                sigma_t = max(sf + (si - sf) * (1.0 - current_training_step / model.pixel_n), sf)

                total_batch += 1
                current_training_step += 1
                mean_kld += float(loss_kld.item())
                mean_nll += float(loss_nll.item())

            scheduler.step()

        # save model
        if iteration % 10**2 == 0:
            state = {
                'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_epoch': scheduler.last_epoch,
                'mean_kld': mean_kld,
                'mean_nll': mean_nll,
            }
            torch.save(state, os.path.join(args.snapshot_path, "itr_{}_loss_{:.3f}.pth".format(iteration, mean_kld+mean_nll)))

        print("\033[2KIteration {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f} - step: {}".
            format(iteration + 1, mean_nll / total_batch,
                   mean_kld / total_batch, get_learning_rate(optimizer), sigma_t,
                   current_training_step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="dataset")
    parser.add_argument("--snapshot-path", type=str, default="snapshot2")
    parser.add_argument("--gpu-enable", "-gpu", type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--training-iterations", "-iter", type=int, default=2 * 10**6)
    parser.add_argument("--with-visualization", "-visualize", action="store_true", default=False)
    args = parser.parse_args()
    main()
