import argparse
import os
import sys
import math
import random
import numpy as np

import torch

sys.path.append(os.path.join("..", "..", ".."))
import gqn

sys.path.append(os.path.join(".."))
from model import GQN


def make_uint8(array):
    if (array.shape[2] == 3):
        return np.uint8(np.clip((array + 1) * 0.5 * 255, 0, 255))
    return np.uint8(
        np.clip((array.transpose(1, 2, 0) + 1) * 0.5 * 255, 0, 255))


def main():
    # Device configuration
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu_enable) else 'cpu')

    # seed
    torch.manual_seed(args.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(args.seed)

    # define model
    model = GQN(gpu_enable=args.gpu_enable).to(device)
    model.load_state_dict(torch.load(args.snapshot_path)['state_dict'])
    model.eval()

    # define screen
    screen_size = model.image_size
    camera = gqn.three.PerspectiveCamera(
        eye=(3, 1, 0),
        center=(0, 0, 0),
        up=(0, 1, 0),
        fov_rad=math.pi / 2.0,
        aspect_ratio=screen_size[0] / screen_size[1],
        z_near=0.1,
        z_far=10)

    # define window
    figure = gqn.imgplot.figure()
    axis_observation = gqn.imgplot.image()
    axis_generation = gqn.imgplot.image()
    figure.add(axis_observation, 0, 0, 0.5, 1)
    figure.add(axis_generation, 0.5, 0, 0.5, 1)
    window = gqn.imgplot.window(figure, (1600, 800), "Viewpoint")
    window.show()

    # prepare images
    raw_observed_images = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    observed_image = torch.from_numpy(np.zeros((1, 3) + screen_size, dtype="float32")).to(device)
    observed_viewpoint = torch.from_numpy(np.zeros((1, 7), dtype="float32")).to(device)
    query_viewpoint = torch.from_numpy(np.zeros((1, 7), dtype="float32")).to(device)

    with torch.no_grad():
        while True:
            if window.closed():
                exit()

            scene, _ = gqn.environment.shepard_metzler.build_scene(num_blocks=random.choice([x for x in range(7, 8)]))
            renderer.set_scene(scene)

            view_num = 15
            r = 0
            for _ in range(view_num):
                rad = random.uniform(0, math.pi * 2)
                eye = (3.0 * math.cos(rad), 0, 3.0 * math.sin(rad))
                center = (0, 0, 0)
                yaw = gqn.math.yaw(eye, center)
                pitch = gqn.math.pitch(eye, center)
                camera.look_at(
                    eye=eye,
                    center=center,
                    up=(0.0, 1.0, 0.0),
                )
                renderer.render(camera, raw_observed_images)

                # [0, 255] -> [-1, 1]
                observed_image[0] = torch.from_numpy((raw_observed_images.transpose((2, 0, 1)) / 255 - 0.5) * 2.0).to(device)
                axis_observation.update(make_uint8(observed_image.data.cpu().numpy()[0]))

                observed_viewpoint[0] = torch.from_numpy(np.array((eye[0], eye[1], eye[2], 
                                                        math.cos(yaw), math.sin(yaw), 
                                                        math.cos(pitch), math.sin(pitch)), dtype="float32")).to(device)

                # representation network
                r += model.compute_observation_representation(torch.unsqueeze(observed_image,0), torch.unsqueeze(observed_viewpoint,0))

            num_samples = 200
            for _ in range(num_samples):
                if window.closed():
                    exit()

                yaw += 0.5
                pitch += 0.1
                query_viewpoint[0] = torch.from_numpy(np.array((eye[0], eye[1], eye[2], 
                                                      math.cos(yaw), math.sin(yaw),
                                                      math.cos(pitch), math.sin(pitch)), dtype="float32")).to(device)

                generated_image = model.generate_image(query_viewpoint, r)
                axis_generation.update(make_uint8(generated_image.data.cpu().numpy()[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_500_loss_3935991.818.pth")
    parser.add_argument("--gpu-enable", "-gpu", type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    main()
