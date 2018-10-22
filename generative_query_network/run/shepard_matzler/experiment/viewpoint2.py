import argparse
import os
import sys
import math
import random
import numpy as np
import cv2

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
    axis1_1 = gqn.imgplot.image()
    axis1_2 = gqn.imgplot.image()
    axis1_3 = gqn.imgplot.image()
    axis1_4 = gqn.imgplot.image()
    axis1_5 = gqn.imgplot.image()
    axis1_6 = gqn.imgplot.image()
    axis1_7 = gqn.imgplot.image()
    axis1_8 = gqn.imgplot.image()
    axis1_9 = gqn.imgplot.image()
    axis2 = gqn.imgplot.image()
    axis3 = gqn.imgplot.image()
    figure.add(axis1_1,         0,     0, 1 / (3*3), 1 / 3)
    figure.add(axis1_2, 1 / (3*3),     0, 1 / (3*3), 1 / 3)
    figure.add(axis1_3, 2 / (3*3),     0, 1 / (3*3), 1 / 3)
    figure.add(axis1_4,         0, 1 / 3, 1 / (3*3), 1 / 3)
    figure.add(axis1_5, 1 / (3*3), 1 / 3, 1 / (3*3), 1 / 3)
    figure.add(axis1_6, 2 / (3*3), 1 / 3, 1 / (3*3), 1 / 3)  
    figure.add(axis1_7,         0, 2 / 3, 1 / (3*3), 1 / 3)
    figure.add(axis1_8, 1 / (3*3), 2 / 3, 1 / (3*3), 1 / 3)
    figure.add(axis1_9, 2 / (3*3), 2 / 3, 1 / (3*3), 1 / 3) 

    figure.add(axis2, 1 / 3, 0, 1 / 3, 1)
    figure.add(axis3, 2 / 3, 0, 1 / 3, 1)
    window = gqn.imgplot.window(
        figure, (300 * 3, 300),
        "Observed image / Query image / Reconstructed image")
    window.show()

    # prepare images
    raw_observed_images = np.zeros(screen_size + (3, ), dtype="uint32")
    raw_query_images = np.zeros(screen_size + (3, ), dtype="uint32")
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    query_image = torch.from_numpy(np.zeros((1, 3) + screen_size, dtype="float32")).to(device)
    observed_image = torch.from_numpy(np.zeros((1, 3) + screen_size, dtype="float32")).to(device)
    observed_viewpoint = torch.from_numpy(np.zeros((1, 7), dtype="float32")).to(device)
    query_viewpoint = torch.from_numpy(np.zeros((1, 7), dtype="float32")).to(device)

    first_flag = False
    with torch.no_grad():
        while True:
            if window.closed():
                exit()

            scene, _ = gqn.environment.shepard_metzler.build_scene(num_blocks=random.choice([x for x in range(7, 8)]))
            renderer.set_scene(scene)

            # reset
            axis1_1.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_2.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_3.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_4.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_5.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_6.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_7.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_8.update(np.zeros(screen_size + (3, ), dtype="float32"))
            axis1_9.update(np.zeros(screen_size + (3, ), dtype="float32"))

            # initialize view pose
            rad_view = 0
            rad2_view = 0
            r = torch.zeros(1,256,16,16).to(device)
            for view_num in range(1+9):
                #if view_num == 0 and first_flag:
                #    continue
                if view_num > 0:
                    rad = random.uniform(0, math.pi * 2)
                    rad2 = random.uniform(0, math.pi * 2)
                    eye = (3.0 * math.cos(rad), 3.0 * math.sin(rad2), 3.0 * math.sin(rad))
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
                    if view_num == 1:
                        axis1_1.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 2:
                        axis1_2.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 3:
                        axis1_3.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 4:
                        axis1_4.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 5:
                        axis1_5.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 6:
                        axis1_6.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 7:
                        axis1_7.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 8:
                        axis1_8.update(make_uint8(observed_image.data.cpu().numpy()[0]))
                    elif view_num == 9:
                        axis1_9.update(make_uint8(observed_image.data.cpu().numpy()[0]))

                    observed_viewpoint[0] = torch.from_numpy(np.array((eye[0], eye[1], eye[2], 
                                                            math.cos(yaw), math.sin(yaw), 
                                                            math.cos(pitch), math.sin(pitch)), dtype="float32")).to(device)

                    # representation network
                    r += model.compute_observation_representation(torch.unsqueeze(observed_image,0), torch.unsqueeze(observed_viewpoint,0))

                num_samples = 50
                if first_flag == False:
                    num_samples = 50
                for _ in range(num_samples):
                    if window.closed():
                        exit()

                    rad_view += 0.1
                    rad2_view += 0.05
                    eye = (3.0 * math.cos(rad_view), 3.0 * math.cos(rad2_view), 3.0 * math.sin(rad_view))
                    center = (0, 0, 0)
                    yaw = gqn.math.yaw(eye, center)
                    pitch = gqn.math.pitch(eye, center)
                    camera.look_at(
                        eye=eye,
                        center=center,
                        up=(0.0, 1.0, 0.0),
                    )
                    query_viewpoint[0] = torch.from_numpy(np.array((eye[0], eye[1], eye[2], 
                                                        math.cos(yaw), math.sin(yaw),
                                                        math.cos(pitch), math.sin(pitch)), dtype="float32")).to(device)

                    renderer.render(camera, raw_query_images)
                    # [0, 255] -> [-1, 1]
                    query_image[0] = torch.from_numpy((raw_query_images.transpose((2, 0, 1)) / 255 - 0.5) * 2.0).to(device)
                    axis2.update(make_uint8(query_image.data.cpu().numpy()[0]))

                    generated_image = model.generate_image(query_viewpoint, r)
                    axis3.update(make_uint8(generated_image.data.cpu().numpy()[0]))
                first_flag = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_0_loss_11128381.659.pth")
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_100_loss_9718931.577.pth")
    parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_200_loss_7992014.418.pth")
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_300_loss_5688351.597.pth")
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_1600_loss_3930634.461.pth")
    parser.add_argument("--gpu-enable", "-gpu", type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    main()
