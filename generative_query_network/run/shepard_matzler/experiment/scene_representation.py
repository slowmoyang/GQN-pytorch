import argparse
import torch
import torchvision
import os
import math
import sys
import csv
import random
import numpy as np
import PIL
import tensorboardX as tbx

sys.path.append(os.path.join("..", "..", ".."))
import gqn

sys.path.append(os.path.join(".."))
from model import GQN


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

    # prepare images
    raw_observed_images = np.zeros(screen_size + (3, ), dtype="uint32")
    observed_image = torch.from_numpy(np.zeros((1, 3) + screen_size, dtype="float32")).to(device)
    observed_viewpoint = torch.from_numpy(np.zeros((1, 7), dtype="float32")).to(device)
    renderer = gqn.three.Renderer(screen_size[0], screen_size[1])

    features = []
    label_imgs = []
    label_meta = []
    with torch.no_grad():
        for scenenum in range(10):
            scene, _ = gqn.environment.shepard_metzler.build_scene(num_blocks=random.choice([x for x in range(7, 8)]))
            renderer.set_scene(scene)

            for viewnum in range(5):
                # prepare renderer
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
                observed_viewpoint[0] = torch.from_numpy(np.array((eye[0], eye[1], eye[2], 
                                                        math.cos(yaw), math.sin(yaw), 
                                                        math.cos(pitch), math.sin(pitch)), dtype="float32")).to(device)

                # representation network
                tmp_r = model.compute_observation_representation(torch.unsqueeze(observed_image,0), torch.unsqueeze(observed_viewpoint,0))
                
                features.append(tmp_r.view(-1))
                label_imgs.append(observed_image[0].clone())
                label_meta.append(str(scenenum))

    features = torch.stack(features)
    label_imgs = (torch.stack(label_imgs)+1.0)/2.0

    # tensorboard
    writer = tbx.SummaryWriter()
    writer.add_embedding(features, metadata=label_meta, label_img=label_imgs)
    writer.close()


"""
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((50, 50)),
    torchvision.transforms.ToTensor(),
])

features = torch.zeros(0)
labels = []
label_imgs = torch.zeros(0)
with open('features.csv') as f_csv:
    for data in csv.reader(f_csv):
        feature = torch.Tensor(np.array(data[:20]).astype(float))
        features = torch.cat((features, feature))
        label = data[20]
        labels.append(label)
        label_img = transform(PIL.Image.open(data[21]).convert('RGB'))
        label_imgs = torch.cat((label_imgs, label_img))

features = features.view(10000, 20)
label_imgs = label_imgs.view(10000, 3, 50, 50)

writer = tbx.SummaryWriter()
writer.add_embedding(features, metadata=labels, label_img=label_imgs)
writer.close()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_0_loss_11128381.659.pth")
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_100_loss_9718931.577.pth")
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_200_loss_7992014.418.pth")
    #parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_300_loss_5688351.597.pth")
    parser.add_argument("--snapshot-path", "-snapshot", type=str, default="../snapshot/itr_1600_loss_3930634.461.pth")
    parser.add_argument("--gpu-enable", "-gpu", type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    main()