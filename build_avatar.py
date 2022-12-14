import yaml
import warnings

import os
import importlib
import argparse
from glob import glob
import cv2

import face_alignment
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from pytorch3d.io import save_obj, IO
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn as nn
from MODNet.src.models.modnet import MODNet
from tqdm import trange, tqdm

from data_utils import calc_ffhq_alignment
from src.rome import ROME
from src.utils import args as args_utils
from src.utils.processing import process_black_shape, prepare_input_data, tensor2image, read_video, read_video_data
from src.utils.visuals import obtain_modnet_mask, mask_errosion
from src.utils.types import DictObj
from models.avatar import Avatar
import ffmpeg


def setup_models(cfg, device):

    avatar = Avatar(cfg, device)
    avatar_ckpt = torch.load(cfg.paths.avatar, map_location='cpu')
    missing_keys, unexpected_keys = avatar.load_state_dict(avatar_ckpt, strict=False)
    avatar = avatar.to(device)

    modnet = nn.DataParallel(MODNet(backbone_pretrained=False))
    modnet.load_state_dict(torch.load(cfg.paths.modnet, map_location='cpu'))
    modnet = modnet.to(device)
    modnet.eval()

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        flip_input=False, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return avatar, modnet, fa


def preprocessing(
    input_img: Image, 
    fa: face_alignment.FaceAlignment,
    modnet: MODNet,
    cfg: DictObj,
    device
):
    
    pose = fa.get_landmarks_from_image(input_img)[0]

    center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
    size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
    center[1] -= size // 5
    crop_region = [
        max(0, center[0]-size), 
        min(input_img.shape[0], center[0]+size), 
        max(0, center[1]-size), 
        min(input_img.shape[1], center[1]+size)
    ]
    cropped_img = input_img[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3]]
    cropped_img = cv2.resize(cropped_img, (cfg.img_size, cfg.img_size))
    
    input_data = TF.to_tensor(cropped_img).unsqueeze(0).to(device)

    mask = obtain_modnet_mask(input_data[0], modnet, ref_size=512)  # np.array (1, 512, 512)
    mask = torch.from_numpy(mask).float().unsqueeze(0).to(device)
    lm_2d = fa.get_landmarks_from_image(cropped_img)[0]
    lm_2d = torch.from_numpy(lm_2d) / cfg.img_size / 2 - 1

    return input_data, mask, lm_2d, crop_region


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Build Avatar')
    parser.add_argument('--cfg', default='', help='Path to the config yaml file.')
    parser.add_argument('--img', default='', help='Path to the portrait image.')
    args = parser.parse_args()
    print(args)
    
    with open(args.cfg, 'r') as f:
        cfg = DictObj(yaml.safe_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    avatar, modnet, fa = setup_models(cfg, device)

    input_img = cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', input_img)
    # cv2.waitKey()
    input_data, mask, lm_2d, crop_region = preprocessing(input_img, fa, modnet, cfg, device)

    avatar_dict = avatar(input_data, mask)

    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    obj_path = os.path.join(cfg.paths.out_dir, os.path.basename(args.img)[:-4] + '.obj')
    avatar.save_obj(obj_path, avatar_dict)
