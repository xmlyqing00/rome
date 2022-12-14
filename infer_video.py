from turtle import shape
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
import torch.nn as nn
from MODNet.src.models.modnet import MODNet
from tqdm import trange, tqdm

from data_utils import calc_ffhq_alignment
from src.rome import ROME
from src.utils import args as args_utils
from src.utils.processing import process_black_shape, prepare_input_data, tensor2image, read_video, read_video_data
from src.utils.visuals import obtain_modnet_mask, mask_errosion
import ffmpeg

warnings.filterwarnings("ignore")


class Infer(object):
    def __init__(self, args):
        super(Infer, self).__init__()

        # Initialize and apply general options
        torch.manual_seed(args.random_seed)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        if args.verbose:
            print('Initialize model.')

        self.model = ROME(args).eval().to(self.device)
        self.image_size = 256
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # Load pre-trained weights
        if args.model_checkpoint:
            ckpt_loaded = torch.load(args.model_checkpoint, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(ckpt_loaded, strict=False)
        self.setup_modnet()
        self.mask_hard_threshold = 0.5

    def setup_modnet(self):
        pretrained_ckpt = self.args.modnet_path

        modnet = nn.DataParallel(MODNet(backbone_pretrained=False))

        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location='cpu'))
        self.modnet = modnet.eval().to(self.device)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                               flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    def process_source_for_input_dict(self, source_img: Image, data_transform, crop_center=False):
        data_dict = {}
        source_pose = self.fa.get_landmarks_from_image(np.asarray(source_img))[0]

        if crop_center or source_img.size[0] != source_img.size[1]:
            pose = source_pose
            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6
            source_img = source_img.crop((center[0] - size, center[1] - size, center[0] + size, center[1] + size))

        source_img = source_img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        source_img.show()
        data_dict['source_img'] = data_transform(source_img)[None]        

        pred_mask = obtain_modnet_mask(data_dict['source_img'][0].to(self.device), self.modnet, ref_size=512)[0]
        data_dict['source_mask'] = torch.from_numpy(pred_mask).float().unsqueeze(0)[None]
        lm_2d = self.fa.get_landmarks_from_image(np.asarray(source_img))[0]
        data_dict['source_keypoints'] = torch.from_numpy(lm_2d)[None]

        if (data_dict['source_mask'].shape) == 3:
            data_dict['source_mask'] = data_dict['source_mask'][..., -1]

        if args.align_source:
            transform_ffhq = calc_ffhq_alignment(lm_2d, size=self.image_size, device=self.device)
            theta = torch.FloatTensor(transform_ffhq['theta'])[None]
            grid = torch.linspace(-1, 1, self.image_size)
            v, u = torch.meshgrid(grid, grid)
            identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)
            # Align input images using theta

            eye_vector = torch.zeros(theta.shape[0], 1, 3)
            eye_vector[:, :, 2] = 1
            theta_ = torch.cat([theta, eye_vector], dim=1).float()

            # Perform 2x zoom-in compared to default theta
            scale = torch.zeros_like(theta_)
            scale[:, [0, 1], [0, 1]] = args.align_scale
            scale[:, 2, 2] = 1

            theta_ = torch.bmm(theta_, scale)[:, :2]
            align_warp = identity_grid.repeat_interleave(theta_.shape[0], dim=0)
            align_warp = align_warp.bmm(theta_.transpose(1, 2)).view(theta_.shape[0], self.image_size, self.image_size, 2)

            source_imgs = F.grid_sample(data_dict['source_img'], align_warp)
            source_masks = F.grid_sample(data_dict['source_mask'], align_warp)
        else:
            source_imgs, source_masks = data_dict['source_img'], data_dict['source_img']

        source_keypoints = torch.from_numpy(self.fa.get_landmarks_from_image(tensor2image(source_imgs[0]))[0])[None]
        output_data_dict = {
            'source_img': source_imgs,
            'source_mask': source_masks,
            'source_keypoints': (source_keypoints / (self.image_size / 2) - 1),
        }
        return output_data_dict

    def process_driver_img(self, data_dict: dict, driver_image: Image, crop_center=False):
        driver_pose = self.fa.get_landmarks_from_image(np.asarray(driver_image))[0]

        if crop_center or driver_image.size[0] != driver_image.size[1]:
            pose = driver_pose
            center = ((pose.min(0) + pose.max(0)) / 2).round().astype(int)
            size = int(max(pose[:, 0].max() - pose[:, 0].min(), pose[:, 1].max() - pose[:, 1].min()))
            center[1] -= size // 6
            driver_image = driver_image.crop((center[0] - size, center[1] - size, center[0] + size, center[1] + size))

        data_dict['target_img'] = self.img_transform(driver_image)[None]
        data_dict['target_mask'] = torch.zeros_like(data_dict['target_img'])
        landmark_input = np.asarray(driver_image)
        kp_scale = landmark_input.shape[0] // 2
        data_dict['target_keypoints'] = \
        torch.from_numpy(self.fa.get_landmarks_from_image(landmark_input)[0] / kp_scale - 1)[None]
        return data_dict


    @torch.no_grad()
    def run_example(self):

        frames_pil = read_video(args.videoPath)
        input_audio, fps = read_video_data(args.videoPath)
        video_name = os.path.basename(args.videoPath).split('.')[0]
        if args.refImgPath:
            ref_img_name = os.path.basename(args.refImgPath).split('.')[0]
            print('img path', args.refImgPath)
            ref_img_pil = Image.open(args.refImgPath)
        else:
            ref_img_pil = frames_pil[0]
            ref_img_name = video_name + '_0'

        data_dict = self.process_source_for_input_dict(ref_img_pil, self.img_transform)

        out_dir_name = f'{video_name}_ref_{ref_img_name}'
        if args.use_deca_details:
            out_dir_name += '_deca_details'
        out_dir = os.path.join(self.args.save_dir, out_dir_name)

        # save_list = ['pred_target_normal', 'pred_target_shape_img', 'pred_target_hard_mask', 'target_shape_final_posed_img', 
                        # 'target_shape_final_frontal_img', 'target_shape_parametric_frontal_img', 'pred_target_img', 'render_masked']
        save_list = ['target_shape_final_posed_img', 'mesh', 'original_mesh', 'pred_target_normal', 'pred_target_shape_img', 'pred_target_hard_mask', 'render_masked', 'target_shape_final_frontal_img', 'target_shape_parametric_frontal_img', 'pred_target_shape_displ_img']
        for save_dir in save_list:
            out_sub_dir = os.path.join(out_dir, save_dir)
            os.makedirs(out_sub_dir, exist_ok=True)
        
        concat_frames_dir = os.path.join(out_dir, 'concat_frames')
        os.makedirs(concat_frames_dir, exist_ok=True)

        for i, frame_pil in enumerate(tqdm(frames_pil)):
            data_dict = self.process_driver_img(data_dict, frame_pil)
            for k, v in data_dict.items():
                data_dict[k] = data_dict[k].to(self.device)

            out = self.model(data_dict)
            # out['source_information']['data_dict'] = data_dict

            for item in save_list:
                if 'mesh' in item:

                    out_path = os.path.join(out_dir, item, f'{i}.obj')
                    IO().save_mesh(out[item], out_path)
                else:
                    if not args.use_deca_details and item == 'pred_target_shape_displ_img':
                        continue
                    out_path = os.path.join(out_dir, item, f'{i}.jpg')
                
                    if len(out[item].shape) == 4:
                        x = out[item].squeeze(0).cpu()
                    else:
                        x = out[item].cpu()
                    
                    x = tensor2image(x)
                    # print(f'Save to {out_path}')
                    cv2.imwrite(out_path, x)
        
        for i in trange(len(frames_pil)):
            img_path = os.path.join(out_dir, 'render_masked', f'{i}.jpg')
            img1 = cv2.imread(img_path)
            img_path = os.path.join(out_dir, 'pred_target_shape_img', f'{i}.jpg')
            img2 = cv2.imread(img_path)
            img_path = os.path.join(out_dir, 'pred_target_shape_displ_img', f'{i}.jpg')
            img3 = cv2.imread(img_path)
            img_path = os.path.join(out_dir, 'pred_target_normal', f'{i}.jpg')
            img4 = cv2.imread(img_path)

            img0 = cv2.cvtColor(np.asarray(frames_pil[i]), cv2.COLOR_RGB2BGR)
            img0 = cv2.resize(img0, img1.shape[:2])

            frame = np.concatenate([img0, img1, img2, img3, img4], axis=1)

            out_path = os.path.join(concat_frames_dir, f'{i:05d}.jpg')
            cv2.imwrite(out_path, frame)

        video_path = os.path.join(out_dir, 'rendered.mp4')
        print('Render video to', video_path)

        out_frames = ffmpeg.input(os.path.join(concat_frames_dir, '*.jpg'), pattern_type='glob', framerate=fps)
        if input_audio:
            stream = ffmpeg.concat(out_frames, input_audio, v=1, a=1)
            stream.output(video_path).run(overwrite_output=True)
        else:
            out_frames.output(video_path).run(overwrite_output=True)        


def main(args):
    infer = Infer(args)
    infer.run_example()


if __name__ == "__main__":
    print('Start infer!')
    default_modnet_path = 'MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
    default_model_path = 'data/rome.pth'

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--videoPath', default='.', type=str)
    parser.add_argument('--refImgPath', type=str)
    parser.add_argument('--save_dir', default='./output', type=str)
    parser.add_argument('--save_render', default='True', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--model_checkpoint', default=default_model_path, type=str)
    parser.add_argument('--modnet_path', default=default_modnet_path, type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', default='False', type=args_utils.str2bool, choices=[True, False])
    args, _ = parser.parse_known_args()

    parser = importlib.import_module(f'src.rome').ROME.add_argparse_args(parser)

    args = parser.parse_args()
    args.deca_path = 'DECA'

    main(args)
