import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from PIL import Image

import ffmpeg


def create_regressor(model_name, num_params):
    models_mapping = {
                'resnet50': lambda x : models.resnet50(pretrained=True),
                'resnet18': lambda x : models.resnet18(pretrained=True),
                'efficientnet_b0': lambda x : models.efficientnet_b0(pretrained=True),
                'efficientnet_b3': lambda x : models.efficientnet_b3(pretrained=True),
                'mobilenet_v3_large': lambda x : models.mobilenet_v3_large(pretrained=True),
                'mobilenet_v3_small': lambda x : models.mobilenet_v3_small(pretrained=True),
                'mobilenet_v2': lambda x: models.mobilenet_v2(pretrained=True),
                 }
    model = models_mapping[model_name](True)
    if model_name == 'resnet50':
        model.fc = nn.Linear(in_features=2048, out_features=num_params, bias=True)
    elif model_name == 'resnet18':
        model.fc = nn.Linear(in_features=512, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v3_large':
        model.classifier[3] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v3_small':
        model.classifier[3] = nn.Linear(in_features=1024, out_features=num_params, bias=True)    
    elif model_name == 'efficientnet_b3':
        model.classifier[1] = nn.Linear(in_features=1536, out_features=num_params, bias=True)
    elif model_name == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    elif model_name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_params, bias=True)
    else:
        model.fc = nn.Linear(in_features=10, out_features=num_params, bias=True)
    return model


def process_black_shape(shape_img):
    black_mask = shape_img == 0.0
    shape_img[black_mask] = 1.0
    return shape_img


def prepare_input_data(data_dict):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    v_ = v_.cuda()
                    v[k_] = v_.view(-1, *v_.shape[2:])
                data_dict[k] = v
            else:
                v = v.cuda()
                data_dict[k] = v.view(-1, *v.shape[2:])

        return data_dict


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    if image.shape[0] == 3:
        image = image.transpose(1,2,0)[:,:,[2,1,0]]
    else:
        image = image.transpose(1,2,0)
    return image.astype(np.uint8).copy()


def read_video(video_path, up_limit=None, resize=None):
    """
    :param video_path:
    :param up_limit:
    :param resize:
    :return: PIL list
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if resize is not None:
                frame = frame.resize((resize, resize))
            frames.append(frame)
        else:
            break
        cnt += 1
        if up_limit is not None and cnt >= up_limit:
            break
    cap.release()
    return frames


def read_video_data(video_path):
    print(video_path)
    video_probe_result = ffmpeg.probe(video_path)
    fps = eval(video_probe_result['streams'][0]['r_frame_rate'])
    print(fps)

    audio_probe_result = ffmpeg.probe(video_path, select_streams='a')

    # If p['streams'] is not empty, clip has an audio stream
    if audio_probe_result['streams']:
        stream = ffmpeg.input(video_path).audio
    else:
        stream = None

    return stream, fps