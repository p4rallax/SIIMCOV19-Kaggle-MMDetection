import dataset
import torch
import yaml 
import mmcv
import mmdet
import os.path as osp
import numpy as np
import pandas as pd
from glob import glob
from config import config

import json
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, StratifiedKFold
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
print(mmdet.__version__)



###################### TRANSFORMS ###############################
train_transforms = A.Compose([
    A.OneOf([
        A.RandomBrightness(limit=.2, p=1), 
        A.RandomContrast(limit=.2, p=1), 
        A.RandomGamma(p=1)
    ], p=.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MedianBlur(blur_limit=3, p=1)
    ], p=.25),
    A.OneOf([
        A.GaussNoise(0.002, p=.5),
        A.IAAAffine(p=.5),
    ], p=.25),
    A.VerticalFlip(p=.5),
    A.HorizontalFlip(p=.5),
    A.Transpose(p=.25),
    A.RandomRotate90(p=.25),
    A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, p=.25),
    A.ShiftScaleRotate(p=.5)
])

#################################################################


########################### MMDet config setup ##################
cfg = Config.fromfile(f'mmdetection/configs/retinanet/{config.PARAMS["config"]}')
cfg.load_from = f'{config.CHKP_PATH}/{config.PARAMS["checkpoint"]}'
cfg.model.bbox_head.num_classes = 1
cfg.dump(f'{config.MDLS_PATH}/init_config.py')
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    ########################################
    # Note that this key is part of bbox_params. 
    # Their difference is format='pascal_voc' means [x1, y1, x2, y2] style box encoding, 
    # while format='coco' means [x, y, w, h].
    dict(
        type='Albu',
        transforms=train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'},
        update_pad_shape=False,
        skip_img_without_anno=True),
    #########################################
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
cfg.dataset_type = 'SIIMDataset'
# cfg.data_root = f'{config.IMGS_PATH}'
cfg.data.test.type = 'SIIMDataset'
# cfg.data.test.data_root = config.IMGS_PATH
cfg.data.test.ann_file = f'{config.MDLS_PATH}/train.txt'
cfg.data.test.img_prefix = ''
cfg.data.train.type = 'SIIMDataset'
# cfg.data.train.data_root = f'{config.IMGS_PATH}'
cfg.data.train.ann_file = f'{config.MDLS_PATH}/train.txt'
cfg.data.train.img_prefix = ''
cfg.data.val.type = 'SIIMDataset'
# cfg.data.val.data_root = f'{config.IMGS_PATH}'
cfg.data.val.ann_file = f'{config.MDLS_PATH}/val.txt'
cfg.data.val.img_prefix = ''
cfg.work_dir = config.MDLS_PATH
cfg.optimizer.lr = .02 / (8 * 16 / config.PARAMS['batch_size'])
cfg.log_config.interval = 128
cfg.runner.max_epochs = config.PARAMS['epochs']
cfg.checkpoint_config.interval = 1
cfg.evaluation = dict(
    interval=1, 
    start=2,
    metric='mAP', 
    save_best='mAP')
cfg.seed = config.PARAMS['seed']
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.data.samples_per_gpu = config.PARAMS['batch_size']
cfg.data.workers_per_gpu = 2
cfg.workflow = [('train', 1)]
cfg.dump(f'{config.MDLS_PATH}/train_config.py')
print(f'Config:\n{cfg.pretty_text}')
#################################################################


########################### TRAINING ############################
datasets = [build_dataset(cfg.data.train)]
if len(cfg.workflow) == 2:
    datasets.append(build_dataset(cfg.data.val))
model = build_detector(
    cfg.model, 
    train_cfg=cfg.get('train_cfg'), 
    test_cfg=cfg.get('test_cfg')
)
model.CLASSES = datasets[0].CLASSES
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


#################################################################