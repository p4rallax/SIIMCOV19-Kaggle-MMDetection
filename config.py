import random
import json
import os
import numpy as np

class config:
    VER = 'v1'
    DEBUG =  True
    PARAMS = {
        'version': VER,
        'folds': 4,
        'val_fold': 0,
        'img_size': 512,
        'batch_size': 1,
        'epochs': 8,
        'seed': 2021,
        'iou_th': .5,
        'th': .5,
        ### vfnet_r50 ###
        'config': 'retinanet_r50_fpn_2x_coco.py',
        'checkpoint': 'retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth',
        'comments': ''
    }
    DATA_PATH = 'data'
    IMGS_PATH = 'data/train'
    CHKP_PATH = 'checkpoints'
    MDLS_PATH = f'models/mmdetection_model_{VER}'
if not os.path.exists(config.MDLS_PATH):
    os.mkdir(config.MDLS_PATH)
with open(f'{config.MDLS_PATH}/params.json', 'w') as file:
    json.dump(config.PARAMS, file)
    
def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_all(config.PARAMS['seed'])