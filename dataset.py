import yaml 
import mmcv
import mmdet
import os.path as osp
import numpy as np
import pandas as pd
from glob import glob
from config import config
import json
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

####################### PARAMS AND CONFIG #############################

#######################################################################




####################### PREPROCESS ####################################
train_df = pd.read_csv(f'{config.DATA_PATH}/meta.csv')
train_df = train_df[train_df.split == 'train']
del train_df['split']
if config.DEBUG:
    train_df = train_df.loc[:100]
df_train_img = pd.read_csv(f'{config.DATA_PATH}/train_image_level.csv')
df_train_sty = pd.read_csv(f'{config.DATA_PATH}/train_study_level.csv')

train_df['id'] = train_df['image_id'].apply(lambda x: ''.join([x.split('/')[-1], '_image']))
#df_train_sty['StudyInstanceUID'] = df_train_sty['id'].apply(lambda x: x.replace('_study', ''))
#del df_train_sty['id']
#df_train_img = df_train_img.merge(df_train_sty, on='StudyInstanceUID')
train_df = df_train_img.merge(train_df, on='id')
train_df['img'] = train_df['image_id'] + '.png'
train_df['labs']  = train_df['label'].apply(lambda x: x.split(' ')[0])
train_df['opacity'] = train_df['labs'].apply(lambda x: int(x=='opacity'))
train_df['none'] = train_df['labs'].apply(lambda x: int(x=='none'))
del train_df['labs']
train_df['target'] = 'none'
train_df.loc[train_df['opacity']==1, 'target'] = 'opacity'
print(train_df.shape)
print(train_df.head())


train_df = train_df[~train_df.boxes.isnull()] 
train_df.reset_index(inplace=True)
classes = [
    'opacity', 
    'none'
]
print('classes:\n', classes,
      '\nclasses labels:\n', np.unique(train_df[classes].values, axis=0))
#######################################################################



###################### KFOLD STRATIFIED SPLIT #########################
skf  = StratifiedKFold(n_splits=config.PARAMS['folds'])
train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y=train_df.target)):
    train_df.loc[val_idx, 'fold'] = fold

split = config.PARAMS['val_fold']
with open(f'{config.MDLS_PATH}/train.txt', 'w') as file:
    tr_ids = list(train_df[train_df['fold'] != split].img.unique())
    print('train:', len(tr_ids))
    file.write('\n'.join(tr_ids))
with open(f'{config.MDLS_PATH}/val.txt', 'w') as file:
    val_ids = list(train_df[train_df['fold'] == split].img.unique())
    print('val:', len(val_ids))
    file.write('\n'.join(val_ids))    
#######################################################################




###################### DATASET CLASS MODULE ###########################

@DATASETS.register_module()
class SIIMDataset(CustomDataset):
    CLASSES = ['opacity' , ]
    ANN_DF = train_df.copy()
    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(self.ann_file)
        data_infos = []
        for img_id in image_list:
            img_anns = self.ANN_DF[self.ANN_DF.img == img_id]
            filename = f'{self.img_prefix}/{img_anns["img"].values[0]}'
            data_info = dict(
                filename=filename, 
                width=config.PARAMS['img_size'], 
                height=config.PARAMS['img_size']
            )
            ratio_x = config.PARAMS['img_size'] / img_anns['dim1'].values[0]
            ratio_y = config.PARAMS['img_size'] / img_anns['dim0'].values[0]
            boxes = img_anns['boxes'].values[0]
            boxes = json.loads(boxes.replace('\'', '\"'))
            gt_bboxes = [
                [int(box['x'] * ratio_x), 
                 int(box['y'] * ratio_y), 
                 int((box['x'] + box['width']) * ratio_x), 
                 int((box['y'] + box['height']) * ratio_y)]
                for box in boxes]
            img_labels = img_anns[self.CLASSES].values[0]
            gt_labels = [1] * len(boxes)
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels),
            )
            data_info.update(ann=data_anno)
            data_infos.append(data_info)
        return data_infos

