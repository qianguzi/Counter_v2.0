import os, cv2, codecs, json

import numpy as np
import tensorflow as tf

from dataset_util import *

def build_train_dataset(imread_dir, imwrite_dir=None, save_resultimg=True):
    """Build crop train dataset for dectection."""
    if imwrite_dir:
        mkdir(imwrite_dir)
        img_dir = imwrite_dir + 'img/'
        json_dir = imwrite_dir + 'result/'
        resultimg_dir = imwrite_dir + 'resultimg/'
        mkdir([img_dir, json_dir, resultimg_dir])
    with open(imread_dir+'name.txt', 'r+') as f:
      for name in f.readlines():
        name = name.strip('\n')
        img = cv2.imread(imread_dir+'img/'+name+'.jpg', 0)
        boxes_flie = codecs.open(imread_dir+'result/'+name+'.json', 'r', encoding='utf-8').read() 
        img_boxes = json.loads(boxes_flie)
        #split_imgs, clip_select = img_random_split(img, 25, half_split_img_size=128)
        split_imgs, clip_select = img_grid_split(img, 4, 3, half_split_img_size=128)
        for n, (split_img, clip_box) in enumerate(zip(split_imgs, clip_select)):
            split_img_boxes = {}
            for box_name, box in img_boxes.items():
                prop, split_img_box = insection(box, clip_box)
                if prop >= 0.7:
                    split_img_box = split_img_box.tolist()
                    split_img_boxes[box_name] = split_img_box
            if imwrite_dir:
                json.dump(split_img_boxes,
                    codecs.open(json_dir+name+str(n)+'.json', 'w', encoding='utf-8'),
                    separators=(',', ':'), indent=4)
                cv2.imwrite(img_dir + name + str(n) + '.jpg', split_img)
                if save_resultimg:
                    resultimg = draw_bbox(split_img, split_img_boxes)
                    cv2.imwrite(resultimg_dir + name + str(n) + '.jpg', resultimg)

if __name__ == '__main__':
    imread_dir = '/media/jun/data/capdataset/detection/train_800x600/'
    imwrite_dir = '/media/jun/data/capdataset/detect/train_256/'
    build_train_dataset(imread_dir, imwrite_dir)