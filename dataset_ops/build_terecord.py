import os, cv2, codecs, json

import numpy as np
import tensorflow as tf

from PIL import Image
from random import shuffle


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(file_path, dataset_dir):
    ''' Covert image dataset to tfrecord. '''
    writer = tf.python_io.TFRecordWriter(file_path)

    with open(dataset_dir+'name.txt', 'r+') as f:
      for name in f.readlines():
        name = name.strip('\n')
        img = cv2.imread(dataset_dir+'img/'+name+'.jpg', 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        boxes_flie = codecs.open(
            dataset_dir+'result/'+name+'.json', 'r', encoding='utf-8').read() 
        img_boxes = json.loads(boxes_flie)

        img_raw = img.tobytes()
        bbox = list(img_boxes.values())
        bbox_num = len(bbox)
        if bbox_num > 18:
            raise ValueError('the num of bbox out of range!')
        else:
            bbox = bbox + (18 - bbox_num)*[[0.0, 0.0, 0.0, 0.0]]
        bbox = np.array(bbox) / 256
        img_name = name.encode()
        bbox_raw = bbox.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw),
            'bbox_raw': _bytes_feature(bbox_raw),
            'bbox_num': _int64_feature(bbox_num),
            'image_name': _bytes_feature(img_name)
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    dataset_dir = '/media/jun/data/capdataset/detect/train_256/'
    file_path = dataset_dir + 'train.tfrecords'
    create_tfrecord(file_path, dataset_dir)