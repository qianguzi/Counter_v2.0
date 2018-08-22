import os, cv2
import codecs, json

import numpy as np
import tensorflow as tf

from PIL import Image
from random import shuffle

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

def _create_tfrecord(filename, dataset_dir):
    ''' Covert Image dataset to tfrecord. '''
    boxes_flie = codecs.open(dataset_dir + 'bbox.json',
                             'r', encoding='utf-8').read()
    img_boxes = json.loads(boxes_flie)

    img_dir = dataset_dir + 'img/'
    writer = tf.python_io.TFRecordWriter(filename)

    for img_name in os.listdir(img_dir):
        img_path = img_dir + img_name
        img = Image.open(img_path)
        img = img.convert("RGB")
        img_raw = img.tobytes()
        img_name = img_name[:-4]
        bbox = list(img_boxes[img_name].values())
        bbox_num = len(bbox)
        if bbox_num > 20:
            raise ValueError('the num of bbox out of range!')
        else:
            bbox = bbox + (20 - bbox_num)*[[0.0, 0.0, 0.0, 0.0]]
        bbox = np.array(bbox)
        #print(bbox)
        img_name = img_name.encode()
        bbox_raw = bbox.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw),
            'bbox_raw': _bytes_feature(bbox_raw),
            'bbox_num': _int64_feature(bbox_num),
            'image_name': _bytes_feature(img_name)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord(filename, shuffle=True):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(
        [filename], shuffle=shuffle)
    # 从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'image_raw': tf.FixedLenFeature([], tf.string),
                  'bbox_raw': tf.FixedLenFeature([], tf.string),
                  'bbox_num': tf.FixedLenFeature([], tf.int64),
                  'image_name': tf.FixedLenFeature([], tf.string)})
    # 将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) / 127.5 - 1
    bbox = tf.decode_raw(features['bbox_raw'], tf.float64)
    bbox = tf.reshape(bbox, [20, 4])
    bbox = tf.cast(bbox, tf.float32)
    num = tf.cast(features['bbox_num'], tf.int32)
    name = tf.cast(features['image_name'], tf.string)

    return img, bbox, num, name


def get_batch(filename, batch_size, shuffle=True):
    '''Get batch.'''

    img, bbox, num, name = read_tfrecord(filename, shuffle)
    capacity = 5 * batch_size

    img_batch, bbox_batch, num_batch, name_batch = tf.train.batch([img, bbox, num, name], batch_size,
                                                            capacity=capacity, num_threads=4)
    return img_batch, bbox_batch, num_batch, name_batch

def area(box):
    if (box[2] - box[0]) and (box[3] - box[1]) <= 0:
        return 0
    else:
        return (box[2] - box[0]) * (box[3] - box[1])

def insection(box_small, box):
  """return: 比例， 坐标"""
  min_cor = np.maximum(box_small[:2], box[:2]) - box[:2]
  max_cor = np.minimum(box_small[2:], box[2:]) - box[:2]
  new_box = np.concatenate((min_cor, max_cor))
  pro = area(new_box) / area(box_small)
  return pro, new_box / 356


def _test():
    dataset_dir = '/media/jun/data/capdataset/detect/train/'
    filename = dataset_dir + 'train.tfrecords'
    _create_tfrecord(filename, dataset_dir)
    sess = tf.Session()
    img_batch, bbox_batch, num_batch, name_batch = get_batch(filename, 2)
    print(img_batch, bbox_batch, num_batch, name_batch)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    sess.run(init_op)    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        img, bbox, num, name= sess.run([img_batch, bbox_batch, num_batch, name_batch])
        #print(img, bbox, num, name)
        
    coord.request_stop()
    coord.join(threads)
    sess.close()


def _main():
  imread_dir = '/media/jun/data/capdataset/detection/train/'
  imwrite_dir = '/media/jun/data/capdataset/detect/'
  file_path = '/media/jun/data/capdataset/detection/result_train.json'
  new_file = '/media/jun/data/capdataset/detect/result_train.json'

  imgs = {i[:-4]: cv2.imread(imread_dir+i, 0) for i in os.listdir(imread_dir)}
  boxes_flie = codecs.open(file_path, 'r', encoding='utf-8').read()
  img_boxes = json.loads(boxes_flie)

  newimg_boxes = {}
  for img_name in imgs:
    img = imgs[img_name]
    h, w = img.shape
    centers = np.meshgrid([w/3, 2*w/3], [h/4, h/2, 3*h/4])
    centers = np.stack(centers, axis=2)
    centers = np.reshape(centers, [-1, 2]).astype(int)
    clip_boxes = np.concatenate((centers - 178, centers + 178), 1)
    for n, i in enumerate(clip_boxes):
      clip_box = {}
      new_img = img[i[1]:i[3], i[0]:i[2]]
      new_img = cv2.resize(new_img, (256, 256))
      color_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

      for box_name, box in img_boxes[img_name].items():
        pro, new_box = insection(box, i)
        if pro >= 0.5:
          draw_box = (new_box * 256).astype(int)
          cv2.rectangle(color_img,
                        (draw_box[0], draw_box[1]),
                        (draw_box[2], draw_box[3]),
                        (0, 255, 0), 2)
          cv2.putText(color_img, box_name,
                      (draw_box[0], draw_box[3]),
                      cv2.FONT_HERSHEY_COMPLEX,
                      0.5, (0, 0, 255), 1)
          new_box = new_box.tolist()
          clip_box[box_name] = new_box
      newimg_boxes[img_name + str(n+1)] = clip_box
      cv2.imwrite(imwrite_dir + 'train/' + img_name + str(n+1) +'.jpg', new_img)
      cv2.imwrite(imwrite_dir + 'result_train/' + img_name + str(n+1) +'.jpg', color_img)
  json.dump(newimg_boxes, codecs.open(new_file, 'w', encoding='utf-8'),
                                      separators=(',', ':'), indent=4)

if __name__ == '__main__':
    _test()