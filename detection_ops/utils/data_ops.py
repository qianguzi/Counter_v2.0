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
        if bbox_num > 15:
            raise ValueError('the num of bbox out of range!')
        else:
            bbox = bbox + (15 - bbox_num)*[[0.0, 0.0, 0.0, 0.0]]
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
    img = tf.reshape(img, [320, 320, 3])

    bbox = tf.decode_raw(features['bbox_raw'], tf.float64)
    bbox = tf.reshape(bbox, [15, 4])
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

def _test():
    dataset_dir = '/media/jun/data/capdataset/detect/test/'
    filename = dataset_dir + 'test.tfrecords'
    _create_tfrecord(filename, dataset_dir)
    sess = tf.Session()
    img_batch, bbox_batch, num_batch, name_batch = get_batch(filename, 2)
    print(img_batch, bbox_batch, num_batch, name_batch)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    sess.run(init_op)    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for clip in range(10):
        img, bbox, num, name= sess.run([img_batch, bbox_batch, num_batch, name_batch])
        #print(img, bbox, num, name)
        
    coord.request_stop()
    coord.join(threads)
    sess.close()

def area(box):
    if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
        return 0
    else:
        return (box[2] - box[0]) * (box[3] - box[1])

def insection(box_small, box_big):
  """return: 比例， 坐标"""
  min_cor = np.maximum(box_small[:2], box_big[:2]) - box_big[:2]
  max_cor = np.minimum(box_small[2:], box_big[2:]) - box_big[:2]
  draw_box = np.concatenate((min_cor, max_cor))
  pro = area(draw_box) / area(box_small)
  return pro, draw_box


def _main():
  imread_dir = '/media/jun/data/capdataset/detection/test_try/'
  imwrite_dir = '/media/jun/data/capdataset/detect/'
  file_path = '/media/jun/data/capdataset/detection/result_test.json'
  new_file = '/media/jun/data/capdataset/detect/result_test.json'

  imgs = {clip[:-4]: cv2.imread(imread_dir+clip, 0) for clip in os.listdir(imread_dir)}
  boxes_flie = codecs.open(file_path, 'r', encoding='utf-8').read()
  img_boxes = json.loads(boxes_flie)

  newimg_boxes = {}
  for img_name in imgs:
    img = imgs[img_name]
    h, w = img.shape
    x = np.arange(160, w - 160)
    y = np.arange(160, h - 160)
    centers = np.meshgrid(x, y)
    centers = np.stack(centers, axis=2)
    centers = np.reshape(centers, [-1, 2])
    random_indices = np.random.randint(0, centers.shape[0], 25)
    centers_select = centers[random_indices]
    clip_select = np.concatenate((centers_select - 160, centers_select + 160), 1)

    for n, clip in enumerate(clip_select):
      clip_box = {}
      new_img = img[clip[1]:clip[3], clip[0]:clip[2]]
      color_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

      for box_name, box in img_boxes[img_name].items():
        pro, draw_box = insection(box, clip)
        if pro >= 0.7:
          cv2.rectangle(color_img,
                        (draw_box[0], draw_box[1]),
                        (draw_box[2], draw_box[3]),
                        (0, 255, 0), 2)
          cv2.putText(color_img, box_name,
                      (draw_box[0], draw_box[3]),
                      cv2.FONT_HERSHEY_COMPLEX,
                      0.5, (0, 0, 255), 1)
          new_box = (draw_box / 320).tolist()
          clip_box[box_name] = new_box
      newimg_boxes[img_name + str(n+1)] = clip_box
      cv2.imwrite(imwrite_dir + 'test/' + img_name + str(n+1) +'.jpg', new_img)
      cv2.imwrite(imwrite_dir + 'result_test/' + img_name + str(n+1) +'.jpg', color_img)
  json.dump(newimg_boxes, codecs.open(new_file, 'w', encoding='utf-8'),
                                      separators=(',', ':'), indent=4)

if __name__ == '__main__':
    _test()