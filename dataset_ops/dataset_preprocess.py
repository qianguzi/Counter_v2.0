import os, cv2
import codecs, json, argparse
import numpy as np

from dataset_util import draw_bbox, mkdir


def save_resultimg(img, img_boxes, img_name, resultimg_dir):
  resultimg = draw_bbox(img, img_boxes)
  cv2.imwrite(resultimg_dir+img_name+'.jpg', resultimg)


def save_dataset(img, img_boxes, img_name, img_dir, json_dir):
  cv2.imwrite(img_dir+img_name+'.jpg', img)
  json.dump(img_boxes, 
            codecs.open(json_dir+img_name+'.json', 'w', encoding='utf-8'),
            separators=(',', ':'), indent=4)


def img_resize(img, img_boxes, size):
  h, w = img.shape[:2]
  newimg = cv2.resize(img, size)
  newimg_boxes = {}
  for box_name in img_boxes:
    box = np.multiply(img_boxes[box_name],
                      [size[0]/w, size[1]/h, size[0]/w, size[1]/h])
    newimg_boxes[box_name] = box.astype(np.int).tolist()
  return newimg, newimg_boxes


def img_flip(img, img_boxes, flipcode):
  h, w = img.shape[:2]
  newimg = cv2.flip(img, flipcode)
  newimg_boxes = {}
  if flipcode == 1:
    for box_name in img_boxes:
      box = np.abs(np.subtract([w, 0, w, 0], img_boxes[box_name]))
      box = box[[2, 1, 0, 3]]
      newimg_boxes[box_name] = box.astype(np.int).tolist()
  elif flipcode == 0:
    for box_name in img_boxes:
      box = np.abs(np.subtract([0, h ,0, h], img_boxes[box_name]))
      box = box[[0, 3, 2, 1]]
      newimg_boxes[box_name] = box.astype(np.int).tolist()
  elif flipcode == -1:
    for box_name in img_boxes:
      box = np.abs(np.subtract([w, h ,w, h], img_boxes[box_name]))
      box = box[[2, 3, 0, 1]]
      newimg_boxes[box_name] = box.astype(np.int).tolist()
  return newimg, newimg_boxes


def dataset_preprocess(imread_dir,
                file_name,
                imwrite_dir=None,
                resize_img=None,
                flip_img=None,
                to_save_dataset=True,
                to_save_resultimg=False,
                to_save_original=False):
  if imwrite_dir:
    mkdir(imwrite_dir)
    if to_save_dataset:
      img_dir = imwrite_dir + 'img/'
      json_dir = imwrite_dir + 'result/'
      mkdir([img_dir, json_dir])
    if to_save_resultimg:
      resultimg_dir = imwrite_dir + 'resultimg/'
      mkdir(resultimg_dir)

  with open(imread_dir+file_name, 'r+') as f:
    for name in f.readlines():
      name = name.strip('\n')
      img = cv2.imread(imread_dir+'img/'+name+'.png', 0)
      boxes_flie = codecs.open(
        imread_dir+'result/'+name+'.json', 'r', encoding='utf-8').read()
      img_boxes = json.loads(boxes_flie)

      if resize_img:
        newimg, newimg_boxes = img_resize(img, img_boxes, resize_img)
        if imwrite_dir:
          img_name = name + '_resize'
          if to_save_dataset:
            save_dataset(newimg, newimg_boxes, img_name, img_dir, json_dir)
          if to_save_resultimg:
            save_resultimg(newimg, newimg_boxes, img_name, resultimg_dir)
      if flip_img:
        for i in flip_img:
          newimg, newimg_boxes = img_flip(img, img_boxes, i)
          if imwrite_dir:
            img_name = name + '_flip' + str(i)
            if to_save_dataset:
              save_dataset(newimg, newimg_boxes, img_name, img_dir, json_dir)
            if to_save_resultimg:
              save_resultimg(newimg, newimg_boxes, img_name, resultimg_dir)
      if to_save_original:
        if imwrite_dir:
          if to_save_dataset:
            save_dataset(img, img_boxes, name, img_dir, json_dir)
          if to_save_resultimg:
            save_resultimg(img, img_boxes, name, resultimg_dir)

if __name__ == '__main__':
  imread_dir = '/media/jun/data/capdataset/detection/test/'
  imwrite_dir = '/media/jun/data/capdataset/detection/test_800x600/'
  dataset_preprocess(imread_dir, 'name.txt', imwrite_dir, resize_img=(600, 800),
                    to_save_original=False)