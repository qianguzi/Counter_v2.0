import os, cv2
import numpy as np


def mkdir(path):
  if type(path) == str:
    if not os.path.exists(path):
      os.mkdir(path)
      print('{} is created.'.format(path))
  elif type(path) == list:
    for i in path:
      if not os.path.exists(i):
        os.mkdir(i)
        print('{} is created.'.format(i))


def area(box):
    if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
        return 0
    else:
        return (box[2] - box[0]) * (box[3] - box[1])


def insection(box_small, box_big):
    """
    Return: 比例， 坐标
    """
    min_cor = np.maximum(box_small[:2], box_big[:2]) - box_big[:2]
    max_cor = np.minimum(box_small[2:], box_big[2:]) - box_big[:2]
    box = np.concatenate((min_cor, max_cor))
    prop = area(box) / area(box_small)
    return prop, box


def img_random_split(img, select_num, half_split_img_size):
    """随机裁取图片.

    Args:
        img: 被裁剪的原始图片.
        select_num: 需要裁取的图片数量.
        half_split_img_size： 需要裁取的图片尺寸的一半.
    Return:
        split_imgs, clip_select: 裁取的图片以及其对应于原图的Bbox.
    """
    h, w = img.shape[:2]
    x = np.arange(half_split_img_size, w - half_split_img_size)
    y = np.arange(half_split_img_size, h - half_split_img_size)
    centers = np.meshgrid(x, y)
    centers = np.stack(centers, axis=2)
    centers = np.reshape(centers, [-1, 2])
    random_indices = np.random.randint(0, centers.shape[0], select_num)
    centers_select = centers[random_indices]
    clip_select = np.concatenate(
        (centers_select - half_split_img_size, centers_select + half_split_img_size), 1)
    split_imgs = []
    for clip in clip_select:
        split_img = img[clip[1]:clip[3], clip[0]:clip[2]]
        split_imgs.append(split_img)
    split_imgs = np.stack(split_imgs)
    return split_imgs, clip_select

def img_grid_split(img, row, col, half_split_img_size):
    """网格裁取图片.

    Args:
        img: 被裁剪的原始图片.
        row, col: 网格的行数和列数.
        half_split_img_size： 需要裁取的图片尺寸的一半.
    Return:
        split_imgs, clip_select: 裁取的图片以及其对应于原图的Bbox.
    """
    h, w = img.shape[:2]
    dy = (h-half_split_img_size*2) / (row-1)
    dx = (w-half_split_img_size*2) / (col-1)
    y = np.arange(half_split_img_size, h-half_split_img_size, dy, np.int32)
    x = np.arange(half_split_img_size, w-half_split_img_size, dx, np.int32)
    y = np.append(y, h-half_split_img_size)
    x = np.append(x, w-half_split_img_size)
    centers = np.meshgrid(x, y)
    centers = np.stack(centers, axis=2)
    centers = np.reshape(centers, [-1, 2])
    clip_select = np.concatenate((centers - half_split_img_size, centers + half_split_img_size), 1)
    split_imgs = []
    for clip in clip_select:
        split_img = img[clip[1]:clip[3], clip[0]:clip[2]]
        split_imgs.append(split_img)
    split_imgs = np.stack(split_imgs)
    return split_imgs, clip_select


def draw_bbox(img, bboxes):
    if img.ndim != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for box_name, bbox in bboxes.items():
        cv2.rectangle(img,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0), 2)
        cv2.putText(img, box_name,
                    (bbox[0], bbox[3]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.4, (0, 0, 255), 1)
    return img