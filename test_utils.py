import os, cv2
import numpy as np
import tensorflow as tf


def load(sess, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("[*] Success to read {}".format(ckpt_name))
        return ckpt_name
    else:
        raise Exception("[*] Failed to find a checkpoint")


def hough_cir(img, minDist, pre, minRadius=15, maxRadius=30):
    pre_img = cv2.equalizeHist(img)
    pre_img = cv2.GaussianBlur(pre_img, (5, 5), 0)
    pre_img = cv2.Laplacian(pre_img, -1, ksize=5)
    pre_img = cv2.medianBlur(pre_img, 5)
    cirs = cv2.HoughCircles(pre_img, cv2.HOUGH_GRADIENT, 1, minDist, param1=100,
                            param2=pre, minRadius=minRadius, maxRadius=maxRadius)
    cirs = np.int32(np.round(cirs[0, :, :2]))
    boxes = np.concatenate((cirs - 20, cirs + 20), 1)
    scores = np.ones(boxes.shape[0], np.float32)
    return boxes, scores


def draw_bbox(img, boxes, scores, color=(0, 255, 0)):
    h, w = img.shape[:2]
    for box, score in zip(boxes, scores):
        min_x = max(0, box[0])
        min_y = max(0, box[1])
        max_x = min(w, box[2])
        max_y = min(h, box[3])
        cv2.rectangle(img, (min_x, min_y),
                      (max_x, max_y), color, 2)
        cv2.putText(img, 'cap:' + str(score)[0:4], (min_x, max_y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
    return img
