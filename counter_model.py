import sys, cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import time

import test as counter_v2
from test_utils import *
from process_op import *

sys.path.append("../counter_v1")
from mobilenetv2 import mobilenetv2
from ops import *


def build_and_model():
    g, img_tensor, _, detection_dict = counter_v2.build_model()
    with g.as_default():
        detection_boxes = detection_dict['detection_boxes']
        detection_scores = detection_dict['detection_scores']
        clip_boxes = tf.stack([detection_boxes[:, 1], detection_boxes[:, 0], \
                                detection_boxes[:, 3], detection_boxes[:, 2]], 1)
        canimg_batch = crop_and_resize(img_tensor, clip_boxes, 64)

        _, predictions, _ = mobilenetv2(canimg_batch, num_classes=2, wid=3, is_train=False)

        classification_scores_with_background = tf.identity(predictions, 'raw_cls_scores')
        classification_scores = tf.slice(classification_scores_with_background, [0, 1],[-1, -1])
        classification_scores = tf.squeeze(classification_scores, 1)
        averange_scores = tf.truediv(tf.add(detection_scores, classification_scores), 2.0)

        result_dict = precise_filter(detection_boxes, 
                                    classification_scores_with_background, 
                                    averange_scores,
                                    scope='And_model/Precise_filter')
        output_node_names = ['And_model/Precise_filter/result_boxes',
                            'And_model/Precise_filter/result_scores']

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            # saver for restore model
            saver_det = get_checkpoint_init_fn('../checkpoints/counter_v2/model.ckpt-150000',
                                                exclude_var=['mobilenetv2'])
            saver_cls = get_checkpoint_init_fn('../checkpoints/64x64-128-0.95-0.001-wid3/mobilenetv2-17',
                                                include_var=['mobilenetv2'])
            saver_det(sess)
            saver_cls(sess)
            write_pb_model('../checkpoints/and_model.pb', sess, g.as_graph_def(), output_node_names)


def and_model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    img_tensor, grid_size_tensor, result_boxes, result_scores = read_pb_model(
            '../checkpoints/model.pb', 
            od_graph_def,
            return_elements=['input_img:0', 'input_grid_size:0',
                            'And_model/Precise_filter/result_boxes:0',
                            'And_model/Precise_filter/result_scores:0'])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        with open('../dataset/name.txt', 'r+') as f:
            for name in f.readlines():
                img_name = name.strip('\n') + '_resize'
                start_time = time()
                img = cv2.imread('../dataset/img/'+img_name+'.jpg', 0)
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                input_img = color_img.astype(np.float32) / 127.5 - 1
                input_img = np.expand_dims(input_img, 0)
                grid_size = np.array([5.0, 3.0], np.float32)

                boxes, scores = sess.run([result_boxes, result_scores],
                                        {img_tensor: input_img, grid_size_tensor: grid_size})
                boxes = np.multiply(boxes, [600,800,600,800])
                boxes = boxes.astype(np.int32)
                print('time: %s' % (time()-start_time))

                drawed_img = draw_bbox(color_img, boxes, scores)
                cv2.imwrite('../dataset/and_result/' + str(img_name) + '_result.jpg', drawed_img)
        print('FINISHED TESTING.')


def build_or_model():
    g, img_tensor, detection_boxes, detection_scores_with_background = counter_v2.build_model(True)
    with g.as_default():
        cls_boxes = tf.placeholder(tf.float32, [None, 4], name='input_boxes')
        canimg_batch = crop_and_resize(img_tensor, cls_boxes, 64)
        _, cls_scores_with_background, _ = mobilenetv2(
            canimg_batch, num_classes=2, wid=3, is_train=False)
        total_boxes = tf.concat([detection_boxes, cls_boxes], 0)
        total_scores_with_background = tf.concat(
            [detection_scores_with_background, cls_scores_with_background], 0)
        result_dict = precise_filter(total_boxes,
                                    total_scores_with_background,
                                    scope='Or_model/Precise_filter')
        output_node_names = ['Or_model/Precise_filter/result_boxes',
                            'Or_model/Precise_filter/result_scores']

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            # saver for restore model
            saver_det = get_checkpoint_init_fn('../checkpoints/counter_v2/model.ckpt-150000',
                                                exclude_var=['mobilenetv2'])
            saver_cls = get_checkpoint_init_fn('../checkpoints/64x64-128-0.95-0.001-wid3/mobilenetv2-17',
                                                include_var=['mobilenetv2'])
            saver_det(sess)
            saver_cls(sess)
            write_pb_model('../checkpoints/or_model.pb', sess, g.as_graph_def(), output_node_names)


def or_model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    img_tensor, grid_size_tensor, input_boxes_tensor, \
        result_boxes, result_scores = read_pb_model(
            '../checkpoints/model.pb', 
            od_graph_def,
            return_elements=['input_img:0', 'input_grid_size:0', 'input_boxes:0',
                            'Or_model/Precise_filter/result_boxes:0',
                            'Or_model/Precise_filter/result_scores:0'])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        with open('../dataset/name.txt', 'r+') as f:
            for name in f.readlines():
                img_name = name.strip('\n') + '_resize'
                start_time = time()
                img = cv2.imread('../dataset/img/'+img_name+'.jpg', 0)
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                input_img = color_img.astype(np.float32) / 127.5 - 1
                input_img = np.expand_dims(input_img, 0)
                grid_size = np.array([5.0, 3.0], np.float32)

                equ_img = cv2.equalizeHist(img)
                gau_img = cv2.GaussianBlur(equ_img, (5, 5), 0)
                lap_img = cv2.Laplacian(gau_img, -1, ksize=5)
                pre_img = cv2.medianBlur(lap_img, 5)
                circles = cv2.HoughCircles(pre_img, cv2.HOUGH_GRADIENT, 1, 30, param1=100,
                            param2=20, minRadius=10, maxRadius=25)
                circles = circles[0, :, :2]
                input_boxes = [[center[1]-30， centers[0]-] for center in circles]
                boxes = np.concatenate((np.minimum(0, cirs - 32), np.maximum(cirs + 32), 1)

                feed_dict = {img_tensor: input_img, 
                            grid_size_tensor: grid_size, 
                            input_boxes_tensor: input_boxes}
                boxes, scores = sess.run([result_boxes, result_scores],
                                        feed_dict)
                boxes = np.multiply(boxes, [600,800,600,800])
                boxes = boxes.astype(np.int32)
                print('time: %s' % (time()-start_time))

                drawed_img = draw_bbox(color_img, boxes, scores)
                cv2.imwrite('../dataset/and_result/' + str(img_name) + '_result.jpg', drawed_img)
        print('FINISHED TESTING.')


if __name__ == '__main__':
    #build_and_model()
    #and_model_test()
    build_or_model()
    or_model_test()