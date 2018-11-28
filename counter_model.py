import sys, cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from time import time

import test as counter_v2
from test_utils import *
from process_op import *
from detection_ops.utils import shape_utils

sys.path.append("../counter_v1")
from mobilenetv2 import mobilenetv2
from ops import *


def build_and_model():
    g, img_tensor, input_boxes, detection_boxes, detection_scores = counter_v2.build_model(apply_and_model=True)
    with g.as_default():
        clip_boxes = tf.stack([detection_boxes[:, 1], detection_boxes[:, 0], \
                                detection_boxes[:, 3], detection_boxes[:, 2]], 1)
        canimg_batch = crop_and_resize(img_tensor, clip_boxes, 64)

        _, cls_scores_with_background, _ = mobilenetv2(canimg_batch, num_classes=2, wid=3, is_train=False)
        cls_scores = tf.slice(cls_scores_with_background, [0, 1],[-1, -1])
        cls_scores = tf.squeeze(cls_scores, 1)
        
        positive_indices = tf.equal(
            tf.argmax(cls_scores_with_background, 1),
            1)
        positive_indices = tf.squeeze(tf.where(positive_indices), 1)
        cls_boxes = tf.gather(detection_boxes, positive_indices)
        cls_scores = tf.gather(cls_scores, positive_indices)
        averange_scores = tf.truediv(tf.add(detection_scores, cls_scores), 2.0)

        num_batch = shape_utils.combined_static_and_dynamic_shape(input_boxes)
        input_scores = tf.tile([0.7], [num_batch[0]])
        total_boxes = tf.concat([cls_boxes, input_boxes], 0)
        total_scores = tf.concat([averange_scores, input_scores], 0)
        result_dict = non_max_suppression(total_boxes, 
                                        total_scores,
                                        iou_threshold=0.03,
                                        scope='And_model/Non_max_suppression')
        output_node_names = ['And_model/Non_max_suppression/result_boxes',
                            'And_model/Non_max_suppression/result_scores',
                            'And_model/Non_max_suppression/abnormal_indices']

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
    img_tensor, grid_size_tensor, input_boxes_tensor, \
        result_boxes, result_scores, abnormal_indices = read_pb_model(
            '../checkpoints/model.pb', 
            od_graph_def,
            return_elements=['input_img:0', 'input_grid_size:0', 'input_boxes:0',
                            'And_model/Precise_filter/result_boxes:0',
                            'And_model/Precise_filter/result_scores:0',
                            'And_model/Non_max_suppression/abnormal_indices'])
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

                input_boxes = hough_cir_detection(img, minDist=30, precise=25, apply_custom_radius=False)
                feed_dict = {img_tensor: input_img, 
                            grid_size_tensor: grid_size,
                            input_boxes_tensor: input_boxes}
                boxes, scores, abn_indices = sess.run([result_boxes, result_scores, abnormal_indices],
                                        feed_dict)
                boxes, scores = abnormal_filter(boxes, scores, abn_indices)
                boxes = np.multiply(boxes, [600,800,600,800])
                boxes = boxes.astype(np.int32)
                print('time: %s' % (time()-start_time))

                drawed_img = draw_bbox(color_img, boxes, scores)
                cv2.imwrite('../dataset/and_result/' + str(img_name) + '_result.jpg', drawed_img)
        print('FINISHED TESTING.')


def build_or_model():
    g, img_tensor, cls_boxes, detection_boxes, detection_scores = counter_v2.build_model(apply_or_model=True)
    with g.as_default():
        clip_boxes = tf.stack([cls_boxes[:, 1], cls_boxes[:, 0], \
                            cls_boxes[:, 3], cls_boxes[:, 2]], 1)
        canimg_batch = crop_and_resize(img_tensor, clip_boxes, 64)
        _, cls_scores_with_background, _ = mobilenetv2(
            canimg_batch, num_classes=2, wid=3, is_train=False)
        
        cls_scores = tf.slice(cls_scores_with_background, [0, 1],[-1, -1])
        cls_scores = tf.squeeze(cls_scores, 1)
        
        positive_indices = tf.equal(
            tf.argmax(cls_scores_with_background, 1),
            1)
        positive_indices = tf.squeeze(tf.where(positive_indices), 1)
        cls_boxes = tf.gather(cls_boxes, positive_indices)
        cls_scores = tf.gather(cls_scores, positive_indices)

        total_boxes = tf.concat([detection_boxes, cls_boxes], 0)
        total_scores = tf.concat([detection_scores, cls_scores], 0)
        result_dict = non_max_suppression(total_boxes,
                                    total_scores,
                                    iou_threshold=0.03,
                                    scope='Or_model/Non_max_suppression')
        output_node_names = ['Or_model/Non_max_suppression/result_boxes',
                            'Or_model/Non_max_suppression/result_scores',
                            'Or_model/Non_max_suppression/abnormal_indices']

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
        result_boxes, result_scores, abnormal_indices = read_pb_model(
            '../checkpoints/or_model.pb', 
            od_graph_def,
            return_elements=['input_img:0', 'input_grid_size:0', 'input_boxes:0', \
                            'Or_model/Precise_filter/result_boxes:0', \
                            'Or_model/Precise_filter/result_scores:0', \
                            'Or_model/Precise_filter/abnormal_indices:0'])
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

                input_boxes = hough_cir_detection(img, minDist=30, precise=25, apply_custom_radius=False)
                feed_dict = {img_tensor: input_img,  
                             grid_size_tensor: grid_size,
                             input_boxes_tensor: input_boxes}
                boxes, scores, abn_indices = sess.run(
                    [result_boxes, result_scores, abnormal_indices], feed_dict)
                boxes, scores = abnormal_filter(boxes, scores, abn_indices)
                boxes = np.multiply(boxes, [600,800,600,800])
                boxes = boxes.astype(np.int32)
                print('time: %s' % (time()-start_time))

                drawed_img = draw_bbox(color_img, boxes, scores)
                cv2.imwrite('../dataset/or_result/' + str(img_name) + '_result.jpg', drawed_img)
        print('FINISHED TESTING.')


if __name__ == '__main__':
    #build_and_model()
    #and_model_test()
    build_or_model()
    or_model_test()