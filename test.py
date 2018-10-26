import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import time

from test_utils import *
from mobilenet import mobilenet_v2
from detection_ops import feature_map_generator, box_predictor, anchor_generator
from process_op import *

tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_float('split_row', 5.0, 'num of row')
tf.app.flags.DEFINE_float('split_col', 3.0, 'num of col')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('image_size', 256, 'Input image resolution')
tf.app.flags.DEFINE_integer('original_image_height',
                            800, 'Height of the original image')
tf.app.flags.DEFINE_integer('original_image_width',
                            600, 'Width of the original image')
tf.app.flags.DEFINE_float('depth_multiplier', 0.75,
                          'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/counter_v2/',
                           'Directory for writing training checkpoints and logs')
tf.app.flags.DEFINE_string('dataset_dir', '../dataset/',
                           'Location of dataset')
tf.app.flags.DEFINE_string('imwrite_dir', '../dataset/result/',
                           'Location of result_imgs')
tf.app.flags.DEFINE_bool('freeze_batchnorm', False,
                         'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('inplace_batchnorm_update', False,
                         'Whether to update batch norm moving average values inplace')
tf.app.flags.DEFINE_bool('is_training', False, 'train or eval')
tf.app.flags.DEFINE_bool('add_hough', False, 'Add hough circle detection')
tf.app.flags.DEFINE_integer('max_output_size', 500, 'Max_output_size')
tf.app.flags.DEFINE_float('iou_threshold', 0.05, 'iou_threshold')
tf.app.flags.DEFINE_float('score_threshold', 0.0, 'score_threshold')

FLAGS = tf.app.flags.FLAGS

_convert_ratio = [
    FLAGS.image_size/FLAGS.original_image_width,
    FLAGS.image_size/FLAGS.original_image_height,
    FLAGS.image_size/FLAGS.original_image_width,
    FLAGS.image_size/FLAGS.original_image_height
]

_ratio_to_value = [
    FLAGS.original_image_width,
    FLAGS.original_image_height,
    FLAGS.original_image_width,
    FLAGS.original_image_height
]

_value_to_ratio = [
    1/FLAGS.original_image_height,
    1/FLAGS.original_image_width,
    1/FLAGS.original_image_height,
    1/FLAGS.original_image_width
]

# _anchors_figure = {
#    'feature_map_dims': [(10, 10), (5, 5)],
#    'scales': [[2.0], [1.0]],
#    'aspect_ratios': [[1.0], [1.0]]
# }
_anchors_figure = {
    'feature_map_dims': [(8, 8), (4, 4)],
    'scales': [[1.6], [0.8]],
    'aspect_ratios': [[1.0], [1.0]]
}


def build_model(apply_or_model=False):
    g = tf.Graph()
    with g.as_default(), tf.device(
            tf.train.replica_device_setter(FLAGS.ps_tasks)):
        anchors = anchor_generator.generate_anchors(**_anchors_figure)
        box_pred = box_predictor.SSDBoxPredictor(
            FLAGS.is_training, FLAGS.num_classes, box_code_size=4)
        batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                         else tf.GraphKeys.UPDATE_OPS)
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32, name='anchors')
        convert_ratio = tf.convert_to_tensor(_convert_ratio, tf.float32, name='convert_ratio')
        value_to_ratio = tf.convert_to_tensor(_value_to_ratio, tf.float32, name='convert_ratio')

        img_tensor = tf.placeholder(tf.float32,
                                    [1, FLAGS.original_image_height, FLAGS.original_image_width, 3],
                                    name='input_img')
        grid_size_tensor = tf.placeholder(tf.float32, [2], 'input_grid_size')
        preimg_batch, grid_points_tl = preprocess(
            img_tensor, grid_size_tensor, FLAGS.image_size, value_to_ratio, apply_or_model)

        with slim.arg_scope([slim.batch_norm], is_training=(
            FLAGS.is_training and not FLAGS.freeze_batchnorm),
            updates_collections=batchnorm_updates_collections),\
            slim.arg_scope(
                mobilenet_v2.training_scope(is_training=None, bn_decay=0.997)):
            _, image_features = mobilenet_v2.mobilenet_base(
                preimg_batch,
                final_endpoint='layer_18',
                depth_multiplier=FLAGS.depth_multiplier,
                finegrain_classification_mode=True)
            feature_maps = feature_map_generator.pooling_pyramid_feature_maps(
                base_feature_map_depth=0,
                num_layers=2,
                image_features={
                    'image_features': image_features['layer_18']
                })
            pred_dict = box_pred.predict(feature_maps.values(), [1, 1])
            box_encodings = tf.concat(pred_dict['box_encodings'], axis=1)
            if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
                box_encodings = tf.squeeze(box_encodings, axis=2)
            class_predictions_with_background = tf.concat(
                pred_dict['class_predictions_with_background'], axis=1)
        detection_boxes, detection_scores_with_background = postprocess(
            anchors, box_encodings, 
            class_predictions_with_background,
            convert_ratio, grid_points_tl,
            num_classes=FLAGS.num_classes)
        detection_dict = precise_filter(detection_boxes, 
                                        detection_scores_with_background,
                                        max_output_size=FLAGS.max_output_size,
                                        iou_threshold=FLAGS.iou_threshold,
                                        score_threshold=FLAGS.score_threshold)
        if apply_or_model:
            return g, img_tensor, detection_boxes, detection_scores_with_background
        else:
            return g, img_tensor, grid_size_tensor, detection_dict


def convert_cpkt_to_pb():
    g, _, _, _ = build_model()
    output_node_names = ['Precise_filter/result_boxes',
                         'Precise_filter/result_scores']
    with g.as_default():
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            # saver for restore model
            saver = tf.train.Saver()
            print('[*] Try to load trained model...')
            ckpt_name = load(sess, saver, FLAGS.checkpoint_dir)
            write_pb_model(FLAGS.checkpoint_dir+ckpt_name+'.pb',
                            sess, g.as_graph_def(), output_node_names)


def test_model():
    g, img_tensor, grid_size_tensor, detection_dict = build_model()
    with g.as_default():
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            # saver for restore model
            saver = tf.train.Saver()
            print('[*] Try to load trained model...')
            _ = load(sess, saver, FLAGS.checkpoint_dir)

            print('START TESTING...')
            with open(FLAGS.dataset_dir+'name.txt', 'r+') as f:
                for name in f.readlines():
                    img_name = name.strip('\n') + '_resize'
                    img = cv2.imread(FLAGS.dataset_dir +
                                     'img/'+img_name+'.jpg', 0)
                    start_time = time()
                    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    input_img = color_img.astype(np.float32) / 127.5 - 1
                    input_img = np.expand_dims(input_img, 0)
                    grid_size = np.array([FLAGS.split_row, FLAGS.split_col], np.float32)

                    detec = sess.run(detection_dict,
                                     {img_tensor: input_img, grid_size_tensor: grid_size})

                    boxes = np.multiply(
                        detec['result_boxes'], _ratio_to_value)
                    boxes = boxes.astype(np.int32)
                    scores = detec['result_scores']
                    print('time: %s' % (time()-start_time))

                    drawed_img = draw_bbox(color_img, boxes, scores)
                    if FLAGS.add_hough:
                        hough_boxes, hough_scores = hough_cir(img, 50, 35)
                        drawed_img = draw_bbox(
                            drawed_img, hough_boxes, hough_scores, color=(0, 0, 255))
                    cv2.imwrite(FLAGS.imwrite_dir + str(img_name) +
                                '_result.jpg', drawed_img)
            print('FINISHED TESTING.')


def model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FLAGS.checkpoint_dir+'model.ckpt-150000.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor, grid_size_tensor, detection_boxes, detection_scores = tf.import_graph_def(od_graph_def,
            return_elements=['input_img:0', 'input_grid_size:0', 'Precise_filter/result_boxes:0','Precise_filter/result_scores:0'])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        with open(FLAGS.dataset_dir+'name.txt', 'r+') as f:
            for name in f.readlines():
                img_name = name.strip('\n') + '_resize'
                start_time = time()
                img = cv2.imread(FLAGS.dataset_dir +
                                'img/'+img_name+'.jpg', 0)
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                input_img = color_img.astype(np.float32) / 127.5 - 1
                input_img = np.expand_dims(input_img, 0)
                grid_size = np.array([FLAGS.split_row, FLAGS.split_col], np.float32)

                boxes, scores = sess.run([detection_boxes, detection_scores],
                                {img_tensor: input_img, grid_size_tensor: grid_size})
                boxes = np.multiply(boxes, _ratio_to_value)
                boxes = boxes.astype(np.int32)
                print('time: %s' % (time()-start_time))

                drawed_img = draw_bbox(color_img, boxes, scores)
                if FLAGS.add_hough:
                    hough_boxes, hough_scores = hough_cir(img, 50, 35)
                    drawed_img = draw_bbox(
                        drawed_img, hough_boxes, hough_scores, color=(0, 0, 255))
                cv2.imwrite(FLAGS.imwrite_dir + str(img_name) +
                            '_result.jpg', drawed_img)
        print('FINISHED TESTING.')


def main(unused_arg):
    convert_cpkt_to_pb()
    test_model()
    #model_test()


if __name__ == '__main__':
    tf.app.run(main)
