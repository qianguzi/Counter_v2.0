import os, cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import time

from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils
from dataset_ops.dataset_util import img_grid_split
from detection_ops import feature_map_generator, box_coder, box_predictor, anchor_generator

tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_integer('split_row', 7, 'num of row')
tf.app.flags.DEFINE_integer('split_col', 4, 'num of col')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('image_size', 256, 'Input image resolution')
tf.app.flags.DEFINE_integer('original_image_height', 800, 'Height of the original image')
tf.app.flags.DEFINE_integer('original_image_width', 600, 'Width of the original image')
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
tf.app.flags.DEFINE_integer('max_output_size', 63, 'Max_output_size')
tf.app.flags.DEFINE_float('iou_threshold', 0.0, 'iou_threshold')
tf.app.flags.DEFINE_float('score_threshold', 0.85, 'score_threshold')

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
    1/FLAGS.original_image_width,
    1/FLAGS.original_image_height,
    1/FLAGS.original_image_width,
    1/FLAGS.original_image_height
]

#_anchors_figure = {
#    'feature_map_dims': [(10, 10), (5, 5)],
#    'scales': [[2.0], [1.0]],
#    'aspect_ratios': [[1.0], [1.0]]
#}
_anchors_figure = {
    'feature_map_dims': [(8, 8), (4, 4)],
    'scales': [[1.6], [0.8]],
    'aspect_ratios': [[1.0], [1.0]]
}

def postprocess(anchors, split_locs, box_encodings,
                class_predictions_with_background,
                scope=None):
    """Converts prediction tensors to final detections."""
    with tf.name_scope(scope, 'Postprocessor',
            [anchors, split_locs, box_encodings, class_predictions_with_background]):
        box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
        detection_boxes = box_coder.batch_decode(box_encodings, anchors)
        detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')
        convert_ratio = tf.constant(_convert_ratio, tf.float32, name='convert_ratio')
        detection_boxes = tf.multiply(detection_boxes, convert_ratio) + split_locs
        detection_boxes = tf.reshape(detection_boxes, [-1, 4])
    
        detection_scores_with_background = tf.nn.softmax(
            class_predictions_with_background, name='softmax')
        detection_scores_with_background = tf.identity(
            detection_scores_with_background, 'raw_box_scores')
        detection_scores_with_background = tf.reshape(
            detection_scores_with_background, [-1, FLAGS.num_classes])
        detection_scores = tf.slice(detection_scores_with_background, [0, 1],
                                    [-1, -1])
        detection_scores = tf.squeeze(detection_scores, 1)

        positive_locs = tf.equal(
            tf.argmax(detection_scores_with_background, 1),
            1)
        positive_indices = tf.cast(
            tf.squeeze(tf.where(positive_locs), 1),
            tf.int32)
        detection_boxes = tf.gather(detection_boxes, positive_indices)
        detection_scores = tf.gather(detection_scores, positive_indices)

        high_score_indices = tf.cast(tf.reshape(
            tf.where(tf.greater(detection_scores, FLAGS.score_threshold)),
            [-1]), tf.int32)
        detection_boxes = tf.gather(detection_boxes, high_score_indices)
        detection_scores = tf.gather(detection_scores, high_score_indices)

        selected_indices = tf.image.non_max_suppression(
            detection_boxes,
            detection_scores,
            max_output_size=FLAGS.max_output_size,
            iou_threshold=FLAGS.iou_threshold)
        detection_boxes = tf.gather(detection_boxes, selected_indices)
        detection_scores = tf.gather(detection_scores, selected_indices)

        detection_dict = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores
        }

        return detection_dict


def build_model():
    g = tf.Graph()
    with g.as_default(), tf.device(
            tf.train.replica_device_setter(FLAGS.ps_tasks)):
        anchors = anchor_generator.generate_anchors(**_anchors_figure)
        box_pred = box_predictor.SSDBoxPredictor(
            FLAGS.is_training, FLAGS.num_classes, box_code_size=4)
        batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                         else tf.GraphKeys.UPDATE_OPS)
        img_batch = tf.placeholder(tf.int32,
            [FLAGS.split_row * FLAGS.split_col, FLAGS.image_size, FLAGS.image_size, 3], 'input_imgs')
        split_loc_batch = tf.placeholder(
            tf.float32, [FLAGS.split_row * FLAGS.split_col, 4], 'input_locs')
        split_locs = tf.expand_dims(split_loc_batch, 1)
        preimg_batch = tf.cast(img_batch, tf.float32) / 127.5 - 1
        anchors = tf.convert_to_tensor(
                anchors, dtype=tf.float32, name='anchors')
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
        detection_dict = postprocess(anchors, split_locs, box_encodings,
                                     class_predictions_with_background)
        return g, img_batch, split_loc_batch, detection_dict


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
    for box, score in zip(boxes, scores):
        min_x = max(0, box[0])
        min_y = max(0, box[1])
        max_x = min(FLAGS.original_image_width, box[2])
        max_y = min(FLAGS.original_image_height, box[3])
        cv2.rectangle(img, (min_x, min_y),
                      (max_x, max_y), color, 2)
        cv2.putText(img, 'cap:' + str(score)[0:4], (min_x, max_y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
    return img


def test_model():
    g, img_batch, split_loc_batch, detection_dict = build_model()
    with g.as_default():
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # saver for restore model
            saver = tf.train.Saver()
            print('[*] Try to load trained model...')
            ckpt_name = load(sess, saver, FLAGS.checkpoint_dir)

            print('START TESTING...')
            with open(FLAGS.dataset_dir+'name.txt', 'r+') as f:
                for name in f.readlines():
                    img_name = name.strip('\n') + '_resize'
                    img = cv2.imread(FLAGS.dataset_dir+'img/'+img_name+'.jpg', 0)
                    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    split_imgs, clip_select = img_grid_split(
                        color_img, FLAGS.split_row, FLAGS.split_col, FLAGS.image_size)
                    split_locs = np.tile(clip_select[:, :2], 2)
                    split_locs = np.multiply(split_locs, _value_to_ratio)
                    split_locs = split_locs.astype(np.float32)
                    
                    start_time = time()
                    detec = sess.run(
                        detection_dict, {img_batch: split_imgs, split_loc_batch: split_locs})
                    print('time: %s' % (time()-start_time))

                    boxes = np.multiply(detec['detection_boxes'], _ratio_to_value)
                    boxes = boxes.astype(np.int32)
                    scores = detec['detection_scores']

                    drawed_img = draw_bbox(color_img, boxes, scores)
                    if FLAGS.add_hough:
                        hough_boxes, hough_scores = hough_cir(img, 50, 35)
                        drawed_img = draw_bbox(drawed_img, hough_boxes, hough_scores, color=(0, 0, 255))
                    cv2.imwrite(FLAGS.imwrite_dir + str(img_name) + '_result.jpg', drawed_img)
            coord.request_stop()
            coord.join(threads)
            print('FINISHED TESTING.')


def main(unused_arg):
    test_model()


if __name__ == '__main__':
    tf.app.run(main)
