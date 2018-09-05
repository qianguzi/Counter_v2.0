import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils
from dataset_ops.dataset_util import img_grid_split
from detection_ops import feature_map_generator, box_coder, box_predictor, anchor_generator

tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_integer('split_row', 6, 'num of row')
tf.app.flags.DEFINE_integer('split_col', 5, 'num of col')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('image_size', 320, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 1.0,
                          'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/myfile/checkpoints/',
                           'Directory for writing training checkpoints and logs')
tf.app.flags.DEFINE_string('dataset_dir', '/media/jun/data/capdataset/detection/test/',
                           'Location of dataset')
tf.app.flags.DEFINE_string('imwrite_dir', '/media/jun/data/capdataset/detection/result_test/',
                           'Location of result_imgs')
tf.app.flags.DEFINE_bool('freeze_batchnorm', False,
                         'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('inplace_batchnorm_update', False,
                         'Whether to update batch norm moving average values inplace')
tf.app.flags.DEFINE_bool('is_training', False, 'train or eval')
tf.app.flags.DEFINE_integer('max_output_size', 20, 'Max_output_size')
tf.app.flags.DEFINE_float('iou_threshold', 0.3, 'iou_threshold')
tf.app.flags.DEFINE_float('score_threshold', 0.7, 'score_threshold')

FLAGS = tf.app.flags.FLAGS

_anchors_figure = {
    'feature_map_dims': [(10, 10), (5, 5)],
    'scales': [[2.0], [1.0]],
    'aspect_ratios': [[1.0], [1.0]]
}


def postprocess(anchors, box_encodings,
                class_predictions_with_background,
                scope=None):
    """Converts prediction tensors to final detections."""
    with tf.name_scope(scope, 'Postprocessor',
                       [anchors, box_encodings, class_predictions_with_background]):
        box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
        detection_boxes = box_coder.batch_decode(box_encodings, anchors)
        detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')

        detection_scores_with_background = tf.nn.softmax(
            class_predictions_with_background, name='softmax')
        detection_scores_with_background = tf.identity(
            detection_scores_with_background, 'raw_box_scores')
        detection_scores = tf.slice(detection_scores_with_background, [0, 0, 1],
                                    [-1, -1, -1])
        detection_scores = tf.squeeze(detection_scores, 2)

        detection_boxes_list = []
        detection_scores_list = []
        for i in range(FLAGS.split_row * FLAGS.split_col):
            positive_loc = tf.equal(
                tf.argmax(detection_scores_with_background[i], 1),
                1)
            positive_indices = tf.cast(
                tf.squeeze(tf.where(positive_loc), 1),
                tf.int32)
            detection_box = tf.gather(detection_boxes[i], positive_indices)
            detection_score = tf.gather(detection_scores[i], positive_indices)

            high_score_indices = tf.cast(tf.reshape(
                tf.where(tf.greater(detection_score, FLAGS.score_threshold)),
                [-1]), tf.int32)
            detection_box = tf.gather(detection_box, high_score_indices)
            detection_score = tf.gather(detection_score, high_score_indices)

            selected_indices = tf.image.non_max_suppression(
                detection_box,
                detection_score,
                max_output_size=FLAGS.max_output_size,
                iou_threshold=FLAGS.iou_threshold)
            detection_box = tf.gather(detection_box, selected_indices)
            detection_score = tf.gather(detection_score, selected_indices)

            detection_boxes_list.append(detection_box)
            detection_scores_list.append(detection_score)

        detection_dict = {
            'detection_boxes': detection_boxes_list,
            'detection_scores': detection_scores_list,
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
        img_batch = tf.placeholder(tf.int32, [None, 320, 320, 3], 'inputs')
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
        detection_dict = postprocess(anchors, box_encodings,
                                     class_predictions_with_background)
        return g, img_batch, detection_dict


def load(sess, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("[*] Success to read {}".format(ckpt_name))
        return ckpt_name
    else:
        raise Exception("[*] Failed to find a checkpoint")


def draw_and_save(img, name, boxes, scores):
    for box, score in zip(boxes, scores):
        cv2.rectangle(img, (box[0], box[1]),
                      (box[2], box[3]), (0, 255, 0), 1)
        cv2.putText(img, 'cap:' + str(score)[0:4], (box[0], box[3]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(FLAGS.imwrite_dir + str(name) + '_result.jpg', img)


def nms(boxes, scores, max_out):
    pass


def test_model():
    g, img_batch, detection_dict = build_model()
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
            input_imgs = {img_path[:-4]: cv2.imread(FLAGS.dataset_dir + img_path, 0)
                          for img_path in os.listdir(FLAGS.dataset_dir)}
            for img_name in input_imgs:
                img = input_imgs[img_name]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                split_imgs, clip_select = img_grid_split(
                    img, FLAGS.split_row, FLAGS.split_col, FLAGS.image_size/2)
                split_locs = np.tile(clip_select[:, :2], 2)
                detec = sess.run(detection_dict, {img_batch: split_imgs})

                boxes_list = []
                scores_list = []
                for split_boxes, split_scores, split_loc in zip(
                        detec['detection_boxes'], detec['detection_scores'], split_locs):
                    if split_boxes.shape[0]:
                        split_boxes = (split_boxes * FLAGS.image_size).astype(int) + split_loc
                        boxes_list.append(split_boxes)
                        scores_list.append(split_scores)
                boxes = np.concatenate(boxes_list)
                scores = np.concatenate(scores_list)
                #boxes, scores = nms(boxes, scores, max_out=60)
                draw_and_save(img, img_name, boxes, scores)
            coord.request_stop()
            coord.join(threads)
            print('FINISHED TESTING.')


def main(unused_arg):
    test_model()


if __name__ == '__main__':
    tf.app.run(main)
