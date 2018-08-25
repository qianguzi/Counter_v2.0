import os, cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from detection_ops import box_coder
from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils, data_ops
from detection_ops import feature_map_generator, box_predictor, anchor_generator

tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_integer('batch_size', 8, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('image_size', 320, 'Input image resolution')
tf.app.flags.DEFINE_integer('num_examples', 6000, 'the number of examples')
tf.app.flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/',
                    'Directory for writing training checkpoints and logs')
tf.app.flags.DEFINE_string('dataset_dir', '../tfrecords/test.tfrecords', 'Location of dataset')
tf.app.flags.DEFINE_string('imwrite_dir', '/media/jun/data/capdataset/detect/result/',
                    'Location of result_imgs')
tf.app.flags.DEFINE_bool('freeze_batchnorm', True,
                     'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('inplace_batchnorm_update', True,
                     'Whether to update batch norm moving average values inplace')
tf.app.flags.DEFINE_bool('is_training', False, 'train or eval')
tf.app.flags.DEFINE_integer('max_output_size', 200, 'Max_output_size')
tf.app.flags.DEFINE_float('iou_threshold', 0.0, 'iou_threshold')
tf.app.flags.DEFINE_float('score_threshold', 0.0, 'score_threshold')

FLAGS = tf.app.flags.FLAGS

def _conv_hyperparams_fn():
  """Defines Box Predictor scope.

  Returns:
    An argument scope to use via arg_scope.
  """
  # Set weight_decay for weights in Conv layers.
  with slim.arg_scope([slim.batch_norm], decay=0.999), \
       slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                    weights_initializer=tf.truncated_normal_initializer(),
                    normalizer_fn=slim.batch_norm), \
       slim.arg_scope([slim.conv2d], \
                    weights_regularizer=slim.l2_regularizer(scale=1.0)) as s:
    return s

def _batch_decode(box_encodings, anchors):
    """Decodes a batch of box encodings with respect to the anchors.

    Args:
      box_encodings: A float32 tensor of shape
        [batch_size, num_anchors, box_code_size] containing box encodings.

    Returns:
      decoded_boxes: A float32 tensor of shape
        [batch_size, num_anchors, 4] containing the decoded boxes.
      decoded_keypoints: A float32 tensor of shape
        [batch_size, num_anchors, num_keypoints, 2] containing the decoded
        keypoints if present in the input `box_encodings`, None otherwise.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors = tf.reshape(tiled_anchor_boxes, [-1, 4])
    decoded_boxes = box_coder.decode(
        tf.reshape(box_encodings, [-1, 4]),
        tiled_anchors)
    decoded_boxes = tf.reshape(decoded_boxes, tf.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes

def postprocess(anchors, box_encodings,
                class_predictions_with_background,
                scope=None):
    """Converts prediction tensors to final detections."""
    with tf.name_scope(scope, 'Postprocessor',
                    [anchors, box_encodings, class_predictions_with_background]):
      max_total_size = tf.constant(FLAGS.max_output_size, dtype=tf.int32)
      box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
      detection_boxes = _batch_decode(box_encodings, anchors)
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
      num_detections_list = []
      for i in range(FLAGS.batch_size):
        positive_loc = tf.equal(
            tf.argmax(detection_scores_with_background[i], 1),
            0)
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
        
        num_detections = shape_utils.combined_static_and_dynamic_shape(detection_box)[0]
        detection_box = tf.pad(detection_box,
                                  [[0, max_total_size - num_detections],[0, 0]])
        detection_score = tf.pad(detection_score,
                                  [[0, max_total_size - num_detections]])
        detection_boxes_list.append(detection_box)
        detection_scores_list.append(detection_score)
        num_detections_list.append(num_detections)

      detection_boxes = tf.stack(detection_boxes_list)
      detection_scores = tf.stack(detection_scores_list)
      num_detections = tf.stack(num_detections_list)
      detection_dict = {
          'detection_boxes': detection_boxes,
          'detection_scores': detection_scores,
          'num_detections': num_detections
      }

      return detection_dict

def build_model():
  anchors = anchor_generator.generate_anchors(feature_map_dims=[(10, 10), (5, 5), (3, 3)],
                                              scales=[[0.60], [0.70, 0.90], [0.50, 0.75]],
                                              aspect_ratios=[[1.0], [1.0, 1.0], [1.0, 1.0]])
  box_pred = box_predictor.SSDBoxPredictor(
        FLAGS.is_training, FLAGS.num_classes, box_code_size=4, 
        conv_hyperparams_fn = _conv_hyperparams_fn)
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    img_batch, _, _, name_batch = data_ops.get_batch(FLAGS.dataset_dir, 8)
    preimg_batch = tf.cast(img_batch, tf.float32) / 127.5 - 1
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32, name='anchors')
    with slim.arg_scope([slim.batch_norm],
            is_training=(FLAGS.is_training and not FLAGS.freeze_batchnorm),
            updates_collections=batchnorm_updates_collections),\
        slim.arg_scope(
            mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)):
      _, image_features = mobilenet_v2.mobilenet_base(
          preimg_batch,
          final_endpoint='layer_18',
          depth_multiplier=FLAGS.depth_multiplier,
          finegrain_classification_mode=True)
      feature_maps = feature_map_generator.pooling_pyramid_feature_maps(
          base_feature_map_depth=0,
          num_layers=3,
          image_features={  
              'image_features': image_features['layer_18']
          })
      pred_dict = box_pred.predict(feature_maps.values(), [1, 2, 2])
      box_encodings = tf.concat(pred_dict['box_encodings'], axis=1)
      if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
        box_encodings = tf.squeeze(box_encodings, axis=2)
      class_predictions_with_background = tf.concat(
          pred_dict['class_predictions_with_background'], axis=1)
    detection_dict = postprocess(anchors, box_encodings,
                                class_predictions_with_background)
  return g, img_batch, name_batch, detection_dict


def load(sess, saver, checkpoint_dir):
    #import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # print(ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return ckpt_name
    else:
        raise Exception("[*] Failed to find a checkpoint")

def draw_and_save(imgs, names, bboxes, scores, nums):
    batch_size, h, w, d = imgs.shape
    bboxes = (bboxes * h).astype(int)
    for n in range(batch_size):
        img = imgs[n]
        bbox = bboxes[n]
        score = scores[n]
        if d == 1:
            color_img = cv2.cvtColor(imgs[n], cv2.COLOR_GRAY2RGB)
        else:
            color_img = imgs[n]
        for i in range(nums[n]):
            cv2.rectangle(color_img, (bbox[i][0], bbox[i][1]),
                          (bbox[i][2], bbox[i][3]), (0, 255, 0), 1)
            cv2.putText(color_img, 'cap' + str(i+1) + ':' + str(score[i])[0:4], (bbox[i][0], bbox[i][3]),
                        cv2.FONT_HERSHEY_COMPLEX, 0.2, (0, 0, 255), 1)
        cv2.imwrite(FLAGS.imwrite_dir + str(names[n]) + '_result.jpg', color_img)


def test_model():
    g, img_batch, name_batch, detection_dict = build_model()
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

        max_steps = int(FLAGS.num_examples / FLAGS.batch_size)
        print('START TESTING...')
        for i in range(max_steps):
          # test
          imgs, names, bboxes, scores, nums= sess.run([img_batch, name_batch, 
                                                detection_dict['detection_boxes'],
                                                detection_dict['detection_scores'],
                                                detection_dict['num_detections']])
          draw_and_save(imgs, names, bboxes, scores, nums)
        coord.request_stop()
        coord.join(threads)
        print('FINISHED TESTING.')


def main(unused_arg):
    test_model()


if __name__ == '__main__':
  tf.app.run(main)
