import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from detection_ops import box_coder
from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils, data_ops
from detection_ops import feature_map_generator, box_predictor, anchor_generator

tf.app.flags.DEFINE_integer('batch_size', 8, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('image_size', 256, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 0.50, 'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/',
                    'Directory for writing training checkpoints and logs')
tf.app.flags.DEFINE_string('dataset_dir', '../tfrecords/train.tfrecords', 'Location of dataset')
tf.app.flags.DEFINE_bool('freeze_batchnorm', True,
                     'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('inplace_batchnorm_update', True,
                     'Whether to update batch norm moving average values inplace')
tf.app.flags.DEFINE_bool('is_training', False, 'train or eval')
tf.app.flags.DEFINE_integer('max_output_size', 18, 'Max_output_size')
tf.app.flags.DEFINE_integer('iou_threshold', 0.4, 'iou_threshold')
tf.app.flags.DEFINE_integer('score_threshold', 0.75, 'score_threshold')

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

def postprocess(box_encodings, class_predictions_with_background, anchors, true_image_shapes):
    """Converts prediction tensors to final detections."""
    with tf.name_scope('Postprocessor'):
      max_total_size = tf.constant(FLAGS.max_output_size, dtype=int32)
      box_encodings = box_encodings
      box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
      class_predictions = class_predictions_with_background
      detection_boxes = _batch_decode(box_encodings, anchors)
      detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')
      detection_boxes = tf.expand_dims(detection_boxes, axis=2)

      detection_scores_with_background = tf.sigmoid(class_predictions, name='sigmoid')
      detection_scores_with_background = tf.identity(
          detection_scores_with_background, 'raw_box_scores')
      detection_scores = tf.slice(detection_scores_with_background, [0, 0, 1],
                                  [-1, -1, -1])
      detection_scores = tf.squeeze(detection_scores, 2)

      detection_boxes_list = []
      detection_scores_list = []
      num_detections_list = []
      for i in range(FLAGS.batch_size):
        positive_indices = tf.cast(tf.reshape(
          tf.where(tf.equal(tf.argmax(detection_scores_with_background[i], 1), 1)),
          [-1]), tf.int32)
        detection_boxes = tf.gather(detection_boxes[i], positive_indices)
        detection_scores = tf.gather(detection_scores[i], positive_indices)

        high_score_indices = tf.cast(tf.reshape(
          tf.where(tf.greater(detection_scores, FLAGS.score_threshold)),
          [-1]), tf.int32)
        detection_boxes = tf.gather(detection_boxes, high_score_indices)
        detection_scores = tf.gather(detection_scores, high_score_indices)

        selected_indices = tf.image.non_max_suppression(
             detection_boxes,
             detection_scores,
             max_output_size=FLAGS.max_output_size,
             iou_threshold==FLAGS.iou_threshold)
        detection_boxes = tf.gather(detection_boxes, selected_indices)
        detection_scores = tf.gather(detection_scores, selected_indices)
        num_detections = shape_utils.combined_static_and_dynamic_shape(detection_boxes)[0]
        detection_boxes = tf.pad(detection_boxes,
                                  [[0, max_total_size - num_detections],[0, 0]])
        detection_scores = tf.pad(detection_scores,
                                  [[0, max_total_size - num_detections]])
        detection_boxes_list.append(detection_boxes)
        detection_scores_list.append(detection_scores)
        num_detections_list.append(num_detections)

      detection_boxes = tf.concat(detection_boxes_list)
      detection_scores = tf.concat(detection_scores_list)
      num_detections = tf.concat(num_detections_list)
      detection_dict = {
          'detection_boxes': detection_boxes,
          'detection_scores': detection_scores,
          'num_detections': num_detections
      }

      return detection_dict

def build_model():
  anchors = anchor_generator.generate_anchors(feature_map_dims=[(8, 8), (4, 4)],
                                              scales=[[0.95], [0.60, 0.80]],
                                              aspect_ratios=[[1.0], [1.0, 1.0]])
  box_pred = box_predictor.SSDBoxPredictor(
        FLAGS.is_training, FLAGS.num_classes, box_code_size=4, 
        conv_hyperparams_fn = _conv_hyperparams_fn)
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    img_batch, bbox_batch, num_batch, name_batch = data_ops.get_batch(FLAGS.dataset_dir, 8)
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32, name='anchors')
    with slim.arg_scope([slim.batch_norm],
            is_training=(FLAGS.is_training and not FLAGS.freeze_batchnorm),
            updates_collections=batchnorm_updates_collections),\
        slim.arg_scope(
            mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)):
      _, image_features = mobilenet_v2.mobilenet_base(
          img_batch,
          final_endpoint='layer_19',
          depth_multiplier=FLAGS.depth_multiplier,
          finegrain_classification_mode=True)
      feature_maps = feature_map_generator.pooling_pyramid_feature_maps(
          base_feature_map_depth=0,
          num_layers=2,
          image_features={  
              'image_features': image_features['layer_19']
          })
      pred_dict = box_pred.predict(feature_maps.values(), [1, 2])
      box_encodings = tf.concat(pred_dict['box_encodings'], axis=1)
      if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
        box_encodings = tf.squeeze(box_encodings, axis=2)
      class_predictions_with_background = tf.concat(
          pred_dict['class_predictions_with_background'], axis=1)
    detection_dict = postprocess(box_encodings, class_predictions_with_background,
                      anchors)
  return g


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

def test_model():
    g, _, _ = build_model()
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

        step = 0
        max_steps = int(FLAGS.num_examples / FLAGS.batch_size)
        print('START TESTING...')
        try:
          while not coord.should_stop():
            for _step in range(step+1, step+max_steps+1):
              # test

          if _step % 20 == 0:
        except tf.errors.OutOfRangeError:
          accuracy = 1 - len(errors_name)/FLAGS.num_examples
          print(time.strftime("%X"),
                'RESULT >>> current_acc:{0:.6f}'.format(accuracy))
        finally:
          coord.request_stop()
          coord.join(threads)
        print('FINISHED TESTING.')


def main(unused_arg):
    test_model()


if __name__ == '__main__':
  tf.app.run(main)
