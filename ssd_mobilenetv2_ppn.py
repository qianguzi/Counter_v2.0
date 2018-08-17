import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np

from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils, visualization_utils
from detection_ops.utils.visualization_utils import add_cdf_image_summary

from detection_ops import feature_map_generator, box_predictor, anchor_generator, argmax_matcher, target_assigner
from detection_ops.losses import Loss, WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss

tf.app.flags.DEFINE_string('master', '', 'Session master')
tf.app.flags.DEFINE_integer('task', 0, 'Task')
tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('number_of_steps', 4000,
                     'Number of training steps to perform before stopping')
tf.app.flags.DEFINE_integer('image_size', 96, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 0.50, 'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_bool('quantize', False, 'Quantize training')
tf.app.flags.DEFINE_string('fine_tune_checkpoint', '/home/myfile/dl_chrome/mobilenet_v2_0.5_96/mobilenet_v2_0.5_96.ckpt',
                    'Checkpoint from which to start finetuning.')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/',
                    'Directory for writing training checkpoints and logs')
tf.app.flags.DEFINE_string('dataset_dir', '../tfrecords/train.tfrecords', 'Location of dataset')
tf.app.flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
tf.app.flags.DEFINE_integer('save_summaries_secs', 10,
                     'How often to save summaries, secs')
tf.app.flags.DEFINE_integer('save_interval_secs', 300,
                     'How often to save checkpoints, secs')
tf.app.flags.DEFINE_bool('freeze_batchnorm', True,
                     'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('inplace_batchnorm_update', True,
                     'Whether to update batch norm moving average values inplace')
tf.app.flags.DEFINE_bool('is_training', True, 'train or eval')
tf.app.flags.DEFINE_bool('add_summaries', True, 'add summaries')
tf.app.flags.DEFINE_bool('normalize_loss_by_num_matches', True, 'normalize loss by num of matches')
tf.app.flags.DEFINE_bool('normalize_loc_loss_by_codesize', True, 'normalize loc loss by codesize')

FLAGS = tf.app.flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.98


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

def reduce_sum_trailing_dimensions(tensor, ndims):
  """Computes sum across all dimensions following first `ndims` dimensions."""
  return tf.reduce_sum(tensor, axis=tuple(range(ndims, tensor.shape.ndims)))

def summarize_anchor_classification_loss(class_ids, cls_losses):
    positive_indices = tf.where(tf.greater(class_ids, 0))
    positive_anchor_cls_loss = tf.squeeze(
        tf.gather(cls_losses, positive_indices), axis=1)
    add_cdf_image_summary(positive_anchor_cls_loss,
                                              'PositiveAnchorLossCDF')
    negative_indices = tf.where(tf.equal(class_ids, 0))
    negative_anchor_cls_loss = tf.squeeze(
        tf.gather(cls_losses, negative_indices), axis=1)
    add_cdf_image_summary(negative_anchor_cls_loss,
                                              'NegativeAnchorLossCDF')

def loss(box_encodings, class_predictions_with_background, 
         gt_box_batch, anchors, matcher):
  """Compute scalar loss tensors with respect to provided groundtruth."""
  with tf.name_scope('Loss', [box_encodings, class_predictions_with_background,
                     gt_box_batch, anchors, matcher]):
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)= target_assigner.target_assign(
                                              gt_box_batch, anchors, matcher)
      
      LocLoss = WeightedSmoothL1LocalizationLoss()
      ClsLoss = WeightedSoftmaxClassificationLoss()
      location_losses = LocLoss(box_encodings,
                                batch_reg_targets,
                                ignore_nan_targets=True,
                                scope='location_loss',
                                weights=batch_reg_weights)
      cls_losses = reduce_sum_trailing_dimensions(
          ClsLoss(class_predictions_with_background,
                  batch_cls_targets,
                  scope='cls_loss',
                  weights=batch_cls_weights),
          ndims=2)

      if FLAGS.add_summaries:
        class_ids = tf.argmax(batch_cls_targets, axis=2)
        flattened_class_ids = tf.reshape(class_ids, [-1])
        flattened_classification_losses = tf.reshape(cls_losses, [-1])
        summarize_anchor_classification_loss(
            flattened_class_ids, flattened_classification_losses)
      localization_loss = tf.reduce_sum(location_losses)
      classification_loss = tf.reduce_sum(cls_losses)

      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if FLAGS.normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                1.0)

      localization_loss_normalizer = normalizer
      if FLAGS.normalize_loc_loss_by_codesize:
        localization_loss_normalizer *= 4
      localization_loss = tf.multiply((1 / localization_loss_normalizer),
                                      localization_loss,
                                      name='localization_loss')
      classification_loss = tf.multiply((1 / normalizer), 
                                        classification_loss,
                                        name='classification_loss')

      loss_dict = {
          str(localization_loss.op.name): localization_loss,
          str(classification_loss.op.name): classification_loss
      }
  return loss_dict

def build_model():
  matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.7,
                                         unmatched_threshold=0.5)
  anchors = anchor_generator.generate_anchors(feature_map_dims=[(7, 7), (4, 4)],
                                              scales=[[0.75, 0.95], [0.70, 0.90]],
                                              aspect_ratios=[[1.0, 1.0], [1.0, 1.0]])
  box_pred = box_predictor.SSDBoxPredictor(
        FLAGS.is_training, FLAGS.num_classes, box_code_size=4, 
        conv_hyperparams_fn = _conv_hyperparams_fn)
  g = tf.Graph()
  with g.as_default():
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    inputs = tf.placeholder(tf.float32, [2, 224, 224, 3], 'Inputs')
    gt_box_0 = tf.placeholder(tf.float32, [5, 4], 'gt_boxes_0')
    gt_box_1 = tf.placeholder(tf.float32, [8, 4], 'gt_boxes_1')
    gt_box_batch = [gt_box_0, gt_box_1]
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32, name='anchors')
    with slim.arg_scope([slim.batch_norm],
            is_training=(FLAGS.is_training and not FLAGS.freeze_batchnorm),
            updates_collections=batchnorm_updates_collections),\
        slim.arg_scope(
            mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)):
      _, image_features = mobilenet_v2.mobilenet_base(
          inputs,
          final_endpoint='layer_19',
          depth_multiplier=FLAGS.depth_multiplier)
      feature_maps = feature_map_generator.pooling_pyramid_feature_maps(
          base_feature_map_depth=0,
          num_layers=2,
          image_features={  
              'image_features': image_features['layer_19']
          })
      pred_dict = box_pred.predict(feature_maps.values(), [2, 2])
      box_encodings = tf.concat(pred_dict['box_encodings'], axis=1)
      if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
        box_encodings = tf.squeeze(box_encodings, axis=2)
      class_predictions_with_background = tf.concat(
          pred_dict['class_predictions_with_background'], axis=1)
    loss_dict = loss(box_encodings, class_predictions_with_background,
                     gt_box_batch, anchors, matcher)

  return g


def main(unused_arg):
  g = build_model()


if __name__ == '__main__':
  tf.app.run(main)