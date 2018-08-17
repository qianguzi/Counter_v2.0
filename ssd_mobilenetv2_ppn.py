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
tf.app.flags.DEFINE_integer('batch_size', 2, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('number_of_steps', 6000,
                     'Number of training steps to perform before stopping')
tf.app.flags.DEFINE_integer('image_size', 256, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 0.50, 'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate for detection model')
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
         gt_box_batch, anchors, matcher, num_gt_boxes):
  """Compute scalar loss tensors with respect to provided groundtruth."""
  with tf.name_scope('Loss', [box_encodings, class_predictions_with_background,
                     gt_box_batch, anchors, matcher]):
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)= target_assigner.target_assign(
                                              gt_box_batch, anchors,
                                              matcher,num_gt_boxes)
      
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

      total_loss = localization_loss + classification_loss
  return total_loss


def build_model():
  matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.7,
                                         unmatched_threshold=0.5)
  anchors = anchor_generator.generate_anchors(feature_map_dims=[(8, 8), (4, 4)],
                                              scales=[[0.75, 0.95], [0.50, 0.80]],
                                              aspect_ratios=[[1.0, 1.0], [1.0, 1.0]])
  box_pred = box_predictor.SSDBoxPredictor(
        FLAGS.is_training, FLAGS.num_classes, box_code_size=4, 
        conv_hyperparams_fn = _conv_hyperparams_fn)
  g = tf.Graph()
  with g.as_default():
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    inputs = tf.placeholder(tf.float32, [2, 256, 256, 3], 'Inputs')
    gt_box_batch = tf.constant([[[0.232,0.156,0.432,0.356],[0.115,0.456,0.345,0.789]],[[0.345,0.678,0.698,0.835],[0.0,0.0,0.0,0.0]]],
                                dtype=tf.float32, name='boxes')
    num_gt_boxes = tf.constant([2, 1], dtype=tf.int32, name='num_boxes')
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
    total_loss = loss(box_encodings, class_predictions_with_background,
                     gt_box_batch, anchors, matcher, num_gt_boxes)
    # Configure the learning rate using an exponential decay.
    total_loss = tf.identity(total_loss, name='total_loss')
    num_epochs_per_decay = 1
    imagenet_size = 3000
    decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf.train.get_or_create_global_step(), 
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=False)
    #opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
    train_tensor = slim.learning.create_train_op(
        total_loss,
        optimizer=opt)

  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
  return g, train_tensor


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    variables_to_restore = slim.get_variables_to_restore(include=['MobilenetV2/'])
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    # When restoring from a floating point model, the min/max values for
    # quantized weights and activations are not present.
    # We instruct slim to ignore variables that are missing during restoration
    # by setting ignore_missing_vars=True
    slim_init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.fine_tune_checkpoint,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
      # If we are restoring from a floating point model, we need to initialize
      # the global step to zero for the exponential decay to result in
      # reasonable learning rates.
      sess.run(global_step_reset)
    return init_fn
  else:
    return None


def train_model():
  """Trains ssd_mobilenet_v2_ppn."""
  g, train_tensor = build_model()
  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.checkpoint_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step())


def main(unused_arg):
  g = build_model()


if __name__ == '__main__':
  tf.app.run(main)