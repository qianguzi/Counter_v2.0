import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import loss_op
from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils, data_ops
from detection_ops import feature_map_generator, box_predictor, anchor_generator, argmax_matcher

tf.app.flags.DEFINE_string('master', '', 'Session master')
tf.app.flags.DEFINE_integer('task', 0, 'Task')
tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('number_of_steps', 30000,
                     'Number of training steps to perform before stopping')
tf.app.flags.DEFINE_integer('image_size', 320, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate for detection model')
tf.app.flags.DEFINE_string('fine_tune_checkpoint',
                    None,
                    'Checkpoint from which to start finetuning.')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/',
                    'Directory for writing training checkpoints and logs')
tf.app.flags.DEFINE_string('dataset_dir', '../tfrecords/train.tfrecords', 'Location of dataset')
tf.app.flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
tf.app.flags.DEFINE_integer('save_summaries_secs', 50,
                     'How often to save summaries, secs')
tf.app.flags.DEFINE_integer('save_interval_secs', 300,
                     'How often to save checkpoints, secs')
tf.app.flags.DEFINE_bool('freeze_batchnorm', False,
                     'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('inplace_batchnorm_update', True,
                     'Whether to update batch norm moving average values inplace')
tf.app.flags.DEFINE_bool('is_training', True, 'train or eval')

FLAGS = tf.app.flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.90

_anchors_figure = {
  'feature_map_dims' : [(10, 10), (5, 5)],
  'scales' : [[2.0], [1.0]],
  'aspect_ratios' : [[1.0], [1.0]]
}


def build_model():
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5,
                                         unmatched_threshold=0.4)
    anchors = anchor_generator.generate_anchors(**_anchors_figure)
    box_pred = box_predictor.SSDBoxPredictor(
      FLAGS.is_training, FLAGS.num_classes, box_code_size=4)
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    img_batch, bbox_batch, num_batch, _ = data_ops.get_batch(FLAGS.dataset_dir,
                                                             FLAGS.batch_size)
    preimg_batch = tf.cast(img_batch, tf.float32) / 127.5 - 1
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32, name='anchors')
    with slim.arg_scope([slim.batch_norm],
            is_training=(FLAGS.is_training and not FLAGS.freeze_batchnorm),
            updates_collections=batchnorm_updates_collections),\
        slim.arg_scope(
            mobilenet_v2.training_scope(is_training=None, bn_decay=0.997)):
      _, image_features = mobilenet_v2.mobilenet_base(
          preimg_batch,
          final_endpoint='layer_18',
          depth_multiplier=FLAGS.depth_multiplier,
          finegrain_classification_mode=True)

      scales = _anchors_figure['scales']
      feature_maps = feature_map_generator.pooling_pyramid_feature_maps(
          base_feature_map_depth=0,
          num_layers=len(scales),
          image_features={  
              'image_features': image_features['layer_18']
          })
      
      pred_dict = box_pred.predict(feature_maps.values(),
                                  [len(scales[0]), len(scales[1])])
      box_encodings = tf.concat(pred_dict['box_encodings'], axis=1)
      if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
        box_encodings = tf.squeeze(box_encodings, axis=2)
      class_predictions_with_background = tf.concat(
          pred_dict['class_predictions_with_background'], axis=1)

    total_loss = loss_op.loss(box_encodings, class_predictions_with_background,
                     bbox_batch, anchors, matcher, num_batch)
    # Configure the learning rate using an exponential decay.
    total_loss = tf.identity(total_loss, name='total_loss')
    num_epochs_per_decay = 2
    imagenet_size = 5400
    decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf.train.get_or_create_global_step(), 
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)

    opt = tf.train.AdamOptimizer(learning_rate)
    train_tensor = slim.learning.create_train_op(
        total_loss,
        optimizer=opt)

  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
  return g, train_tensor


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    variables_to_restore = slim.get_variables_to_restore()
    #global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
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
      #sess.run(global_step_reset)
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
        global_step=tf.train.get_global_step(),
        saver=tf.train.Saver(max_to_keep=100))


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)