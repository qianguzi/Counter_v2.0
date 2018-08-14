import tensorflow as tf
import tensorflow.contrib.slim as slim

from mobilenet import mobilenet_v2
from detection_ops import feature_map_generator
from detection_ops.utils import shape_utils

tf.app.flags.DEFINE_string('master', '', 'Session master')
tf.app.flags.DEFINE_integer('task', 0, 'Task')
tf.app.flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('number_of_steps', 4000,
                     'Number of training steps to perform before stopping')
tf.app.flags.DEFINE_integer('image_size', 96, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 0.5, 'Depth multiplier for mobilenet')
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
tf.app.flags.DEFINE_bool('freeze_batchnorm', True,
                     'Whether to freeze batch norm parameters during training or not')
tf.app.flags.DEFINE_bool('is_training', True, 'train or eval')

FLAGS = tf.app.flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.98


def _get_feature_map_spatial_dims(self, feature_maps):
  """Return list of spatial dimensions for each feature map in a list.

  Args:
    feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i].

  Returns:
    a list of pairs (height, width) for each feature map in feature_maps
  """
  feature_map_shapes = [
      shape_utils.combined_static_and_dynamic_shape(
          feature_map) for feature_map in feature_maps
  ]
  return [(shape[1], shape[2]) for shape in feature_map_shapes]


def build_model():
  g = tf.Graph()
  with g.as_default():
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    inputs, labels = ....
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
      #######################generate_anchors#################################
      feature_map_spatial_dims = _get_feature_map_spatial_dims(
          feature_maps.values())
      image_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
      self._anchors = box_list_ops.concatenate(
          self._anchor_generator.generate(
              feature_map_spatial_dims,
              im_height=image_shape[1],
              im_width=image_shape[2]))
      ############################--待修改--###################################
      