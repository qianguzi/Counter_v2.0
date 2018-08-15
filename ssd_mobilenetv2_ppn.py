import tensorflow as tf
import tensorflow.contrib.slim as slim

from mobilenet import mobilenet_v2
from detection_ops.utils import shape_utils
from detection_ops import feature_map_generator, box_predictor
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
  box_pred = box_predictor.SSDBoxPredictor(
        FLAGS.is_training, FLAGS.num_classes, box_code_size=4, 
        conv_hyperparams_fn = _conv_hyperparams_fn)
  g = tf.Graph()
  with g.as_default():
    batchnorm_updates_collections = (None if FLAGS.inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    inputs = tf.placeholder(tf.float32, [2, 224, 224, 3], 'Inputs')
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
    #  feature_map_spatial_dims = _get_feature_map_spatial_dims(
   #       feature_maps.values())
   #   image_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
   #   self._anchors = box_list_ops.concatenate(
   #       self._anchor_generator.generate(
#              feature_map_spatial_dims,
  #            im_height=image_shape[1],
  #            im_width=image_shape[2]))
      ############################--待修改--###################################
      pred_dict = box_pred.predict(feature_maps.values(), [2, 2])
      box_encodings = tf.concat(pred_dict['box_encodings'], axis=1)
      if box_encodings.shape.ndims == 4 and box_encodings.shape[2] == 1:
        box_encodings = tf.squeeze(box_encodings, axis=2)
      class_predictions_with_background = tf.concat(
          pred_dict['class_predictions_with_background'], axis=1)
    with tf.name_scope('Loss', [box_encodings, class_predictions_with_background]):
    ########################gt vs anchors###############################
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, match_list) = self._assign_targets(
           self.groundtruth_lists(fields.BoxListFields.boxes),
           self.groundtruth_lists(fields.BoxListFields.classes),
           keypoints, weights)
      if self._add_summaries:
        self._summarize_target_assignment(
            self.groundtruth_lists(fields.BoxListFields.boxes), match_list)

##################################################################
      location_losses = WeightedSmoothL1LocalizationLoss(
          box_encodings,
          batch_reg_targets,
          ignore_nan_targets=True,
          scope='location_loss',
          weights=batch_reg_weights)
      cls_losses = reduce_sum_trailing_dimensions(
          WeightedSoftmaxClassificationLoss(
              class_predictions_with_background,
              batch_cls_targets,
              scope='cls_loss'
              weights=batch_cls_weights),
          ndims=2)
#--------------------------------------
      if self._hard_example_miner:
        (localization_loss, classification_loss) = self._apply_hard_mining(
            location_losses, cls_losses, pred_dict, match_list)
        if self._add_summaries:
          self._hard_example_miner.summarize()
      else:
        if self._add_summaries:
          class_ids = tf.argmax(batch_cls_targets, axis=2)
          flattened_class_ids = tf.reshape(class_ids, [-1])
          flattened_classification_losses = tf.reshape(cls_losses, [-1])
          self._summarize_anchor_classification_loss(
              flattened_class_ids, flattened_classification_losses)
        localization_loss = tf.reduce_sum(location_losses)
        classification_loss = tf.reduce_sum(cls_losses)

      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if self._normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                1.0)

      localization_loss_normalizer = normalizer
      if self._normalize_loc_loss_by_codesize:
        localization_loss_normalizer *= self._box_coder.code_size
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


def main(unused_arg):
  build_model()


if __name__ == '__main__':
  tf.app.run(main)