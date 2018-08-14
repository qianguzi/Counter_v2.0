import tensorflow as tf
import tensorflow.contrib.slim as slim

from detection_ops.utils import shape_utils

class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


def ssd_box_predict(image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """
    box_encodings_list = []
    class_predictions_list = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]

    for (image_feature,
         num_predictions_per_location, box_predictor_scope) in zip(
             image_features, num_predictions_per_location_list,
             box_predictor_scopes):
      with box_predictor_scope:
        # Add a slot for the background class.
        num_class_slots = self.num_classes + 1
        net = image_feature
        with slim.arg_scope(self._conv_hyperparams_fn()), \
             slim.arg_scope([slim.dropout], is_training=self._is_training):
          # Add additional conv layers before the class predictor.
          features_depth = static_shape.get_depth(image_feature.get_shape())
          depth = max(min(features_depth, self._max_depth), self._min_depth)
          tf.logging.info('depth of additional conv before box predictor: {}'.
                          format(depth))
          if depth > 0 and self._num_layers_before_predictor > 0:
            for i in range(self._num_layers_before_predictor):
              net = slim.conv2d(
                  net, depth, [1, 1], scope='Conv2d_%d_1x1_%d' % (i, depth))
          with slim.arg_scope([slim.conv2d], activation_fn=None,
                              normalizer_fn=None, normalizer_params=None):
            if self._use_depthwise:
              box_encodings = slim.separable_conv2d(
                  net, None, [self._kernel_size, self._kernel_size],
                  padding='SAME', depth_multiplier=1, stride=1,
                  rate=1, scope='BoxEncodingPredictor_depthwise')
              box_encodings = slim.conv2d(
                  box_encodings,
                  num_predictions_per_location * self._box_code_size, [1, 1],
                  scope='BoxEncodingPredictor')
            else:
              box_encodings = slim.conv2d(
                  net, num_predictions_per_location * self._box_code_size,
                  [self._kernel_size, self._kernel_size],
                  scope='BoxEncodingPredictor')
            if self._use_dropout:
              net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
            if self._use_depthwise:
              class_predictions_with_background = slim.separable_conv2d(
                  net, None, [self._kernel_size, self._kernel_size],
                  padding='SAME', depth_multiplier=1, stride=1,
                  rate=1, scope='ClassPredictor_depthwise')
              class_predictions_with_background = slim.conv2d(
                  class_predictions_with_background,
                  num_predictions_per_location * num_class_slots,
                  [1, 1], scope='ClassPredictor')
            else:
              class_predictions_with_background = slim.conv2d(
                  net, num_predictions_per_location * num_class_slots,
                  [self._kernel_size, self._kernel_size],
                  scope='ClassPredictor',
                  biases_initializer=tf.constant_initializer(
                      self._class_prediction_bias_init))
            if self._apply_sigmoid_to_scores:
              class_predictions_with_background = tf.sigmoid(
                  class_predictions_with_background)

        combined_feature_map_shape = (shape_utils.
                                      combined_static_and_dynamic_shape(
                                          image_feature))
        box_encodings = tf.reshape(
            box_encodings, tf.stack([combined_feature_map_shape[0],
                                     combined_feature_map_shape[1] *
                                     combined_feature_map_shape[2] *
                                     num_predictions_per_location,
                                     1, self._box_code_size]))
        box_encodings_list.append(box_encodings)
        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))
        class_predictions_list.append(class_predictions_with_background)
    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }