import tensorflow as tf
import tensorflow.contrib.slim as slim

from abc import abstractmethod
from detection_ops.utils import shape_utils, static_shape

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'

class BoxPredictor(object):
  """BoxPredictor."""

  def __init__(self, is_training, num_classes):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
    """
    self._is_training = is_training
    self._num_classes = num_classes

  @property
  def num_classes(self):
    return self._num_classes

  def predict(self, image_features, num_predictions_per_location,
              scope=None, **params):
    """Computes encoded object locations and corresponding confidences.

    Takes a list of high level image feature maps as input and produces a list
    of box encodings and a list of class scores where each element in the output
    lists correspond to the feature maps in the input list.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      scope: Variable and Op scope name.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.

    Raises:
      ValueError: If length of `image_features` is not equal to length of
        `num_predictions_per_location`.
    """
    if len(image_features) != len(num_predictions_per_location):
      raise ValueError('image_feature and num_predictions_per_location must '
                       'be of same length, found: {} vs {}'.
                       format(len(image_features),
                              len(num_predictions_per_location)))
    if scope is not None:
      with tf.variable_scope(scope):
        return self._predict(image_features, num_predictions_per_location,
                             **params)
    return self._predict(image_features, num_predictions_per_location,
                         **params)

  # TODO(rathodv): num_predictions_per_location could be moved to constructor.
  # This is currently only used by ConvolutionalBoxPredictor.
  @abstractmethod
  def _predict(self, image_features, num_predictions_per_location, **params):
    """Implementations must override this method.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
    """
    pass


class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class SSDBoxPredictor(BoxPredictor):
  """Convolutional Box Predictor.

  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.

  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_code_size,
               conv_hyperparams_fn,
               min_depth=0,
               max_depth=0,
               num_layers_before_predictor=0,
               use_dropout=True,
               kernel_size=1,
               apply_sigmoid_to_scores=False,
               class_prediction_bias_init=0.0,
               use_depthwise=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      min_depth: Minimum feature depth prior to predicting box encodings
        and class predictions.
      max_depth: Maximum feature depth prior to predicting box encodings
        and class predictions. If max_depth is set to 0, no additional
        feature map will be inserted before location and class predictions.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      use_dropout: Option to use dropout for class prediction or not.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      box_code_size: Size of encoding for each box.
      apply_sigmoid_to_scores: if True, apply the sigmoid on the output
        class_predictions.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(SSDBoxPredictor, self).__init__(is_training, num_classes)
    if min_depth > max_depth:
      raise ValueError('min_depth should be less than or equal to max_depth')
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._use_dropout = use_dropout
    self._kernel_size = kernel_size
    self._box_code_size = box_code_size
    self._apply_sigmoid_to_scores = apply_sigmoid_to_scores
    self._class_prediction_bias_init = class_prediction_bias_init
    self._use_depthwise = use_depthwise

  def _predict(self, image_features, num_predictions_per_location_list):
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
        num_class_slots = self.num_classes
        net = image_feature
        with slim.arg_scope(self._conv_hyperparams_fn()):
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
              net = slim.dropout(net, keep_prob=0.8)
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