import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim


def pooling_pyramid_feature_maps(base_feature_map_depth, num_layers,
                                 image_features):
  """Generates pooling pyramid feature maps.

  The pooling pyramid feature maps is motivated by
  multi_resolution_feature_maps. The main difference are that it is simpler and
  reduces the number of free parameters.

  More specifically:
   - Instead of using convolutions to shrink the feature map, it uses max
     pooling, therefore totally gets rid of the parameters in convolution.
   - By pooling feature from larger map up to a single cell, it generates
     features in the same feature space.
   - Instead of independently making box predictions from individual maps, it
     shares the same classifier across different feature maps, therefore reduces
     the "mis-calibration" across different scales.

  See go/ppn-detection for more details.

  Args:
    base_feature_map_depth: Depth of the base feature before the max pooling.
    num_layers: Number of layers used to make predictions. They are pooled
      from the base feature.
    image_features: A dictionary of handles to activation tensors from the
      feature extractor.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  Raises:
    ValueError: image_features does not contain exactly one entry
  """
  if len(image_features) != 1:
    raise ValueError('image_features should be a dictionary of length 1.')
  image_features = image_features[image_features.keys()[0]]

  feature_map_keys = []
  feature_maps = []
  feature_map_key = 'Base_Conv2d_1x1_%d' % base_feature_map_depth
  if base_feature_map_depth > 0:
    image_features = slim.conv2d(
        image_features,
        base_feature_map_depth,
        [1, 1],  # kernel size
        padding='SAME', stride=1, scope=feature_map_key)
    # Add a 1x1 max-pooling node (a no op node) immediately after the conv2d for
    # TPU v1 compatibility.  Without the following dummy op, TPU runtime
    # compiler will combine the convolution with one max-pooling below into a
    # single cycle, so getting the conv2d feature becomes impossible.
    image_features = slim.max_pool2d(
        image_features, [1, 1], padding='SAME', stride=1, scope=feature_map_key)
  feature_map_keys.append(feature_map_key)
  feature_maps.append(image_features)
  feature_map = image_features
  with slim.arg_scope([slim.max_pool2d], padding='SAME', stride=2):
    for i in range(num_layers - 1):
      feature_map_key = 'MaxPool2d_%d_2x2' % i
      feature_map = slim.max_pool2d(
          feature_map, [2, 2], padding='SAME', scope=feature_map_key)
      feature_map_keys.append(feature_map_key)
      feature_maps.append(feature_map)
  return collections.OrderedDict(
      [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])