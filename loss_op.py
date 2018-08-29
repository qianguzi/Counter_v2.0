import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

from detection_ops.utils import shape_utils
from detection_ops import target_assigner, box_coder, minibatch_sampler
from detection_ops import balanced_positive_negative_sampler as sampler
from detection_ops.utils.visualization_utils import add_cdf_image_summary
from detection_ops.losses import Loss, WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss, HardExampleMiner

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

def apply_hard_mining(location_losses, cls_losses, box_encodings,
                         anchors, match_list, hard_example_miner):
    """Applies hard mining to anchorwise losses.

    Args:
      location_losses: Float tensor of shape [batch_size, num_anchors]
        representing anchorwise location losses.
      cls_losses: Float tensor of shape [batch_size, num_anchors]
        representing anchorwise classification losses.
      prediction_dict: p a dictionary holding prediction tensors with
        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        2) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions.
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.

    Returns:
      mined_location_loss: a float scalar with sum of localization losses from
        selected hard examples.
      mined_cls_loss: a float scalar with sum of classification losses from
        selected hard examples.
    """
    decoded_boxes = box_coder.batch_decode(box_encodings, anchors)
    decoded_box_list = tf.unstack(decoded_boxes)

    return hard_example_miner(
        location_losses=location_losses,
        cls_losses=cls_losses,
        decoded_boxlist_list=decoded_box_list,
        match_list=match_list)

def minibatch_subsample_fn(inputs):
    """Randomly samples anchors for one image.

    Args:
      inputs: a list of 2 inputs. First one is a tensor of shape [num_anchors,
        num_classes] indicating targets assigned to each anchor. Second one
        is a tensor of shape [num_anchors] indicating the class weight of each
        anchor.

    Returns:
      batch_sampled_indicator: bool tensor of shape [num_anchors] indicating
        whether the anchor should be selected for loss computation.
    """
    cls_targets, cls_weights = inputs
    # Set background_class bits to 0 so that the positives_indicator
    # computation would not consider background class.
    background_class = tf.zeros_like(tf.slice(cls_targets, [0, 0], [-1, 1]))
    regular_class = tf.slice(cls_targets, [0, 1], [-1, -1])
    cls_targets = tf.concat([background_class, regular_class], 1)
    positives_indicator = tf.reduce_sum(cls_targets, axis=1)
    random_example_sampler = sampler.BalancedPositiveNegativeSampler()
    return random_example_sampler.subsample(
        tf.cast(cls_weights, tf.bool),
        batch_size=64,
        labels=tf.cast(positives_indicator, tf.bool))

def loss(box_encodings, class_predictions_with_background, 
         bbox_batch, anchors, matcher, num_batch,
         add_summaries=True,
         normalize_loss_by_num_matches=True,
         normalize_loc_loss_by_codesize=True,
         random_example=True,
         scope=None):
  """Compute scalar loss tensors with respect to provided groundtruth."""
  with tf.name_scope(scope, 'Loss', [box_encodings, class_predictions_with_background,
                     bbox_batch, anchors, matcher, num_batch]):
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)= target_assigner.target_assign(
                                              bbox_batch, anchors,
                                              matcher, num_batch)
      if random_example:
        batch_sampled_indicator = tf.to_float(
            shape_utils.static_or_dynamic_map_fn(
                minibatch_subsample_fn,
                [batch_cls_targets, batch_cls_weights],
                dtype=tf.bool,
                parallel_iterations=16,
                back_prop=True))
        batch_reg_weights = tf.multiply(batch_sampled_indicator,
                                        batch_reg_weights)
        batch_cls_weights = tf.multiply(batch_sampled_indicator,
                                        batch_cls_weights)
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
      hard_example_miner = HardExampleMiner()
      (localization_loss, classification_loss) = apply_hard_mining(
        location_losses, cls_losses, box_encodings, anchors, match_list, hard_example_miner)
      if add_summaries:
        hard_example_miner.summarize()
      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                1.0)

      localization_loss_normalizer = normalizer
      if normalize_loc_loss_by_codesize:
        localization_loss_normalizer *= 4
      localization_loss = tf.multiply((1 / localization_loss_normalizer),
                                      localization_loss,
                                      name='localization_loss')
      classification_loss = tf.multiply((1 / normalizer), 
                                        classification_loss,
                                        name='classification_loss')

      total_loss = localization_loss + classification_loss
  return total_loss
