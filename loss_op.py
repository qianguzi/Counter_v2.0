import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

from detection_ops import target_assigner
from detection_ops.utils.visualization_utils import add_cdf_image_summary
from detection_ops.losses import Loss, WeightedSmoothL1LocalizationLoss, WeightedSoftmaxClassificationLoss

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
         bbox_batch, anchors, matcher, num_batch,
         add_summaries=False,
         normalize_loss_by_num_matches=True,
         normalize_loc_loss_by_codesize=True,
         scope=None):
  """Compute scalar loss tensors with respect to provided groundtruth."""
  with tf.name_scope(scope, 'Loss', [box_encodings, class_predictions_with_background,
                     bbox_batch, anchors, matcher, num_batch]):
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)= target_assigner.target_assign(
                                              bbox_batch, anchors,
                                              matcher, num_batch)
      
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

      if add_summaries:
        class_ids = tf.argmax(batch_cls_targets, axis=2)
        flattened_class_ids = tf.reshape(class_ids, [-1])
        flattened_classification_losses = tf.reshape(cls_losses, [-1])
        summarize_anchor_classification_loss(
            flattened_class_ids, flattened_classification_losses)
      localization_loss = tf.reduce_sum(location_losses)
      classification_loss = tf.reduce_sum(cls_losses)

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