import tensorflow as tf
from detection_ops import box_coder
from detection_ops.utils import shape_utils


def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: Boxlist holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: Boxlist holding N boxes
    boxlist2: Boxlist holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope(scope, 'Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: Boxlist holding N boxes
    boxlist2: Boxlist holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def assign(gt_boxes, anchors,
           matcher,
           gt_labels=None,
           gt_weights=None):
  gt_boxes = tf.identity(gt_boxes, name='gt_boxes')
  num_batch = shape_utils.combined_static_and_dynamic_shape(gt_boxes)
  if gt_labels is None:
    gt_labels = tf.ones([num_batch[0]], dtype=tf.float32)
    gt_labels = tf.expand_dims(gt_labels, -1)
    gt_labels = tf.pad(gt_labels, [[0, 0], [1, 0]], mode='CONSTANT')
  if gt_weights is None:
    gt_weights = tf.ones([num_batch[0]], dtype=tf.float32)
  
  match_quality_matrix = iou(gt_boxes, anchors)
  match = matcher.match(match_quality_matrix)
  reg_targets = create_regression_targets(anchors,
                                           gt_boxes,
                                           match)
  cls_targets = create_classification_targets(gt_labels,
                                               match)
  reg_weights = create_regression_weights(match, gt_weights)
  cls_weights = create_classification_weights(match,
                                               gt_weights)
  print(reg_targets, cls_targets, reg_weights, cls_weights)
  num_anchors = anchors.get_shape()[0].value
  if num_anchors is not None:
    reg_targets = reset_target_shape(reg_targets, num_anchors)
    cls_targets = reset_target_shape(cls_targets, num_anchors)
    reg_weights = reset_target_shape(reg_weights, num_anchors)
    cls_weights = reset_target_shape(cls_weights, num_anchors)
  
  return cls_targets, cls_weights, reg_targets, reg_weights, match
  
def reset_target_shape(target, num_anchors):
  """Sets the static shape of the target.

  Args:
    target: the target tensor. Its first dimension will be overwritten.
    num_anchors: the number of anchors, which is used to override the target's
      first dimension.

    Returns:
    A tensor with the shape info filled in.
  """
  target_shape = target.get_shape().as_list()
  target_shape[0] = num_anchors
  target.set_shape(target_shape)
  return target

def create_regression_targets(anchors, groundtruth_boxes, match):
    """Returns a regression target for each anchor.

    Args:
      anchors: a BoxList representing N anchors
      groundtruth_boxes: a BoxList representing M groundtruth_boxes
      match: a matcher.Match object

    Returns:
      reg_targets: a float32 tensor with shape [N, box_code_dimension]
    """
    matched_gt_boxes = match.gather_based_on_match(
        groundtruth_boxes,
        unmatched_value=tf.zeros(4),
        ignored_value=tf.zeros(4))
    matched_reg_targets = box_coder.encode(matched_gt_boxes, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(
        match.match_results)

    # Zero out the unmatched and ignored regression targets.
    unmatched_ignored_reg_targets = tf.tile(
        default_regression_target(), [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    reg_targets = tf.where(matched_anchors_mask,
                           matched_reg_targets,
                           unmatched_ignored_reg_targets)
    return reg_targets

def default_regression_target():
    """Returns the default target for anchors to regress to.

    Default regression targets are set to zero (though in
    this implementation what these targets are set to should
    not matter as the regression weight of any box set to
    regress to the default target is zero).

    Returns:
      default_target: a float32 tensor with shape [1, box_code_dimension]
    """
    return tf.constant([4*[0]], tf.float32)

def create_classification_targets(groundtruth_labels, match):
    """Create classification targets for each anchor.

    Assign a classification target of for each anchor to the matching
    groundtruth label that is provided by match.  Anchors that are not matched
    to anything are given the target self._unmatched_cls_target

    Args:
      groundtruth_labels:  a tensor of shape [num_batch, d_1, ... d_k]
        with labels for each of the ground_truth boxes. The subshape
        [d_1, ... d_k] can be empty (corresponding to scalar labels).
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.

    Returns:
      a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
      subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
      shape [num_batch, d_1, d_2, ... d_k].
    """
    return match.gather_based_on_match(
        groundtruth_labels,
        unmatched_value=tf.constant([1,0], tf.float32),
        ignored_value=tf.constant([0,0], tf.float32))

def create_regression_weights(match, groundtruth_weights):
    """Set regression weight for each anchor.

    Only positive anchors are set to contribute to the regression loss, so this
    method returns a weight of 1 for every positive anchor and 0 for every
    negative anchor.

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing regression weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights, ignored_value=0., unmatched_value=0.)

def create_classification_weights(match,
                                  groundtruth_weights):
    """Create classification weights for each anchor.

    Positive (matched) anchors are associated with a weight of
    positive_class_weight and negative (unmatched) anchors are associated with
    a weight of negative_class_weight. When anchors are ignored, weights are set
    to zero. By default, both positive/negative weights are set to 1.0,
    but they can be adjusted to handle class imbalance (which is almost always
    the case in object detection).

    Args:
      match: a matcher.Match object that provides a matching between anchors
        and groundtruth boxes.
      groundtruth_weights: a float tensor of shape [M] indicating the weight to
        assign to all anchors match to a particular groundtruth box.

    Returns:
      a float32 tensor with shape [num_anchors] representing classification
      weights.
    """
    return match.gather_based_on_match(
        groundtruth_weights,
        ignored_value=0.,
        unmatched_value=1.0)

def target_assign(bbox_batch, anchors,
                  matcher, num_batch):
  cls_targets_list = []
  cls_weights_list = []
  reg_targets_list = []
  reg_weights_list = []
  match_list = []
  batch_size = shape_utils.combined_static_and_dynamic_shape(num_batch)[0]
  for i in range(batch_size):
    (cls_targets, cls_weights, reg_targets,
     reg_weights, match) = assign(bbox_batch[i][:num_batch[i]], anchors,
                                  matcher)
    cls_targets_list.append(cls_targets)
    cls_weights_list.append(cls_weights)
    reg_targets_list.append(reg_targets)
    reg_weights_list.append(reg_weights)
    match_list.append(match)
  batch_cls_targets = tf.stack(cls_targets_list)
  batch_cls_weights = tf.stack(cls_weights_list)
  batch_reg_targets = tf.stack(reg_targets_list)
  batch_reg_weights = tf.stack(reg_weights_list)
  print(batch_cls_targets)
  return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
          batch_reg_weights, match_list)