# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

import tensorflow as tf

EPSILON = 1e-8

def get_center_coordinates_and_sizes(box_corners, scope=None):
    """Computes the center coordinates, height and width of the boxes.

    Args:
      scope: name scope of the function.

    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
      ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
      width = xmax - xmin
      height = ymax - ymin
      ycenter = ymin + height / 2.
      xcenter = xmin + width / 2.
      return [ycenter, xcenter, height, width]

def encode(boxes, anchors, scale_factors=True):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)
    ycenter, xcenter, h, w = get_center_coordinates_and_sizes(boxes)
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    # Scales location targets as used in paper for joint training.
    if scale_factors:
      ty *= 10.0
      tx *= 10.0
      th *= 5.0
      tw *= 5.0
    return tf.transpose(tf.stack([ty, tx, th, tw]))

def decode(rel_codes, anchors, scale_factors=True):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    if scale_factors:
      ty /= 10.0
      tx /= 10.0
      th /= 10.0
      tw /= 10.0
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
