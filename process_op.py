import tensorflow as tf
from detection_ops.utils import shape_utils
from detection_ops import box_coder

def postprocess(anchors, box_encodings,
                class_predictions_with_background,
                convert_ratio=None,
                clip_windows_tl=None,
                num_classes=2,
                scope=None):
    """Converts prediction tensors to detections."""
    with tf.name_scope(scope, 'Postprocessor', [num_classes]):
        box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
        detection_boxes = box_coder.batch_decode(box_encodings, anchors)
        if convert_ratio is not None:
            detection_boxes = tf.multiply(
                detection_boxes, convert_ratio) + clip_windows_tl
        detection_boxes = tf.reshape(detection_boxes, [-1, 4], 'raw_box_locations')

        class_predictions_with_background = tf.reshape(
            class_predictions_with_background, [-1, num_classes])
        detection_scores_with_background = tf.nn.softmax(
            class_predictions_with_background, name='raw_box_scores')

        return detection_boxes, detection_scores_with_background


def precise_filter(detection_boxes, 
                    detection_scores_with_background, 
                    detection_scores=None, 
                    max_output_size=70,
                    iou_threshold=0.05,
                    score_threshold=0.6,
                    high_score_screening=False,
                    non_max_suppression_screening=True,
                    scope=None):
    with tf.name_scope(
        scope, 'Precise_filter', [max_output_size, iou_threshold, score_threshold]):
        if detection_scores is None:
            detection_scores = tf.slice(detection_scores_with_background, [0, 1],[-1, -1])
            detection_scores = tf.squeeze(detection_scores, 1)
        
        positive_indices = tf.equal(
            tf.argmax(detection_scores_with_background, 1),
            1)
        positive_indices = tf.squeeze(tf.where(positive_indices), 1)
        result_boxes = tf.gather(detection_boxes, positive_indices)
        result_scores = tf.gather(detection_scores, positive_indices)

        if high_score_screening:
            high_score_indices = tf.reshape(
                tf.where(tf.greater(result_scores, score_threshold)),
                [-1])
            result_boxes = tf.gather(result_boxes, high_score_indices)
            result_scores = tf.gather(result_scores, high_score_indices)
        if non_max_suppression_screening:
            selected_indices = tf.image.non_max_suppression(
                result_boxes,
                result_scores,
                max_output_size,
                iou_threshold)
            result_boxes = tf.gather(result_boxes, selected_indices)
            result_scores = tf.gather(result_scores, selected_indices)

        result_boxes = tf.identity(result_boxes, name='result_boxes')
        result_scores = tf.identity(result_scores, name='result_scores')
        result_dict = {
            'result_boxes': result_boxes,
            'result_scores': result_scores
        }
        return result_dict


def crop_and_resize(img_tensor, grid_points, clip_size, scope=None):
    with tf.name_scope(scope, 'Crop_and_resize', [img_tensor, grid_points, clip_size]):
        num_batch = shape_utils.combined_static_and_dynamic_shape(
            grid_points)
        box_idx = tf.zeros(num_batch[0], tf.int32)
        preimg_batch = tf.image.crop_and_resize(
            img_tensor, grid_points, box_idx, [clip_size, clip_size])
        return preimg_batch


def generate_grid(img_shape, grid_size_tensor, 
                  clip_size, value_to_ratio,
                  apply_or_model=False, scope=None):
    with tf.name_scope(scope, 'Grid_builder',
                        [img_shape, grid_size_tensor, clip_size, value_to_ratio]):
        h, w = img_shape[1:3]
        half_clip_size = tf.cast(clip_size / 2, tf.float32)
        dy = tf.cast((h-clip_size) / grid_size_tensor[0], tf.float32)
        dx = tf.cast((w-clip_size) / grid_size_tensor[1], tf.float32)
        if apply_or_model:
            y = tf.range(half_clip_size+dy, h-half_clip_size-1, dy, tf.float32)
            x = tf.range(half_clip_size+dx, w-half_clip_size-1, dx, tf.float32)
        else:
            y = tf.range(half_clip_size, h-half_clip_size-1, dy, tf.float32)
            x = tf.range(half_clip_size, w-half_clip_size-1, dx, tf.float32)
            y = tf.concat([y, [h-half_clip_size]], 0)
            x = tf.concat([x, [w-half_clip_size]], 0)
        centers = tf.meshgrid(y, x)
        centers = tf.stack(centers, axis=2)
        centers = tf.reshape(centers, [-1, 2])
        grid_points = tf.concat(
            [centers - half_clip_size, centers + half_clip_size], 1)
        grid_points = tf.multiply(grid_points, value_to_ratio)
        grid_points = tf.cast(grid_points, tf.float32, name='grid_points')
        return grid_points


def preprocess(img_tensor, grid_size_tensor, 
              clip_size, value_to_ratio,
              apply_or_model=False, scope=None):
    with tf.name_scope(scope, 'Preprocessor',
                        [img_tensor, grid_size_tensor, clip_size, value_to_ratio]):
        img_shape = shape_utils.combined_static_and_dynamic_shape(img_tensor)
        grid_points = generate_grid(
          img_shape, grid_size_tensor, clip_size, value_to_ratio, apply_or_model)
        grid_points_tl = tf.stack([grid_points[:, 1], grid_points[:, 0]], 1)
        grid_points_tl = tf.tile(grid_points_tl, [1, 2])
        grid_points_tl = tf.expand_dims(grid_points_tl, 1)
        preimg_batch = crop_and_resize(img_tensor, grid_points, clip_size)
        return preimg_batch, grid_points_tl