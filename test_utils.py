import os, cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_checkpoint_init_fn(fine_tune_checkpoint, include_var=None, exclude_var=None):
    """Returns the checkpoint init_fn if the checkpoint is provided."""
    variables_to_restore = slim.get_variables_to_restore(include_var, exclude_var)
    slim_init_fn = slim.assign_from_checkpoint_fn(
        fine_tune_checkpoint,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
    return init_fn


def read_pb_model(pb_path, graph_def, return_elements):
    """Read pb file to get input/output tensors.
    
    Args:
        pb_path: the path of pb file.
        graph_def: a `GraphDef` proto containing operations to be imported into
        the default graph.
        return_elemments: a list of strings containing operation names in
        `graph_def` that will be returned as `Operation` objects; and/or
        tensor names in `graph_def` that will be returned as `Tensor` objects.
    Returns:
        A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`.
    """
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        return_tensors = tf.import_graph_def(graph_def, return_elements=return_elements)
        return return_tensors


def write_pb_model(pb_path, sess, graph_def, output_node_names):
    """Write model as pb file."""
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, output_node_names)
    with tf.gfile.GFile(pb_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def load(sess, saver, checkpoint_dir):
    """Load model variable from checkpoint file."""
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("[*] Success to read {}".format(ckpt_name))
        return ckpt_name
    else:
        raise Exception("[*] Failed to find a checkpoint")


def hough_cir_detection(img, minDist=55, precise=35, minRadius=10, maxRadius=30, 
                        custom_radius=28.0, apply_custom_radius=True):
    """Hough circle detecction."""
    equ_img = cv2.equalizeHist(img)
    gau_img = cv2.GaussianBlur(equ_img, (5, 5), 0)
    lap_img = cv2.Laplacian(gau_img, -1, ksize=5)
    pre_img = cv2.medianBlur(lap_img, 5)
    circles = cv2.HoughCircles(pre_img, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=100,
                                param2=precise, minRadius=minRadius, maxRadius=maxRadius)
    centers = circles[0, :, :2]
    if apply_custom_radius:
        hough_boxes = [[max(0, c[0]-custom_radius)/600, max(0, c[1]-custom_radius)/800, \
                    min(c[0]+custom_radius, 600)/600, min(c[1]+custom_radius, 800)/800] for c in centers]
    else:
        radius = np.minimum(circles[0, :, 2] + 5, custom_radius)
        hough_boxes = [[max(0, c[0]-r)/600, max(0, c[1]-r)/800, \
                    min(c[0]+r, 600)/600, min(c[1]+r, 800)/800] for c, r in zip(centers,radius)]
    return hough_boxes


def abnormal_filter(boxes, scores, abnormal_indices):
    """Filter out the boundingboxes that are misdetected by the edge."""
    abnormal=[]
    center0 = (boxes[abnormal_indices[0,0], 2]-boxes[abnormal_indices[0,0], 0])/2 + boxes[abnormal_indices[0,0], 0]
    center1 = (boxes[abnormal_indices[0,1], 3]-boxes[abnormal_indices[0,1], 1])/2 + boxes[abnormal_indices[0,1], 1]
    center2 = (boxes[abnormal_indices[-1,0], 2]-boxes[abnormal_indices[-1,0], 0])/2 + boxes[abnormal_indices[-1,0], 0]
    center3 = (boxes[abnormal_indices[-1,1], 3]-boxes[abnormal_indices[-1,1], 1])/2 + boxes[abnormal_indices[-1,1], 1]

    if center0<boxes[abnormal_indices[1,0], 0]:
        abnormal.append(abnormal_indices[0,0])
    if center1<boxes[abnormal_indices[1,1], 1]:
        abnormal.append(abnormal_indices[0,1])
    if center2>boxes[abnormal_indices[2,0], 2]:
        abnormal.append(abnormal_indices[-1,0])
    if center3>boxes[abnormal_indices[2,1], 3]:
        abnormal.append(abnormal_indices[-1,1])
    abnormal_idx = list(set(abnormal))
    new_boxes = np.delete(boxes, abnormal_idx, axis=0)
    new_scores = np.delete(scores, abnormal_idx)
    return new_boxes, new_scores



def draw_bbox(img, boxes, scores, color=(0, 255, 0)):
    """Draw boundingboxes on the given image."""
    h, w = img.shape[:2]
    for box, score in zip(boxes, scores):
        min_x = max(0, box[0])
        min_y = max(0, box[1])
        max_x = min(w, box[2])
        max_y = min(h, box[3])
        cv2.rectangle(img, (min_x, min_y),
                      (max_x, max_y), color, 2)
        cv2.putText(img, 'cap:' + str(score)[0:4], (min_x, max_y),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
    return img
