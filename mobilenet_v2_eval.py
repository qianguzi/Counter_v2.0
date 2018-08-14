# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Validate mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from mobilenet import mobilenet_v2


tf.app.flags.DEFINE_string('master', '', 'Session master')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes to distinguish')
tf.app.flags.DEFINE_integer('num_examples', 13342, 'Number of examples to evaluate')
tf.app.flags.DEFINE_integer('image_size', 96, 'Input image resolution')
tf.app.flags.DEFINE_float('depth_multiplier', 0.5, 'Depth multiplier for mobilenet')
tf.app.flags.DEFINE_bool('quantize', False, 'Quantize training')
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/', 'The directory for checkpoints')
tf.app.flags.DEFINE_string('eval_dir', '../eval/', 'Directory for writing eval event logs')
tf.app.flags.DEFINE_string('dataset_dir', '../tfrecords/test.tfrecords', 'Location of dataset')

FLAGS = tf.app.flags.FLAGS


def read_tfrecord(filename, shuffle=True):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(
        [filename], shuffle=shuffle, num_epochs=1)
    # 从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个样例，如果需要解析多个，可以用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string),
                  'image_name': tf.FixedLenFeature([], tf.string)})
    # 将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [96, 96, 3])
    #img = tf.reshape(img, [64, 64, 3])
    #img = tf.cast(img, tf.float32)*1/255 - 0.5
    img = tf.cast(img, tf.float32)*1/127.5 - 1
    label = tf.cast(features['label'], tf.int64)
    name = tf.cast(features['image_name'], tf.string)

    return img, label, name


def get_batch(filename, batch_size, shuffle=True):
    '''Get batch.'''

    image, label, name = read_tfrecord(filename, shuffle)
    capacity = 5 * batch_size

    img_batch, label_batch, name_batch = tf.train.batch([image, label, name], batch_size,
                                                            capacity=capacity, num_threads=4,)
    return img_batch, label_batch, name_batch


def metrics(logits, labels):
  """Specify the metrics for eval.

  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.

  Returns:
     Eval Op for the graph.
  """
  labels = tf.squeeze(labels)
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy': tf.metrics.accuracy(tf.argmax(logits, 1), labels),
      'Recall_2': tf.metrics.recall_at_k(labels, logits, 2),
  })
  for name, value in names_to_values.items():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
  return list(names_to_updates.values())


def build_model():
  """Build the mobilenet_v1 model for evaluation.

  Returns:
    g: graph with rewrites after insertion of quantization ops and batch norm
    folding.
    eval_ops: eval ops for inference.
    variables_to_restore: List of variables to restore from checkpoint.
  """
  g = tf.Graph()
  with g.as_default():
    inputs, labels, _= get_batch(FLAGS.dataset_dir, FLAGS.batch_size)
    scope = mobilenet_v2.training_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      _, end_points = mobilenet_v2.mobilenet_v2_050(
          inputs,
          is_training=False,
          num_classes=FLAGS.num_classes)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph() 

    eval_ops = metrics(end_points['Predictions'], labels)

  return g, eval_ops


def eval_model():
  """Evaluates mobilenet_v1."""
  g, eval_ops = build_model()
  with g.as_default():
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
    slim.evaluation.evaluate_once(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops)


def load(sess, saver, checkpoint_dir):
    #import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # print(ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return ckpt_name
    else:
        raise Exception("[*] Failed to find a checkpoint")

def test_model():
    errfile = '../_error/other_errors.txt'

    sess = tf.Session()

    inputs, labels, name_batch= get_batch(FLAGS.dataset_dir, FLAGS.batch_size, shuffle=False)
    scope = mobilenet_v2.training_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      logits, end_points = mobilenet_v2.mobilenet_v2_050(
          inputs,
          is_training=False,
          num_classes=FLAGS.num_classes)

    # evaluate model, for classification
    correct_pred = tf.equal(tf.argmax(end_points['Predictions'], 1), labels)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # saver for restore model
    saver = tf.train.Saver()
    print('[*] Try to load trained model...')
    ckpt_name = load(sess, saver, FLAGS.checkpoint_dir)

    step = 0
    accs = 0
    me_acc = 0
    errors_name = []
    max_steps = int(FLAGS.num_examples / FLAGS.batch_size)
    print('START TESTING...')
    try:
      while not coord.should_stop():
        for _step in range(step+1, step+max_steps+1):
          # test
          #_label, _logits, _points = sess.run([labels, logits, end_points])
          _name, _logits, _corr, _acc = sess.run([name_batch, logits, correct_pred, acc])
          if (~_corr).any():
              errors_name.extend(list(_name[~_corr]))
          accs += _acc
          me_acc = accs/_step
          if _step % 20 == 0:
              print(time.strftime("%X"), 'global_step:{0}, current_acc:{1:.6f}'.format
                    (_step, me_acc))
    except tf.errors.OutOfRangeError:
      accuracy = 1 - len(errors_name)/FLAGS.num_examples
      print(time.strftime("%X"),
            'RESULT >>> current_acc:{0:.6f}'.format(accuracy))
      # print(errors_name)
      errorsfile = open(errfile, 'a')
      errorsfile.writelines('\n' + ckpt_name + '--' + str(accuracy))
      for err in errors_name:
          errorsfile.writelines('\n' + err.decode('utf-8'))
      errorsfile.close()
    finally:
      coord.request_stop()
      coord.join(threads)
      sess.close()
    print('FINISHED TESTING.')


def main(unused_arg):
    test_model()


if __name__ == '__main__':
  tf.app.run(main)
