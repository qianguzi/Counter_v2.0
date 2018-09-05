import tensorflow as tf

def read_tfrecord(file_path, shuffle=True):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer(
        [file_path], shuffle=shuffle)
    # 从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'image_raw': tf.FixedLenFeature([], tf.string),
                  'bbox_raw': tf.FixedLenFeature([], tf.string),
                  'bbox_num': tf.FixedLenFeature([], tf.int64),
                  'image_name': tf.FixedLenFeature([], tf.string)})
    # 将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])

    bbox = tf.decode_raw(features['bbox_raw'], tf.float64)
    bbox = tf.reshape(bbox, [18, 4])
    bbox = tf.cast(bbox, tf.float32)
    num = tf.cast(features['bbox_num'], tf.int32)
    name = tf.cast(features['image_name'], tf.string)

    return img, bbox, num, name


def get_batch(file_path, batch_size, shuffle=True):
    '''Get batch.'''

    img, bbox, num, name = read_tfrecord(file_path, shuffle)
    capacity = 5 * batch_size

    img_batch, bbox_batch, num_batch, name_batch = tf.train.batch([img, bbox, num, name], batch_size,
                                                                  capacity=capacity, num_threads=4)
    return img_batch, bbox_batch, num_batch, name_batch


if __name__ == '__main__':
    file_path = '../tfrecords/train.tfrecords'
    img_batch, bbox_batch, num_batch, name_batch = get_batch(file_path, 10)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                imgs, bboxes, nums, names = sess.run([img_batch, bbox_batch, num_batch, name_batch])
                print(nums)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            coord.join(threads)