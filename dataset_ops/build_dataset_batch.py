import tensorflow as tf

def read_tfrecord(file_path, shuffle=True):
    """read tfrecord files and convert to tensors queue."""
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(
        [file_path], shuffle=shuffle)

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'image_raw': tf.FixedLenFeature([], tf.string),
                  'bbox_raw': tf.FixedLenFeature([], tf.string),
                  'bbox_num': tf.FixedLenFeature([], tf.int64),
                  'image_name': tf.FixedLenFeature([], tf.string)})

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])

    bbox = tf.decode_raw(features['bbox_raw'], tf.float64)
    bbox = tf.reshape(bbox, [18, 4])
    bbox = tf.cast(bbox, tf.float32)
    num = tf.cast(features['bbox_num'], tf.int32)
    name = tf.cast(features['image_name'], tf.string)

    return img, bbox, num, name


def get_batch(file_path, batch_size, shuffle=True):
    """Get batch.
    
    Args:
        flie_path: the path to the tfrecord file.
        batch_size: an integer. Batch size.
        shuffle: whether to shuffle the input dataset.
    Returns:
        img_batch: a batch of image data. 
        bbox_batch: a batch of object boundingboxes.  
        num_batch: a batch of object numbers.
        name_batch: a bacth of image names.
    """

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