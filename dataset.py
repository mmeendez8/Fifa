import tensorflow as tf


def remove_noise(image):
    alpha = tf.greater(image[:, :, 3], 50)
    alpha = tf.expand_dims(tf.cast(alpha, dtype=tf.uint8), 2)
    noise_filtered = tf.multiply(alpha, image)

    return noise_filtered[..., :3]


def parse_function(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=4)
    image = remove_noise(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape([48, 48, 3])

    return image


def load_and_process_data(filenames, batch_size, shuffle=True):
    '''
    Reveices a list of filenames and returns preprocessed images as a tensorflow dataset
    :param filenames: list of file paths
    :param batch_size: mini-batch size
    :param shuffle: Boolean
    :return:
    '''
    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(parse_function, num_parallel_calls=4)

        if shuffle:
            dataset = dataset.shuffle(5000) # Number of imgs to keep in a buffer to randomly sample

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

    return dataset

#
# base_dir = 'Data/Images'
# dataset, n_batches = load_and_process_data(base_dir=base_dir, batch_size=5, shuffle=False)
#
#
# iterator = dataset.make_initializable_iterator()
# input_batch = iterator.get_next()
#
#
#
# init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
#
#
#
# with tf.Session() as sess:
#     sess.run([init_vars, iterator.initializer]) # Initialize variables and the iterator
#
#     while 1:    # Iterate until we get out of range error!
#         try:
#             batch = sess.run(input_batch)
#             print(batch.shape)  # Get batch dimensions
#             plt.imshow(batch[0,...])
#             plt.show()
#         except tf.errors.OutOfRangeError:  # This exception is triggered when all batches are iterated
#             print('All batches have been iterated!')
#             break