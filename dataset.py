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
