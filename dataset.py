import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def get_files(base_dir):
    files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    return files



def remove_noise(image, channels):
    alpha = tf.greater(image[:, :, 3], 5)
    alpha = tf.expand_dims(tf.cast(alpha, dtype=tf.uint8), 2)
    noise_filtered = tf.multiply(alpha, image)

    return noise_filtered[..., :channels]


def parse_function(filename, channels=4):

    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=4)
    image = remove_noise(image, channels)
    image = tf.image.rgb_to_grayscale(image[:,:,:3])
    image = tf.image.resize_images(image, [224, 224])

    return image


def load_and_process_data(base_dir, batch_size, shuffle=True):
    '''
    Reveices a list of filenames and returns preprocessed images as a tensorflow dataset
    :param filenames: list of file paths
    :param batch_size: mini-batch size
    :param shuffle: Boolean
    :return:
    '''
    with tf.device('/cpu:0'):
        with tf.name_scope('DataPipe'):
            filenames = get_files(base_dir)
            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            dataset = dataset.map(parse_function, num_parallel_calls=4)

            if shuffle:
                dataset = dataset.shuffle((len(filenames)//4)+1) # Number of imgs to keep in a buffer to randomly sample

            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(batch_size)
            n_batches = len(filenames) // batch_size

            if len(filenames) % batch_size != 0:
                n_batches += 1

    return dataset, n_batches


base_dir = 'Data/Images'
dataset, n_batches = load_and_process_data(base_dir=base_dir, batch_size=5, shuffle=False)


iterator = dataset.make_initializable_iterator()
input_batch = iterator.get_next()



init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]



with tf.Session() as sess:
    sess.run([init_vars, iterator.initializer]) # Initialize variables and the iterator

    while 1:    # Iterate until we get out of range error!
        try:
            batch = sess.run(input_batch)
            print(batch.shape)  # Get batch dimensions
            plt.imshow(batch[0,:,:,0] , cmap='gray')
            plt.show()
        except tf.errors.OutOfRangeError:  # This exception is triggered when all batches are iterated
            print('All batches have been iterated!')
            break