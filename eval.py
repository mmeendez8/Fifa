import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_model_description

# Params from model to recover
params = dict()
params['learning_rate'] = 0.0001
params['latent_dim'] = 2
max_epoch = 0  # Last epoch saved
model = get_model_description(params)

model_dir = 'Models/{}/'.format(model)
graph = tf.Graph()


with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph(model_dir+'{}-{}.meta'.format(model, max_epoch))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Recover tensors and ops from graph
    filenames_input = graph.get_tensor_by_name('DataPipe/filenames_tensor:0')
    output = graph.get_tensor_by_name('Decoder/decoded_output:0')
    z = graph.get_tensor_by_name('Encoder/add:0')
    iterator = graph.get_operation_by_name('DataPipe/MakeIterator')

    # Init iterator
    sess.run(iterator)

    # Recover Messi image
    artificial_image = sess.run(output, feed_dict={filenames_input: ['Data/Images/0.png']})
    plt.imshow(artificial_image[0].reshape((48, 48, 3)))
    plt.show()

    # # Create random images
    # for i in range(5):
    #     # Create artificial image from unit norm sample
    #     artificial_image = sess.run(output, feed_dict={z: np.random.normal(0, 1, (1, 10))})
    #     plt.figure()
    #     with sns.axes_style("white"):
    #         plt.imshow(artificial_image[0].reshape((48, 48, 3)))
    #         plt.show()

    # Create mesh grid of values
    if params['latent_dim'] == 2:

        values = np.arange(-3, 4, .5)
        xx, yy = np.meshgrid(values, values)
        input_holder = np.zeros((1, 2))

        # Matrix that will contain the grid of images
        container = np.zeros((48 * len(values), 48 * len(values),3))

        for row in range(xx.shape[0]):
            for col in range(xx.shape[1]):
                input_holder[0, :] = [xx[row, col], yy[row, col]]
                artificial_image = sess.run(output, feed_dict={z: input_holder})
                container[row * 48: (row + 1) * 48, col * 48: (col + 1) * 48, :3] = artificial_image[0].reshape((48, 48, 3))

        my_dpi = 96
        plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.imshow(container)
        plt.savefig('Results/{}/Test/grid.png'.format(model), dpi=my_dpi)