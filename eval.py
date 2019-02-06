import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_model_description, get_interp
import pandas as pd
import os


# Params from model to recover
tf.flags.DEFINE_float('learning_rate', .0001, 'Initial learning rate.')
tf.flags.DEFINE_integer('latent_dim', 15, 'Number of latent dimensions')
tf.flags.DEFINE_integer('max_epoch', 980, 'Max epoch saved')
tf.flags.DEFINE_integer('create_fake_players', 1, 'Number of fake players to generate')
tf.flags.DEFINE_boolean('recover_top_5', True, 'Reconstructs top 5 FIFA players from their latent vector')
tf.flags.DEFINE_boolean('meshgrid', True, 'Creates meshgrid of points in latent space and shows reconstructed images '
                                          '[only when latent_dim = 2]')
tf.flags.DEFINE_list('mix_players', ['Data/Images/35.png', 'Data/Images/8.png'], 'List with two image paths to mix')
tf.flags.DEFINE_integer('n_interpolations', 5, 'Number of interpolations between mixed players')
tf.flags.DEFINE_list('countries', ['Spain', 'Germany', 'Norway', 'France', 'Argentina', 'Brazil', 'Japan', 'Nigeria',
                                   'United States'], 'List of countries to compute average player')
FLAGS = tf.flags.FLAGS


# Get model name
model = get_model_description(FLAGS)
model_dir = 'Models/{}/'.format(model)

# Create graph
graph = tf.Graph()

# Restore network
with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph(model_dir+'{}-{}.meta'.format(model, FLAGS.max_epoch))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Recover tensors and ops from graph
    filenames_input = graph.get_tensor_by_name('DataPipe/filenames_tensor:0')
    z = graph.get_tensor_by_name('Encoder/add:0')
    output = graph.get_tensor_by_name('Decoder/decoded_output:0')
    iterator = graph.get_operation_by_name('DataPipe/MakeIterator')


    # Create fake images
    if FLAGS.create_fake_players:
        for i in range(FLAGS.create_fake_players):
            # Create artificial image from unit norm sample
            artificial_image = sess.run([output], feed_dict={z: np.random.normal(0, 1, (1, FLAGS.latent_dim))})
            plt.figure()
            with sns.axes_style("white"):
                plt.imshow(artificial_image[0].reshape((48, 48, 3)))
            plt.title('Fake image {}'.format(i))
            plt.show()


    # Recover top 5 players reconstructed images
    if FLAGS.recover_top_5:
        f, axarr = plt.subplots(1, 5)
        filenames = [['Data/Images/{}.png'.format(i)] for i in range(5)]
        player_names = ['Messi', 'Ronaldo', 'Neymar', 'De Gea', 'De Bruyne']
        for j, fname in enumerate(filenames):
            sess.run(iterator, feed_dict={filenames_input: fname})
            artificial_image = sess.run(output)
            axarr[j].imshow(artificial_image.reshape((48, 48, 3)))
            axarr[j].axis('off')
            axarr[j].set_title(player_names[j])
        plt.show()


    # Create mesh grid of values
    if FLAGS.latent_dim == 2 and FLAGS.meshgrid:

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

        plt.figure()
        plt.imshow(container)
        plt.show()


    # Mix players
    if len(FLAGS.mix_players) == 2:
        # Number of interpolations between inputs
        interpolations = 5
        # Get latent vectors
        sess.run(iterator, feed_dict={filenames_input: FLAGS.mix_players})
        latents = sess.run([z])[0]

        # Get interpolation vectors between points
        interp = get_interp(latents[0, :], latents[1, :], interpolations)

        # Encode new vectors
        artificial_images = sess.run(output, feed_dict={z: interp})

        f, axarr = plt.subplots(1, interpolations + 2)
        for j, artificial_image in enumerate(artificial_images):
            with sns.axes_style("white"):
                axarr[j].imshow(artificial_image.reshape((48, 48, 3)))
                axarr[j].axis('off')
        plt.show()


    if FLAGS.countries:
        # Get subplot
        f, axarr = plt.subplots(1, len(FLAGS.countries))
        data = pd.read_csv('Data/data.csv', index_col=0)
        # Iterate through countries
        for i, country in enumerate(FLAGS.countries):
            # Get all players from country
            indexes = data[data.Nationality == country].index
            # Get filename if exists
            player_files = [filename for filename in ['Data/Images/{}.png'.format(i) for i in indexes] if
                            os.path.exists(filename)]
            print('There are a total of {} players from {}'.format(len(indexes), country))

            if len(indexes) == 0:
                continue

            # Get latent vectors
            sess.run(iterator, feed_dict={filenames_input: player_files})
            latents = sess.run([z])[0]
            # Get average vector
            average = np.sum(latents, axis=0) / latents.shape[0]

            # Reconstruct image from average latent vector
            average_player = sess.run([output], feed_dict={z: np.expand_dims(average, 0)})[0]

            with sns.axes_style("white"):
                axarr[i].imshow(average_player.reshape((48, 48, 3)))
                axarr[i].axis('off')
                axarr[i].set_title(country)

        plt.show()