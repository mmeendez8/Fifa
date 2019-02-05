import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_model_description
import pandas as pd
import os

# Define countries to obtain centroids
countries = ['Spain', 'Germany', 'Norway', 'France', 'Argentina', 'Brazil', 'Japan', 'Nigeria', 'United States']
n_countries = len(countries)

# Get subplot
f, axarr = plt.subplots(1, n_countries)
data = pd.read_csv('Data/data.csv', index_col=0)

# Params from model to recover
tf.flags.DEFINE_float('learning_rate', .0001, 'Initial learning rate.')
tf.flags.DEFINE_integer('latent_dim', 15, 'Number of latent dimensions')
tf.flags.DEFINE_integer('max_epoch', 980, 'Max epoch saved')
FLAGS = tf.flags.FLAGS

# Get model name
model = get_model_description(FLAGS)
model_dir = 'Models/{}/'.format(model)

# Create graph
graph = tf.Graph()

with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph(model_dir+'{}-{}.meta'.format(model, FLAGS.max_epoch))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Recover tensors and ops from graph
    filenames_input = graph.get_tensor_by_name('DataPipe/filenames_tensor:0')
    output = graph.get_tensor_by_name('Decoder/decoded_output:0')
    z = graph.get_tensor_by_name('Encoder/add:0')
    iterator = graph.get_operation_by_name('DataPipe/MakeIterator')

    # Iterate through countries
    for i, country in enumerate(countries):
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