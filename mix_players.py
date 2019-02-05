import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_model_description, get_interp

# Number of interpolations between inputs
interpolations = 5

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

# Select players to mix
player1 = 'Data/Images/35.png'
player2 = 'Data/Images/8.png'
players = [player1, player2]


with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph(model_dir+'{}-{}.meta'.format(model, FLAGS.max_epoch))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Recover tensors and ops from graph
    filenames_input = graph.get_tensor_by_name('DataPipe/filenames_tensor:0')
    output = graph.get_tensor_by_name('Decoder/decoded_output:0')
    z = graph.get_tensor_by_name('Encoder/add:0')
    iterator = graph.get_operation_by_name('DataPipe/MakeIterator')

    # Get latent vectors
    sess.run(iterator, feed_dict={filenames_input: players})
    latents = sess.run([z])[0]

    # Get interpolation vectors between points
    interp = get_interp(latents[0,:], latents[1,:], interpolations)

    # Encode new vectors
    artificial_images = sess.run(output, feed_dict={z: interp})

    f, axarr = plt.subplots(1, interpolations+2)
    for j, artificial_image in enumerate(artificial_images):
        with sns.axes_style("white"):
            axarr[j].imshow(artificial_image.reshape((48, 48, 3)))
            axarr[j].axis('off')
    plt.show()






