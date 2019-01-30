from network import VAE
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_and_process_data
import os
import shutil



# Network parameters
tf.flags.DEFINE_float('learning_rate', .0005, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
tf.flags.DEFINE_integer('batch_size', 128, 'Minibatch size')
tf.flags.DEFINE_integer('latent_dim', 10, 'Number of latent dimensions')
tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
tf.flags.DEFINE_integer('inputs_decoder', 49, 'Size of decoder input layer')
tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')
tf.flags.DEFINE_bool('plot_latent', True, 'Plot latent space')

FLAGS = tf.flags.FLAGS

# Define and create results folders
results_folder = 'Results'
[os.makedirs(os.path.join(results_folder, folder)) for folder in ['Test', 'Train']
    if not os.path.exists(os.path.join(results_folder, folder))]

# Empty log folder
try:
    if not len(os.listdir(FLAGS.logdir)) == 0:
        shutil.rmtree(FLAGS.logdir)
except:
    pass

# Create tf dataset
dataset, n_batches = load_and_process_data(base_dir='Data/Images', batch_size=32, shuffle=False)

iterator = dataset.make_initializable_iterator()
input_batch = iterator.get_next()
input_batch = tf.reshape(input_batch, shape=[-1, 48, 48, 1])


vae = VAE(input_batch, FLAGS.latent_dim, FLAGS.learning_rate, )

init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
gpu_options = tf.GPUOptions(allow_growth=True)

# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(init_vars)

    for epoch in range(FLAGS.epochs):
        print(epoch)
        sess.run(iterator.initializer)
        flag = True
        while True:
            try:
                sess.run(vae.optimize)
                if flag:
                    flag = False
                    # Get input and recover output images comparison
                    target, output_ = sess.run([input_batch, vae.decode])
                    f, axarr = plt.subplots(FLAGS.test_image_number, 2)
                    for j in range(FLAGS.test_image_number):
                        for pos, im in enumerate([target, output_]):
                            axarr[j, pos].imshow(im[j].reshape((48, 48)), cmap='gray')
                            axarr[j, pos].axis('off')

                    plt.show()

            except tf.errors.OutOfRangeError:
                break

