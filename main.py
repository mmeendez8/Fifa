from network import VAE
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_and_process_data
import os
from utils import create_results_folder, get_model_description, delete_old_logs


# Network parameters
tf.flags.DEFINE_float('learning_rate', .0005, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 20, 'Number of steps to run trainer.')
tf.flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
tf.flags.DEFINE_integer('latent_dim', 6, 'Number of latent dimensions')
tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')
tf.flags.DEFINE_bool('plot_latent', True, 'Plot latent space')
tf.flags.DEFINE_bool('shuffle', True, 'Plot latent space')

FLAGS = tf.flags.FLAGS

# Prepare output directories
model_description = get_model_description(FLAGS.flag_values_dict())
results_folder = create_results_folder(os.path.join('Results', model_description))
model_folder = os.path.join('Models', model_description)
delete_old_logs(FLAGS.logdir)

# Create tf dataset
dataset, n_batches = load_and_process_data(base_dir='Data/Images', batch_size=FLAGS.batch_size, shuffle=False)

iterator = dataset.make_initializable_iterator()
input_batch = iterator.get_next()

vae = VAE(input_batch, FLAGS.latent_dim, FLAGS.learning_rate, )

init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
gpu_options = tf.GPUOptions(allow_growth=True)


# Training loop
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(init_vars)

    for epoch in range(FLAGS.epochs):
        print('Actual epochs is: {}'.format(epoch))
        sess.run(iterator.initializer)
        flag = True



        while True:
            try:
                sess.run(vae.optimize)

                # Save model
                if not epoch % 1:
                    if os.path.exists(model_folder):
                        delete_old_logs(model_folder)
                    inputs, outputs = vae.get_inputs_outputs()
                    tf.saved_model.simple_save(
                        sess, model_folder, inputs, outputs
                    )

                if flag:
                    flag = False
                    # Get input and recover output images comparison
                    target, output_ = sess.run([input_batch, vae.decode])
                    f, axarr = plt.subplots(FLAGS.test_image_number, 2)
                    for j in range(FLAGS.test_image_number):
                        for pos, im in enumerate([target, output_]):
                            axarr[j, pos].imshow(im[j].reshape((48, 48, 3)))
                            axarr[j, pos].axis('off')

                    plt.savefig(os.path.join(results_folder, 'Train/Epoch_{}').format(epoch))
                    plt.close(f)

            except tf.errors.OutOfRangeError:
                pass


