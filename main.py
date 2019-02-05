from network import VAE
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import load_and_process_data
import os
from utils import create_folder, get_model_description, delete_old_logs, get_files, create_results_folder
from time import time

# Program parameters
tf.flags.DEFINE_float('learning_rate', .0001, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 100, 'Number of steps to run trainer.')
tf.flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
tf.flags.DEFINE_integer('latent_dim', 2, 'Number of latent dimensions')
tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
tf.flags.DEFINE_integer('epochs_to_plot', 2, 'Number of epochs before saving test sample of reconstructed images')
tf.flags.DEFINE_integer('save_after_n', 20, 'Number of epochs before saving network')
tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')
tf.flags.DEFINE_string('data_path', './Data/Images', 'Logs folder')
tf.flags.DEFINE_bool('plot_latent', True, 'Plot latent space')
tf.flags.DEFINE_bool('shuffle', True, 'Plot latent space')
FLAGS = tf.flags.FLAGS


# Prepare output directories
model_description = get_model_description(FLAGS)
results_folder = create_results_folder(os.path.join('Results', model_description))
model_folder = create_folder(os.path.join('Models', model_description))
delete_old_logs(FLAGS.logdir)


# Create tf dataset
with tf.name_scope('DataPipe'):
    filenames = tf.placeholder_with_default(get_files(FLAGS.data_path), shape=[None], name='filenames_tensor')
    dataset = load_and_process_data(filenames, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle)
    iterator = dataset.make_initializable_iterator()
    input_batch = iterator.get_next()

# Create model
vae = VAE(input_batch, FLAGS.latent_dim, FLAGS.learning_rate, )

init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]

saver = tf.train.Saver()

# Training loop
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(init_vars)
    merged_summary_op = tf.summary.merge_all()
    write_graph = True

    for epoch in range(FLAGS.epochs):
        print('Actual epochs is: {}'.format(epoch), end='', flush=True)
        sess.run(iterator.initializer)
        flag = True
        ts = time()

        while True:
            try:
                sess.run(vae.optimize)

                # Get sample of images and their decoded couples
                if flag and not epoch % FLAGS.epochs_to_plot:
                    flag = False
                    summ, target, output_ = sess.run([merged_summary_op, input_batch, vae.decode])
                    writer.add_summary(summ, epoch)
                    f, axarr = plt.subplots(FLAGS.test_image_number, 2)
                    for j in range(FLAGS.test_image_number):
                        for pos, im in enumerate([target, output_]):
                            axarr[j, pos].imshow(im[j].reshape((48, 48, 3)))
                            axarr[j, pos].axis('off')

                    plt.savefig(os.path.join(results_folder, 'Train/Epoch_{}').format(epoch))
                    plt.close(f)

            except tf.errors.OutOfRangeError:
                print('\t Epoch time: {}'.format(time() - ts))

                # Save model
                if not epoch % FLAGS.save_after_n and epoch > 0:
                    print('Saving model...')
                    saver.save(sess, model_folder, global_step=epoch)
                break


