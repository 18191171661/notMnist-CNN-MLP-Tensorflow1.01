# import the related packages
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import time
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage
from six.moves import xrange
from scipy.misc import imsave
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

from function import *
#from noMnist_dataset import *


# define the changeable parameters
flags = tf.app.flags
#flags.DEFINE_boolean('fake__data', False, 'the fake data used for unit testing')
flags.DEFINE_integer('max__steps', 200, 'the number of epoch')
flags.DEFINE_integer('dropout', 0.9, 'the value of the droup out')
flags.DEFINE_integer('learning__rate', 0.001, 'the learning rate of the model')
flags.DEFINE_string('optimizer', 'adam', 'the optimizer of the model')
flags.DEFINE_string('data__dir', '/input_data', 'the direction of the data')
flags.DEFINE_string('log__dir', 'D:/Data Minning/train_code/train/noMnist/model/', 'the direction the log file')
FLAGS = flags.FLAGS
 
def choose_optimizer(name):
    if name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning__rate)
    elif name == 'adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning__rate)
    elif name == 'adag':
        optimizer = tf.train.AdagradOptimizer(FLAGS.learning__rate)
    elif name == 'adad':
        optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning__rate)
    elif name == 'rmsp':
        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning__rate)
    else:
        print('please add you optimizer...')
        raise Exception('Error...')
    return optimizer


def train():
    
    start_time = time.time()
    num_classes = 10          # The class of the directions
    np.random.seed(133)         

    train_size = 200000       # The size of training datasets
    valid_size = 10000        # The size of validation datasets
    test_size = 10000         # The size of testing datasets
    
    image_size = 28           # Pixel width and height.
    pixel_depth = 255.0       # Number of levels per pixel.
    num_labels = 10           # The number of lables
    
    url = 'http://commondatastorage.googleapis.com/books1000/'
    last_percent_reported = None
    data_root = '.'           # Change me to store data elsewhere
    
    # Download the notMnist datasets(tar file)
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    
    # Tar the file to the folders
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    
    # Change the file to the pickle file
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)     
           
    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
    
    
    # Shuffer the datasets
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    
    # Save the pickle file and check it
    save_pickle(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
    
    # Change the format of the datasets
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)
    
    # Create a Session layer
    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')

    with tf.name_scope('input_reshape'):
        image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('image', image, 10)
        
    hidden1 = nn_layer(x, 784, 500, 'layer1')
    
    print(hidden1.shape)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('druoput__keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)
        
        print(dropped.shape)

    # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
    
    print(y.shape)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = choose_optimizer(name = FLAGS.optimizer).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to the log_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log__dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log__dir + '/test')
    tf.global_variables_initializer().run()
             
    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = get_batch_data(data = train_dataset, label = train_labels, batch_size = 100)
            k = FLAGS.dropout
        else:
            xs, ys = get_batch_data(data = test_dataset, label = test_labels, batch_size = 100)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}
    
    accuracy = []
    epoch = []
    for i in xrange(FLAGS.max__steps):
        saver = tf.train.Saver()
        saver.save(sess, FLAGS.log__dir, global_step = i)
        if i % 20 == 0:  
            # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %d: %s' % (i, acc))
            
            # Collect the accuracy and the number of epoch
            j = i
            accuracy.append(acc)
            epoch.append(j)
            
        else:  
            # Record train set summaries, and train
            if i % 100 == 0:  
                # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  
                # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    
    train_writer.close()
    test_writer.close()
    
    # Plot the accuracy of the model
    plt.plot(epoch, accuracy)
    plt.xlable('the number of each epoch')
    plt.ylable('the accuracy of each epoch')
    plt.title('the accuracy of the model')
    plt.show()
    
    print('ending...')
    print('This whole code host %d seconds...' %(time.time() - start_time))

def main(_):
    if tf.gfile.Exists(FLAGS.log__dir):
        tf.gfile.DeleteRecursively(FLAGS.log__dir)
    tf.gfile.MakeDirs(FLAGS.log__dir)
    train()

          
if __name__ == '__main__':
    tf.app.run()
