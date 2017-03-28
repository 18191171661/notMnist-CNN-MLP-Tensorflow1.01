from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage
from six.moves import xrange
from scipy.misc import imsave
from six.moves import cPickle as pickle
from IPython.display import display, Image
from six.moves.urllib.request import urlretrieve

class notMnist(object):
    def __init__(self, num_classes, train_size, valid_size, test_size, image_size, 
                 pixel_depth, num_labels, url, last_percent_reported, data_root):
        self.num_classes = num_classes
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.image_size = image_size
        self.pixel_depth = pixel_depth
        self.num_labels = num_labels
        self.url = url
        self.last_percent_reported = last_percent_reported
        self.data_root = data_root
        
        # define some funtion to sample the code
    def download_progress_hook(self, count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 5% change in download progress.
        """
        #global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)

        if self.last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
      
        self.last_percent_reported = percent
    
    def maybe_download(self, filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        dest_filename = os.path.join(self.data_root, filename)
        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', filename) 
            filename, _ = urlretrieve(self.url + filename, dest_filename, reporthook=download_progress_hook)
            print('\nDownload Complete!')
        statinfo = os.stat(dest_filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', dest_filename)
        else:
            raise Exception( 'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
        return dest_filename

    def maybe_extract(self, filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
        # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall(self.data_root)
            tar.close()
        data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != self.num_classes:
            raise Exception( 'Expected %d folders, one per class. Found %d instead.' 
                            % (self.num_classes, len(data_folders)))
        print(data_folders)
        return data_folders


    def make_arrays(self, nb_rows, img_size):
        """Change the data and lables to the arrays """
        if nb_rows:
            dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
            label = np.ndarray(nb_rows, dtype=np.int32)
        else:
            dataset, label = None, None
        return dataset, label
    
    def merge_datasets(self, pickle_files, train_size, valid_size=0):
        num_classes = len(pickle_files)
        valid_dataset, valid_labels = self.make_arrays(valid_size, self.image_size)
        train_dataset, train_labels = self.make_arrays(train_size, self.image_size)
        vsize_per_class = valid_size // self.num_classes
        tsize_per_class = train_size // self.num_classes
    
        start_v, start_t = 0, 0
        end_v, end_t = vsize_per_class, tsize_per_class
        end_l = vsize_per_class+tsize_per_class
        for label, pickle_file in enumerate(pickle_files):       
            try:
                with open(pickle_file, 'rb') as f:
                    letter_set = pickle.load(f)
                    np.random.shuffle(letter_set)
                    if valid_dataset is not None:
                        valid_letter = letter_set[:vsize_per_class, :, :]
                        valid_dataset[start_v:end_v, :, :] = valid_letter
                        valid_labels[start_v:end_v] = label
                        start_v += vsize_per_class
                        end_v += vsize_per_class
                    
                    train_letter = letter_set[vsize_per_class:end_l, :, :]
                    train_dataset[start_t:end_t, :, :] = train_letter
                    train_labels[start_t:end_t] = label
                    start_t += tsize_per_class
                    end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise
    
        return valid_dataset, valid_labels, train_dataset, train_labels

    def load_letter(self, folder, min_num_images):
        """Load the data for a single letter label."""
        image_files = os.listdir(folder)
        dataset = np.ndarray(shape=(len(image_files), self.image_size, self.image_size),dtype=np.float32)
        print(folder)
        num_images = 0
        for image in image_files:
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
                if image_data.shape != (self.image_size, self.image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[num_images, :, :] = image_data
                num_images = num_images + 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
        
        dataset = dataset[0:num_images, :, :]
        if num_images < min_num_images:
            raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))
    
        print('Full dataset tensor:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))
        return dataset

    def maybe_pickle(self, data_folders, min_num_images_per_class, force=False):
        """Check the pickle file and load the pictures"""
        dataset_names = []
        for folder in data_folders:
            set_filename = folder + '.pickle'
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:
                print('Pickling %s.' % set_filename)
                dataset = load_letter(folder, min_num_images_per_class)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)
  
        return dataset_names

    def save_pickle(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        pickle_file = os.path.join(self.data_root, 'notMNIST.pickle')

        try:
            f = open(pickle_file, 'wb')
            save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    
        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)

    def randomize(self, dataset, labels):
        """Random the datas and the lables"""
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels
    
    def reformat(self, dataset, lables):
        """Change the shape of the datasets and lables"""
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        lables = (np.arange(self.num_labels) == lables[:,None]).astype(np.float32)
        return dataset, lables
    
    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape, value):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
    # Define the weights and the biases of the layers 
    def layer_weights(self, patch_size, num_channels, depth, image_size, num_hidden, num_labels):
        with tf.name_scope('layer1'):
            weights1 = self.weight_variable([patch_size, patch_size, num_channels, depth])
            biases1 = self.bias_variable([depth], value = 0.0)
            self.variable_summaries(weights1)
            self.variable_summaries(biases1)
        
        with tf.name_scope('layer2'):
            weights2 = self.weight_variable([patch_size, patch_size, depth, depth])
            biases2 = self.bias_variable([depth], value = 1.0)
            self.variable_summaries(weights2)
            self.variable_summaries(biases2)
        
        with tf.name_scope('layer3'):
            weights3 = self.weight_variable([image_size // 4 * image_size // 4 * depth, num_hidden])
            biases3 = self.bias_variable([num_hidden], value = 1.0)
            self.variable_summaries(weights3)
            self.variable_summaries(biases3)
        
        with tf.name_scope('layer4'):
            weights4 = self.weight_variable([num_hidden, num_labels])
            biases4 = self.bias_variable([num_labels], value = 1.0)
            self.variable_summaries(weights4)
            self.variable_summaries(biases4)
        return weights1, weights2, weights3, weights4, biases1, biases2, biases3, biases4
    
    # Build the CNN model
    def notMnist_CNN(self, data, weights1, weights2, weights3, weights4, biases1, biases2, biases3, biases4):
        conv = tf.nn.conv2d(data, weights1, [1, 1, 1, 1], padding='SAME')
        pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + biases1)

        conv = tf.nn.conv2d(hidden, weights2, [1, 1, 1, 1], padding='SAME')
        pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + biases2)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, weights3) + biases3)
        drop = tf.nn.dropout(hidden, 1)

        output_layer = tf.matmul(drop, weights4) + biases4
        return output_layer
    
 
    def get_batch_data(self, data,label,batch_size):
        """Get the batch datas and the lables"""
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index : start_index + batch_size], label[start_index : start_index + batch_size]
        
    