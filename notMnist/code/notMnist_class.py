from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage
from six.moves import xrange
from six.moves import cPickle as pickle
from IPython.display import display, Image
from six.moves.urllib.request import urlretrieve

"""
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

#
"""
#global train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

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

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
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
    
    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations
 
    def get_batch_data(self, data,label,batch_size):
        """Get the batch datas and the lables"""
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index : start_index + batch_size], label[start_index : start_index + batch_size]
        
    