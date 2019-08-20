# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a trial AI script for testing task-processing pipeline on
# Nebula AI Orion working node. Original article:
# https://medium.com/deep-math-machine-learning-ai/chapter-9-2-nlp-
# code-for-word2vec-neural-network-Tensorflow-544db99f5334
# It uses Tensorflow 1.13.1
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import os
import random
import sys
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector

print("tensorflow version = ", tf.VERSION)
current = time.time()
# Give a folder path as an argument with '--log_dir' to save
# TensorBoard summaries. Default is a log folder in current directory.
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# Step 1: Download the data.
# url = 'http://mattmahoney.net/dc/'
# Download a small chunk of Wikipedia articles collection
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = './datadir/'
try:
    os.makedirs(os.path.dirname(data_path))
except FileExistsError:
    pass
filename = os.path.join(data_path, 'text8.zip')
if not os.path.exists(filename):
    print("Downloading the datadset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, filename)
    print("Done!")


# # pylint: disable=redefined-outer-name
# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   local_filename = os.path.join(gettempdir(), filename)
#   if not os.path.exists(local_filename):
#     local_filename, _ = urllib.request.urlretrieve(url + filename,
#                                                    local_filename)
#   statinfo = os.stat(local_filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception('Failed to verify ' + local_filename +
#                     '. Can you get to it with a browser?')
#   return local_filename
# filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(file_name):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(file_name) as file:
        data_fun = tf.compat.as_str(file.read(file.namelist()[0])).split()
    return data_fun


# Read files

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count_data = [['UNK', -1]]
    count_data.extend(collections.Counter(words).most_common(n_words - 1))
    dic = dict()
    for word, _ in count_data:
        dic[word] = len(dic)
    data_fun = list()
    unk_count = 0
    for word in words:
        index = dic.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data_fun.append(index)
    count_data[0][1] = unk_count
    reversed_dictionary = dict(zip(dic.values(), dic.keys()))
    return data_fun, count_data, dic, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size_fun, num_skips_fun, skip_window_fun):
    global data_index
    assert batch_size_fun % num_skips_fun == 0
    assert num_skips_fun <= 2 * skip_window_fun
    batch_fun = np.ndarray(shape=batch_size_fun, dtype=np.int32)
    labels_fun = np.ndarray(shape=(batch_size_fun, 1), dtype=np.int32)
    span = 2 * skip_window_fun + 1  # [ skip_window_fun target skip_window_fun ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for m in range(batch_size_fun // num_skips_fun):
        context_words = [w for w in range(span) if w != skip_window_fun]
        words_to_use = random.sample(context_words, num_skips_fun)
        for n, context_word in enumerate(words_to_use):
            batch_fun[m * num_skips_fun + n] = buffer[skip_window_fun]
            labels_fun[m * num_skips_fun + n, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch_fun
    data_index = (data_index + len(data) - span) % len(data)
    return batch_fun, labels_fun


batch, labels = generate_batch(batch_size_fun=8, num_skips_fun=2, skip_window_fun=1)
# for i in range(8):
#   print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
#         reverse_dictionary[labels[i, 0]])


# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the GPU
    with tf.device('/gpu:0'):
        # with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

# Step 5: Begin training.
# num_steps = 200000
num_steps = 100000

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True), graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()

    print('Initialized and begin training')

    start_time = time.time()
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                    skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at STEPS ', str(round((step / 20000 * 100), 1)), '% : ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 20000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
        for i in range(vocabulary_size):
            f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))
    print("training done")
    end_time = time.time()
    print("trainng time: ", end_time - start_time)

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()


# Step 6: Visualize the embeddings,Using TSNE to Plot a Subset of Similar Words from Word2Vec.
# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_emb, labels_fun, file_name):
    assert low_dim_emb.shape[0] >= len(labels_fun), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for j, label in enumerate(labels_fun):
        x, y = low_dim_emb[j, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(file_name)


try:
    print("Start plotting")
    start_plotting = time.time()
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 350
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(FLAGS.log_dir, 'Display of word vectors.png'))
    end_plotting = time.time()
    print("plotting time: ", str(round(end_plotting - start_plotting, 1)), "seconds")
    print("Plot saved.")
except ImportError as ex:
    print(ex)
