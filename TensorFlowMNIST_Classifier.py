#	Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#	Licensed under the Apache License, Version 2.0 (the "License");
#	you may not use this file except in compliance with the License.
#	You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
#	Unless required by applicable law or agreed to in writing, software
#	distributed under the License is distributed on an "AS IS" BASIS,
#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#	See the License for the specific language governing permissions and
#	limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, random
import scipy as sp

from tensorflow.python.framework import random_seed
from tensorflow.python.eager import context


class TensorFlowMNIST_Classifier(object):

	def __init__(self, model_dir="/tmp/mnist_convnet_model"): 

		def cnn_model_fn(features, labels, mode):
			"""Model function for CNN."""
			# Input Layer
			# Reshape X to 4-D tensor: [batch_size, width, height, channels]
			# MNIST images are 28x28 pixels, and have one color channel
			input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

			#print("#"*10, input_layer.name)

			# Convolutional Layer #1
			# Computes 32 features using a 5x5 filter with ReLU activation.
			# Padding is added to preserve width and height.
			# Input Tensor Shape: [batch_size, 28, 28, 1]
			# Output Tensor Shape: [batch_size, 28, 28, 32]
			conv1 = tf.layers.conv2d(
					inputs=input_layer,
					filters=32,
					kernel_size=[5, 5],
					padding="same",
					activation=tf.nn.relu)

			# Pooling Layer #1
			# First max pooling layer with a 2x2 filter and stride of 2
			# Input Tensor Shape: [batch_size, 28, 28, 32]
			# Output Tensor Shape: [batch_size, 14, 14, 32]
			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

			# Convolutional Layer #2
			# Computes 64 features using a 5x5 filter.
			# Padding is added to preserve width and height.
			# Input Tensor Shape: [batch_size, 14, 14, 32]
			# Output Tensor Shape: [batch_size, 14, 14, 64]
			conv2 = tf.layers.conv2d(
					inputs=pool1,
					filters=64,
					kernel_size=[5, 5],
					padding="same",
					activation=tf.nn.relu)

			# Pooling Layer #2
			# Second max pooling layer with a 2x2 filter and stride of 2
			# Input Tensor Shape: [batch_size, 14, 14, 64]
			# Output Tensor Shape: [batch_size, 7, 7, 64]
			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

			# Flatten tensor into a batch of vectors
			# Input Tensor Shape: [batch_size, 7, 7, 64]
			# Output Tensor Shape: [batch_size, 7 * 7 * 64]
			pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

			# Dense Layer
			# Densely connected layer with 1024 neurons
			# Input Tensor Shape: [batch_size, 7 * 7 * 64]
			# Output Tensor Shape: [batch_size, 1024]
			dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

			# Add dropout operation; 0.6 probability that element will be kept
			dropout = tf.layers.dropout(
					inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

			# Logits layer
			# Input Tensor Shape: [batch_size, 1024]
			# Output Tensor Shape: [batch_size, 10]
			logits = tf.layers.dense(inputs=dropout, units=10)

			#print("#"*10, logits.name)

			predictions = {
					# Generate predictions (for PREDICT and EVAL mode)
					"classes": tf.argmax(input=logits, axis=1),
					# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
					# `logging_hook`.
					"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
			}
			if mode == tf.estimator.ModeKeys.PREDICT:
				return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

			# Calculate Loss (for both TRAIN and EVAL modes)
			loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

			# Configure the Training Op (for TRAIN mode)
			if mode == tf.estimator.ModeKeys.TRAIN:
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
				train_op = optimizer.minimize(
						loss=loss,
						global_step=tf.train.get_global_step())
				return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

			# Add evaluation metrics (for EVAL mode)
			eval_metric_ops = {
					"accuracy": tf.metrics.accuracy(
							labels=labels, predictions=predictions["classes"])}
			return tf.estimator.EstimatorSpec(
					mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


		tf.logging.set_verbosity(tf.logging.INFO)

		# Load training and eval data
		self.mnist = tf.contrib.learn.datasets.load_dataset("mnist")
		train_data = self.mnist.train.images	# Returns np.array
		train_labels = np.asarray(self.mnist.train.labels, dtype=np.int32)

		"""
		np.set_printoptions(linewidth=1000)
		for i in range(20):
			print(np.reshape(np.array(train_data[i]*100, dtype=np.int), (28, 28)))
		#sys.exit(0)
		"""

		# Create the Estimator
		self.mnist_classifier = tf.estimator.Estimator(
				model_fn=cnn_model_fn, model_dir=model_dir)

		# Set up logging for predictions
		# Log the values in the "Softmax" tensor with label "probabilities"
		tensors_to_log = {"probabilities": "softmax_tensor"}
		self.logging_hook = tf.train.LoggingTensorHook(
				tensors=tensors_to_log, every_n_iter=50)

		self.batch_size = 100

		# Train the model
		self.train_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": train_data},
				y=train_labels,
				batch_size=self.batch_size,
				num_epochs=None,
				shuffle=True)

		self.model_dir = model_dir

		self.fast_predict = None

	def train(self, n):

		self.mnist_classifier.train(
				input_fn=self.train_input_fn,
				steps=n,
				hooks=[self.logging_hook])


	def prepare_predict(self):

		checkpoint_file=tf.train.latest_checkpoint(self.model_dir)
		graph=tf.Graph()
		
		def reset_seed():
			return
			s = 1
			random.seed(s)
			np.random.seed(s)
			tf.set_random_seed(s)
			random_seed.set_random_seed(s)
			graph._next_id_counter = s
			graph.seed = s
			context.set_global_seed(s)

		reset_seed()

		with graph.as_default():
			reset_seed()
			session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement =False)
			sess = tf.Session(config = session_conf)
			reset_seed()
			with sess.as_default():
				reset_seed()
				saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
				saver.restore(sess, checkpoint_file)

				prediction = graph.get_tensor_by_name("softmax_tensor:0")
				#prediction = graph.get_tensor_by_name("dense_1/BiasAdd:0")
				input = graph.get_operation_by_name("Reshape").inputs[0]

				def fp(batch):
					with graph.as_default():
						reset_seed()
						with sess.as_default():
							reset_seed()
							return sess.run(prediction, feed_dict={input: batch})

				self.fast_predict = fp

			
	def image_shape(self):
		return (28, 28)

	def image_offset(self):
		return 1.0

	def image_scaling(self):
		return -1.0/255.0


	# image: 28x28 image scales [0,1]
	def predict(self, image):
		
		#print(np.array(image.reshape((28, 28))*9, dtype=np.int))


		image = np.array(image, dtype=np.float32)

		# translate center of mass to center of image
		image = np.reshape(image, (28, 28))
		c = sp.ndimage.measurements.center_of_mass(image)
		sp.ndimage.interpolation.shift(np.copy(image), [14-ci for ci in c], image, 1, 'constant', 0.0, False)
		image = image.ravel()
		image = image.reshape((1, len(image)))
		#image /= np.max(image)
		#print(c)

		#print(np.array(image.reshape((28, 28))*100, dtype=np.int))


		if not self.fast_predict is None:
			batch = np.tile(image, (self.batch_size, 1))
			results = self.fast_predict(batch)
			result = np.mean(np.array(results), axis=0)
			return result
	

		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": image},
				num_epochs=1,
				shuffle=False)

		eval_results = self.mnist_classifier.predict(input_fn=eval_input_fn)

		# TODO: average probabilities
		for r in eval_results:
			 return r['probabilities']

		raise RuntimeError("problem here")

	def eval(self):

		eval_data = self.mnist.test.images	# Returns np.array
		eval_labels = np.asarray(self.mnist.test.labels, dtype=np.int32)

		# Evaluate the model and print results
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
				x={"x": eval_data},
				y=eval_labels,
				num_epochs=1,
				shuffle=False)
		eval_results = self.mnist_classifier.evaluate(input_fn=eval_input_fn)
		print(eval_results)


if __name__ == "__main__":
	c = TensorFlowMNIST_Classifier()
	c.train(100)
	c.predict(0)


