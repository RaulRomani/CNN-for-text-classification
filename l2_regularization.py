#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn

from cnn_utils import *
from word_embedings import *

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")


# train data
# data = build_data_set_from_xml('datasets/intertass-train-tagged.xml')
data = build_data_set_from_xml('datasets/intertass-development-tagged.xml')
x_train, y_train = get_batch(data, 0, 506) #1008  #506

# test data
data = build_data_set_test_from_xml('datasets/intertass-test.xml','datasets/intertass-sentiment.qrel')
x_test, y_test = get_batch(data, 0, 1899)  #1899

# validation data
# data = build_data_set_from_xml('datasets/intertass-development-tagged.xml')
# x_test, y_test = get_batch(data, 0, 506)  #returns 501

num_classes = 4
y_train = np.eye(num_classes)[y_train]  #convert_to_one_hot
y_test  = np.eye(num_classes)[y_test]   #convert_to_one_hot   


print ("Data was loaded sucessfully")
print (x_test.shape)
print (y_test.shape)



l2_penalty_values = np.logspace(1, 7, num=13)

i = 1;
for l2_penalty in l2_penalty_values:
	FLAGS.l2_reg_lambda = l2_penalty

	print(str(i) + " VALIDATING WITH ")
	print("l2_penalty = " , FLAGS.l2_reg_lambda)

	# Training
	# ==================================================
	with tf.Graph().as_default():
	    session_conf = tf.ConfigProto(
	      allow_soft_placement = FLAGS.allow_soft_placement, #to avoid erros when running in differnet machines
	      log_device_placement = FLAGS.log_device_placement) #for seeing logs - debugging porpuses


	    sess = tf.Session(config=session_conf)
	    with sess.as_default():

	        #creates the model
	        cnn = TextCNN(
	            max_document_length = x_train.shape[1],
	            num_classes         = num_classes,
	            embedding_dim       = FLAGS.embedding_dim,
	            filter_sizes        = list(map(int, FLAGS.filter_sizes.split(","))),
	            num_filters         = FLAGS.num_filters,
	            l2_reg_lambda       = FLAGS.l2_reg_lambda)



	        # Define Training procedure - MINIMIZE COST
	        global_step    = tf.Variable(0, name="global_step", trainable=False)
	        optimizer      = tf.train.AdamOptimizer(1e-3)
	        grads_and_vars = optimizer.compute_gradients(cnn.loss)
	        train_op       = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


	        # Keep track of gradient values and sparsity (optional)
	        grad_summaries = []
	        for g, v in grads_and_vars:
	            if g is not None:
	                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
	                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
	                grad_summaries.append(grad_hist_summary)
	                grad_summaries.append(sparsity_summary)
	        grad_summaries_merged = tf.summary.merge(grad_summaries)


	        # Output directory for models and summaries
	        timestamp = str(int(time.time()))
	        out_dir   = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp)) # + str(i)
	        print("Writing to {}\n".format(out_dir))

	        # Summaries for loss and accuracy
	        loss_summary = tf.summary.scalar("loss", cnn.loss)
	        acc_summary  = tf.summary.scalar("accuracy", cnn.accuracy)

	        # Train Summaries
	        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
	        train_summary_dir = os.path.join(out_dir, "summaries", "train")
	        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	        # Dev summaries
	        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
	        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
	        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

	        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	        checkpoint_dir    = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	        if not os.path.exists(checkpoint_dir):
	            os.makedirs(checkpoint_dir)
	        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


	        # Initialize all variables
	        sess.run(tf.global_variables_initializer())

	        def train_step(x_batch, y_batch):
	            """
	            A single training step
	            """
	            feed_dict = {
	              cnn.input_x: x_batch,
	              cnn.input_y: y_batch,
	              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
	            }

	            #RUN MULTIPLE GRAPH INPUT -> OUTPUT. Is a 1 to 1 relationship
	            _, step, summaries, loss, accuracy = sess.run(
	                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
	                        feed_dict  )


	            # time_str = datetime.datetime.now().isoformat()
	            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
	            train_summary_writer.add_summary(summaries, step)

	        ### Test step?
	        def dev_step(x_batch, y_batch, writer=None):
	            """
	            Evaluates model on a dev set
	            """
	            feed_dict = {
	              cnn.input_x: x_batch,
	              cnn.input_y: y_batch,
	              cnn.dropout_keep_prob: 1.0
	            }
	            step, summaries, loss, accuracy = sess.run(
	                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
	                feed_dict)

	            time_str = datetime.datetime.now().isoformat()
	            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
	            if writer:
	                writer.add_summary(summaries, step)



	        i = i +1
	        seed = 3  
	        # Do the training loop
	        for epoch in range(FLAGS.num_epochs):

	            minibatch_cost = 0.
	            num_minibatches = int(x_train.shape[0] / FLAGS.batch_size) # number of minibatches of size minibatch_size in the train set
	            seed = seed + 1
	            minibatches = random_mini_batches(x_train, y_train, FLAGS.batch_size, seed)

	            for minibatch in minibatches:

	                # Select a minibatch
	                (x_batch, y_batch) = minibatch

	                # IMPORTANT: The line that runs the graph on a minibatch.
	                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
	                # _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
	                # minibatch_cost += temp_cost / num_minibatches

	                train_step(x_batch, y_batch)
	                current_step = tf.train.global_step(sess, global_step)

	                # PRINTING INFO per every N batchs
	                if current_step % FLAGS.evaluate_every == 0:
	                    print("\nEvaluation:")
	                    dev_step(x_test, y_test, writer=dev_summary_writer)
	                    print("")
	                if current_step % FLAGS.checkpoint_every == 0:
	                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
	                    print("Saved model checkpoint to {}\n".format(path))

	            # Print the cost every epoch
	            # if print_cost == True and epoch % 5 == 0:
	            #     print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
	            # if print_cost == True and epoch % 1 == 0:
	            #     costs.append(minibatch_cost)

