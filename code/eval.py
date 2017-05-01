#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime

from tqdm import tqdm

import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels('./../data/eval_questions.txt', './../data/eval_answers.txt')
    y_test = np.argmax(y_test, axis=1)
    # print x_raw, y_test
else:
    x_raw = ["Is this watch waterproof?", "everything is off."]
    y_test = [0, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir,
                          "/Users/sampanna.kahu/PycharmProjects/question_detection/code/runs/1491745147", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

model_file = '/Users/sampanna.kahu/PycharmProjects/question_detection/code/runs/1491745147/checkpoints/model-830'

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(model_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(model_file))
        saver.restore(sess, model_file)

        print "\nModel restored."

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.generate_batch_iterator(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        print "\nBatching done."
        # Collect the predictions here
        all_predictions = []
        print batches

        for x_test_batch in batches:
            # print str(x_test_batch)
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
print 'all predictions', str(all_predictions)
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
