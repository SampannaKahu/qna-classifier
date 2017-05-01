import os

from flask import Flask, render_template, request, url_for, current_app, send_from_directory
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import data_helpers

app = Flask(__name__)


@app.route('/<path:path>')
def send_html(path):
    return send_from_directory('html', path)


@app.route('/moderate', methods=['POST'])
def someThing():
    print 'Received test for moderation: ' + str(request.form['sentence'])
    x_raw = [str(request.form['sentence'])]
    x_raw[0] = data_helpers.clean_string(x_raw[0])
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    batches = data_helpers.generate_batch_iterator(list(x_test), FLAGS.batch_size, 1, shuffle=False)
    for x_test_batch in batches:
        # print str(x_test_batch)
        batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        prediction = batch_predictions[0]
        print str(prediction)
        status = "unknown"
        if prediction == 0:
            status = "Question"
        else:
            status = "Answer"
    return status


if __name__ == "__main__":
    print "Importing parameters"
    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print "Parameters\n"
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print "Parameters import done."

    # Map data into vocabulary
    vocab_path = os.path.join("./code/runs/1491745147",
                              "vocab")
    model_file = './code/runs/1491745147/checkpoints/model-830'
    global vocab_processor
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    checkpoint_file = tf.train.latest_checkpoint(model_file)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        global sess
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(model_file))
            saver.restore(sess, model_file)
            print "\nModel restored."
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    # try to read port from os environment
    try:
        port = os.environ['PORT']
    except KeyError:
        port = 5000
    # set up logging
    import logging

    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    # Run the server
    app.run(threaded=True, host='0.0.0.0')
