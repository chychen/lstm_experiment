from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string("train_dir", "/tmp/train_seq2seq",
                           "")
tf.app.flags.DEFINE_integer("input_length", 20,
                            "")
tf.app.flags.DEFINE_integer("train_length", 20,
                            "")
tf.app.flags.DEFINE_integer("test_length", 20,
                            "")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("word_size", 256 + 1,
                            "")
tf.app.flags.DEFINE_integer("hidden_size", 500,
                            "")
tf.app.flags.DEFINE_integer("embedding_size", 100,
                            "")
tf.app.flags.DEFINE_integer("iterations", 10000,
                            "")
tf.app.flags.DEFINE_integer("log_steps", 100,
                            "")
tf.app.flags.DEFINE_integer("lr", 0.5,
                            "")
tf.app.flags.DEFINE_integer("momentum", 0.9,
                            "")

FLAGS = tf.app.flags.FLAGS


def train():

    # sess = tf.InteractiveSession(config=tf.ConfigProto( # TODO: why InteractiveSession failed!!!!
    #     allow_soft_placement=True, log_device_placement=False))

    with tf.Session() as sess:

        encoder_inputs = [tf.placeholder(tf.int32, shape=(
            FLAGS.batch_size), name="inp%i" % t) for t in range(FLAGS.input_length)]
        decoder_inputs = [tf.zeros_like(encoder_inputs[0], dtype=np.int32,
                                        name="decoder_inputs%i" % t) for t in range(FLAGS.input_length)]
        labels = [tf.placeholder(tf.int32, shape=(
            FLAGS.batch_size), name="labels%i" % t) for t in range(FLAGS.input_length)]
        weights = []
        for length_idx in xrange(FLAGS.input_length):
            # weight = np.ones(FLAGS.batch_size, dtype=np.float32)
            if length_idx < (FLAGS.input_length - FLAGS.train_length):
                weight = np.zeros(FLAGS.batch_size, dtype=np.float32)
            else:
                weight = np.ones(FLAGS.batch_size, dtype=np.float32)
            weights.append(weight)

        cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_size)
        decoder_outputs, decders_memory = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            encoder_inputs=encoder_inputs,
            decoder_inputs=decoder_inputs,
            cell=cell,
            num_encoder_symbols=FLAGS.word_size,
            num_decoder_symbols=FLAGS.word_size,
            embedding_size=FLAGS.embedding_size,
            feed_previous=False
        )

        loss = tf.contrib.legacy_seq2seq.sequence_loss(
            logits=decoder_outputs,
            targets=labels,
            weights=weights
        )
        tf.summary.scalar("loss", loss)
        magnitude = tf.sqrt(tf.reduce_sum(tf.square(decders_memory[1])))
        tf.summary.scalar("magnitude at t=1", magnitude)

        def accuracy(dec_outputs_batch, labels):
            anss = [tf.argmax(logits_t, axis=1) for logits_t in dec_outputs_batch]

            table = tf.equal(tf.to_int32(anss, name='ToInt32'), labels, name="equal")
            # only compare the sequence in the range (0, test_length)
            table = table[:FLAGS.test_length]
            acc_counter = tf.reduce_sum(tf.to_int32(table, name='ToInt32'))
            ac = acc_counter / (FLAGS.test_length * FLAGS.batch_size)

            return ac, anss

        acc, ans = accuracy(decoder_outputs, labels)
        tf.summary.scalar("accuracy", acc)

        summary_op = tf.summary.merge_all()

        optimizer = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)
        train_op = optimizer.minimize(loss)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        def train_batch():
            X = []
            for _ in range(FLAGS.batch_size):
                temp = []
                for i in range(FLAGS.input_length):
                    if i < (FLAGS.input_length - FLAGS.train_length):
                        temp.append(0)
                    else:
                        temp.append(np.random.randint(1, FLAGS.word_size))
                X.append(temp)

            Y = X[:]

            # show result as one seqence
            # print ("Train_X::", X[0])
            # print ("LABEL_Y::", Y[0])

            # Dimshuffle to seq_len * batch_size
            X = (np.array(X).T)
            Y = (np.array(Y).T)

            feed_dict = {encoder_inputs[t]: X[t]
                         for t in range(FLAGS.input_length)}
            feed_dict.update({labels[t]: Y[t]
                              for t in range(FLAGS.input_length)})

            _, loss_t, summary = sess.run(
                [train_op, loss, summary_op], feed_dict)


            return loss_t, summary

        def test_result():
            X_batchs = []
            for _ in range(FLAGS.batch_size):
                temp = []
                for i in range(FLAGS.input_length):
                    if i < FLAGS.test_length:
                        temp.append(np.random.randint(1, FLAGS.word_size))
                    else:
                        temp.append(0)
                X_batchs.append(temp)

            Y = X_batchs[:]
            # Dimshuffle to seq_len * batch_size
            X_batchs = (np.array(X_batchs).T)
            Y = (np.array(Y).T)

            feed_dict = {encoder_inputs[t]: X_batchs[t]
                         for t in range(FLAGS.input_length)}
            feed_dict.update({labels[t]: Y[t]
                              for t in range(FLAGS.input_length)})
            answer, accc, accc_summary = sess.run(
                [ans, acc, summary_op], feed_dict)

            # show result as one seqence
            X_batchs = np.array(X_batchs).T
            answer = np.array(answer).T
            print ("X_batch::", X_batchs[0])
            print ("answerr::", answer[0])
            
            return accc, accc_summary

        for t in range(FLAGS.iterations):
            loss_t, summary = train_batch()
            summary_writer.add_summary(summary, t)
            if t % FLAGS.log_steps is 0 or t == FLAGS.iterations - 1:
                acccc, acccc_summary = test_result()
                summary_writer.add_summary(acccc_summary, t)
                print ("steppppp:", t)
                print ("lossssss:", loss_t)
                print ("accuracy:", acccc)

        summary_writer.flush()


def main(_):
    """
    main
    """
    train()


if __name__ == "__main__":
    tf.app.run()
