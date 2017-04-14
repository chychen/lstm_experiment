from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_integer("seq_length", 256,
                            "")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("word_size", 256,
                            "")
tf.app.flags.DEFINE_integer("hidden_size", 500,
                            "")
tf.app.flags.DEFINE_integer("embedding_size", 100,
                            "")
tf.app.flags.DEFINE_integer("iterations", 10000,
                            "")

def train():
    pass

def main(_):
    """
    main
    """
    train()


if __name__ == "__main__":
    tf.app.run()
