# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a network.

Usage:
  BOARD_SIZE=19 python train.py tfrecord1 tfrecord2 tfrecord3
"""

import logging

from absl import app, flags
import numpy as np

import policy_value_net
import preprocessing
import utils

# See www.moderndescartes.com/essays/shuffle_viz for discussion on sizing
flags.DEFINE_integer('shuffle_buffer_size', 2000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_integer('steps_to_train', None,
                     'Number of training steps to take. If not set, iterates '
                     'once over training data.')

flags.DEFINE_string('export_path', None,
                    'Where to export the model after training.')

flags.DEFINE_bool('use_bt', False,
                  'Whether to use Bigtable as input.  '
                  '(Only supported with --use_tpu, currently.)')

# From dual_net.py
flags.declare_key_flag('work_dir')
flags.declare_key_flag('train_batch_size')


FLAGS = flags.FLAGS




def train(*tf_records: "Records to train on"):
    """Train on examples."""

    effective_batch_size = FLAGS.train_batch_size



def main(argv):
    """Train on examples and export the updated model weights."""
    tf_records = argv[1:]
    logging.info("Training on %s records: %s to %s",
                 len(tf_records), tf_records[0], tf_records[-1])
    with utils.logged_timer("Training"):
        train(*tf_records)
    if FLAGS.export_path:
        dual_net.export_model(FLAGS.export_path)


if __name__ == "__main__":
    app.run(main)
