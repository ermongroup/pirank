# Copyright 2021 The TensorFlow Ranking Authors.
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

r"""TF Ranking sample code for LETOR datasets in LibSVM format.
WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.
A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:
<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]
For example:
1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76
In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------
Sample command lines:
OUTPUT_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train.txt && \
VALI=tensorflow_ranking/examples/data/vali.txt && \
TEST=tensorflow_ranking/examples/data/test.txt && \
rm -rf $OUTPUT_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_libsvm_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_libsvm_py_binary \
--train_path=$TRAIN \
--vali_path=$VALI \
--test_path=$TEST \
--output_dir=$OUTPUT_DIR \
--num_features=136
You can use TensorBoard to display the training results stored in $OUTPUT_DIR.
Notes:
  * Use --alsologtostderr if the output is not printed into screen.
  * In addition, you can enable multi-objective learning by adding the following
  flags: --secondary_loss=<the secondary loss key>.
"""

from absl import flags

import numpy as np
import six

import submitit
import os
import sys
from neuralsort.tf import util

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_ranking as tfr

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", None, "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["1024", "512", "256"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string(
    "secondary_loss", None, "The RankingLossKey for the secondary loss for "
                            "multi-objective learning.")
flags.DEFINE_float(
    "secondary_loss_weight", 0.5, "The weight for the secondary loss in "
                                  "multi-objective learning.")

# Additional learning options
flags.DEFINE_string("optimizer", "Adagrad", "The optimizer for gradient descent.")
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to train, set 0 to just test')
# flags.DEFINE_integer('early_stopping', False, 'Use early stopping')

# NeuralSort-related
flags.DEFINE_float('tau', 5, 'temperature (dependent meaning)')
flags.DEFINE_float('taustar', 1e-10, 'Temperature to use for trues (hard or soft sort)')
flags.DEFINE_float('lr', 1e-4, 'initial learning rate')

# Submit-related
flags.DEFINE_boolean('submit', False, 'use cluster')
flags.DEFINE_string('exp', 'dummy', 'experiment name')

## PiRank-related
flags.DEFINE_string("loss_fn", "pirank_simple_loss",
                    "The loss function to use (either a TFR RankingLossKey, or loss function from the script).")
flags.DEFINE_boolean('ste', True, 'Whether to use the Straight-Through Estimator')
flags.DEFINE_integer('ndcg_k', 15, 'NDCG@k cutoff when using NS-NDCG loss')


# NS addition
if FLAGS.submit:
    FLAGS.exp = FLAGS.output_dir.split('/')[-1]
    print(FLAGS.output_dir.split('/'))

flag_dict = {}
for attr,flag_obj in FLAGS.__flags.items():
    flag_dict[attr] = getattr(FLAGS, attr)
print(flag_dict)

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"


def _use_multi_head():
    """Returns True if using multi-head."""
    return flag_dict['secondary_loss'] is not None


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = ["{}".format(i + 1) for i in range(flag_dict['num_features'])]
    return {
        name:
            tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
        for name in feature_names
    }


def load_libsvm_data(path, list_size):
    """Returns features and labels in numpy.array."""

    def _parse_line(line):
        """Parses a single line in LibSVM format."""
        tokens = line.split("#")[0].split()
        assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
        label = float(tokens[0])
        qid = tokens[1]
        kv_pairs = [kv.split(":") for kv in tokens[2:]]
        features = {k: float(v) for (k, v) in kv_pairs}
        return qid, features, label

    tf.compat.v1.logging.info("Loading data from {}".format(path))

    # The 0-based index assigned to a query.
    qid_to_index = {}
    # The number of docs seen so far for a query.
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of queries.
    feature_map = {k: [] for k in example_feature_columns()}
    label_list = []
    total_docs = 0
    discarded_docs = 0
    with open(path, "rt") as f:
        for line in f:
            qid, features, label = _parse_line(line)
            if qid not in qid_to_index:
                # Create index and allocate space for a new query.
                qid_to_index[qid] = len(qid_to_index)
                qid_to_ndoc[qid] = 0
                for k in feature_map:
                    feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
                label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
            total_docs += 1
            batch_idx = qid_to_index[qid]
            doc_idx = qid_to_ndoc[qid]
            qid_to_ndoc[qid] += 1
            # Keep the first 'list_size' docs only.
            if doc_idx >= list_size:
                discarded_docs += 1
                continue
            for k, v in six.iteritems(features):
                assert k in feature_map, "Key {} not found in features.".format(k)
                feature_map[k][batch_idx][doc_idx, 0] = v
            label_list[batch_idx][doc_idx] = label

    tf.compat.v1.logging.info("Number of queries: {}".format(len(qid_to_index)))
    tf.compat.v1.logging.info(
        "Number of documents in total: {}".format(total_docs))
    tf.compat.v1.logging.info(
        "Number of documents discarded: {}".format(discarded_docs))

    # Convert everything to np.array.
    for k in feature_map:
        feature_map[k] = np.array(feature_map[k])
    return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
    """Set up training input in batches."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _train_input_fn():
        """Defines training input fn."""
        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in six.iteritems(features)
        }
        if _use_multi_head():
            placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            labels_placeholder = {
                _PRIMARY_HEAD: placeholder,
                _SECONDARY_HEAD: placeholder,
            }
        else:
            labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices(
            (features_placeholder, labels_placeholder))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        if _use_multi_head():
            feed_dict = {
                labels_placeholder[head_name]: labels
                for head_name in labels_placeholder
            }
        else:
            feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(features, labels):
    """Set up eval inputs in a single batch."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _eval_input_fn():
        """Defines eval input fn."""
        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in six.iteritems(features)
        }
        if _use_multi_head():
            placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            labels_placeholder = {
                _PRIMARY_HEAD: placeholder,
                _SECONDARY_HEAD: placeholder,
            }
        else:
            labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensors(
            (features_placeholder, labels_placeholder))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        if _use_multi_head():
            feed_dict = {
                labels_placeholder[head_name]: labels
                for head_name in labels_placeholder
            }
        else:
            feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _eval_input_fn, iterator_initializer_hook


def make_serving_input_fn():
    """Returns serving input fn to receive tf.Example."""
    feature_spec = tf.feature_column.make_parse_example_spec(
        example_feature_columns().values())
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)


def make_transform_fn():
    """Returns a transform_fn that converts features to dense Tensors."""

    def _transform_fn(features, mode):
        """Defines transform_fn."""
        if mode == tf.estimator.ModeKeys.PREDICT:
            # We expect tf.Example as input during serving. In this case, group_size
            # must be set to 1.
            if flag_dict['group_size'] != 1:
                raise ValueError(
                    "group_size should be 1 to be able to export model, but get %s" %
                    flag_dict['group_size'])
            context_features, example_features = (
                tfr.feature.encode_pointwise_features(
                    features=features,
                    context_feature_columns=None,
                    example_feature_columns=example_feature_columns(),
                    mode=mode,
                    scope="transform_layer"))
        else:
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=None,
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")

        return context_features, example_features

    return _transform_fn


def make_score_fn():
    """Returns a groupwise score fn to build `EstimatorSpec`."""

    def _score_fn(unused_context_features, group_features, mode, unused_params,
                  unused_config):
        """Defines the network to score a group of documents."""
        with tf.compat.v1.name_scope("input_layer"):
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(example_feature_columns())
            ]
            input_layer = tf.concat(group_input, 1)
            tf.compat.v1.summary.scalar("input_sparsity",
                                        tf.nn.zero_fraction(input_layer))
            tf.compat.v1.summary.scalar("input_max",
                                        tf.reduce_max(input_tensor=input_layer))
            tf.compat.v1.summary.scalar("input_min",
                                        tf.reduce_min(input_tensor=input_layer))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.compat.v1.layers.batch_normalization(
            input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in flag_dict['hidden_layer_dims']):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                        tf.nn.zero_fraction(cur_layer))
        cur_layer = tf.compat.v1.layers.dropout(
            cur_layer, rate=flag_dict['dropout_rate'], training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=flag_dict['group_size'])
        if _use_multi_head():
            # Duplicate the logits for both heads.
            return {_PRIMARY_HEAD: logits, _SECONDARY_HEAD: logits}
        else:
            return logits

    return _score_fn


def neuralsort_permutation_loss(labels, logits, features):
    '''
    Modeled after tensorflow_ranking/python/losses.py _loss_fn
    :param labels: True scores
    :param logits: Scores from the NN
    :param tau: Temperature parameter
    :return:
    '''
    tau = flag_dict["tau"]
    taustar = flag_dict["taustar"]
    ste = flag_dict["ste"]

    false_tensor = tf.convert_to_tensor(False)
    evaluation = tf.placeholder_with_default(false_tensor, ())
    temperature = tf.cond(evaluation,
                          false_fn=lambda: tf.convert_to_tensor(
                              tau, dtype=tf.float32),
                          true_fn=lambda: tf.convert_to_tensor(
                              taustar, dtype=tf.float32)  # simulate hard sort
                          )

    labels = tf.expand_dims(labels, 2)
    labels = tf.stop_gradient(labels)
    logits = tf.expand_dims(logits, 2)
    P_true = util.neuralsort(labels, taustar)
    P_true = tf.stop_gradient(P_true)
    if ste:
        P_hat_backward = util.neuralsort(logits, temperature)
        P_hat_forward = util.neuralsort(logits, taustar)
        P_hat = P_hat_backward + tf.stop_gradient(P_hat_forward - P_hat_backward)
    else:
        P_hat = util.neuralsort(logits, temperature)

    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=P_true, logits=tf.log(P_hat + 1e-10), dim=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)
    return loss


# @ex.capture
def pirank_arp_simple_loss(labels, logits, features):
    '''
    Modeled after tensorflow_ranking/python/losses.py _loss_fn
    :param labels: True scores
    :param logits: Scores from the NN
    :param tau: Temperature parameter
    :return:
    '''
    tau = flag_dict["tau"]
    taustar = flag_dict["taustar"]
    ste = flag_dict["ste"]

    with tf.name_scope("pirank_scope"):
        false_tensor = tf.convert_to_tensor(False)
        evaluation = tf.placeholder_with_default(false_tensor, ())

        temperature = tf.cond(evaluation,
                              false_fn=lambda: tf.convert_to_tensor(
                                  tau, dtype=tf.float32),
                              true_fn=lambda: tf.convert_to_tensor(
                                  1e-10, dtype=tf.float32)  # simulate hard sort
                              )

        is_label_valid = tfr.utils.is_label_valid(labels)
        labels = tf.where(is_label_valid, labels, tf.zeros_like(labels))
        logits = tf.where(is_label_valid, logits, -1e-6 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
        logits = tf.expand_dims(logits, 2, name="logits")
        labels = tf.expand_dims(labels, 2, name="labels")
        list_size = tf.shape(input=labels)[1]

        if ste:
            P_hat_backward = util.neuralsort(logits, temperature)
            P_hat_forward = util.neuralsort(logits, taustar)
            P_hat = P_hat_backward + tf.stop_gradient(P_hat_forward - P_hat_backward)
        else:
            P_hat = util.neuralsort(logits, temperature)
        P_hat = tf.identity(P_hat, name="P_hat")
        labels = tf.cast(labels, dtype=tf.float32, name="labels")
        position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32, name="arp_position")
        sorted_labels = tf.linalg.matmul(P_hat, labels)
        numerator = tf.reduce_sum(sorted_labels * position, axis=1, name="arp_numerator")
        denominator = tf.reduce_sum(labels, axis=1, name="arp_denominator")
        loss = numerator / (1 + denominator)
        return tf.reduce_sum(loss)


def pirank_simple_mean_loss(labels, logits, features):
    '''
    Modeled after tensorflow_ranking/python/losses.py _loss_fn
    :param labels: True scores
    :param logits: Scores from the NN
    :param tau: Temperature parameter
    :return:
    '''
    tau = flag_dict["tau"]
    taustar = flag_dict["taustar"]
    ndcg_k = flag_dict["ndcg_k"]
    ste = flag_dict["ste"]

    with tf.name_scope("pirank_scope"):
        false_tensor = tf.convert_to_tensor(False)
        evaluation = tf.placeholder_with_default(false_tensor, ())

        temperature = tf.cond(evaluation,
                              false_fn=lambda: tf.convert_to_tensor(
                                  tau, dtype=tf.float32),
                              true_fn=lambda: tf.convert_to_tensor(
                                  1e-10, dtype=tf.float32)  # simulate hard sort
                              )

        is_label_valid = tfr.utils.is_label_valid(labels)
        labels = tf.where(is_label_valid, labels, tf.zeros_like(labels))
        logits = tf.where(is_label_valid, logits, -1e-6 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
        logits = tf.expand_dims(logits, 2, name="logits")
        labels = tf.expand_dims(labels, 2, name="labels")
        list_size = tf.shape(input=labels)[1]

        if ste:
            P_hat_backward = util.neuralsort(logits, temperature)
            P_hat_forward = util.neuralsort(logits, taustar)
            P_hat = P_hat_backward + tf.stop_gradient(P_hat_forward - P_hat_backward)
        else:
            P_hat = util.neuralsort(logits, temperature)
        P_hat = tf.identity(P_hat, name="P_hat")
        label_powers = tf.pow(2.0, tf.cast(labels, dtype=tf.float32), name="label_powers") - 1.0
        sorted_powers = tf.linalg.matmul(P_hat, label_powers)
        numerator = tf.reduce_sum(sorted_powers, axis=-1, name="dcg_numerator")
        position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32, name="dcg_position")
        denominator = tf.math.log(position + 1, name="dcg_denominator")
        dcg = numerator / denominator
        dcg = dcg[:, :ndcg_k]
        dcg = tf.reduce_sum(input_tensor=dcg, axis=1, keepdims=True, name="dcg")

        P_true = util.neuralsort(labels, 1e-10)
        ideal_sorted_labels = tf.linalg.matmul(P_true, labels)
        ideal_sorted_labels = tf.reduce_sum(ideal_sorted_labels, axis=-1,
                                            name="ideal_sorted_labels")
        numerator = tf.pow(2.0, tf.cast(ideal_sorted_labels, dtype=tf.float32),
                           name="ideal_dcg_numerator") - 1.0
        ideal_dcg = numerator / denominator
        ideal_dcg = ideal_dcg[:, :ndcg_k]
        ideal_dcg = tf.reduce_sum(ideal_dcg, axis=1, keepdims=True, name="ideal_dcg")
        def safe_div(numerator, denominator):
            return tf.where(
                tf.equal(denominator, 0),
                tf.zeros_like(numerator),
                tf.compat.v1.div(numerator, denominator))
        ndcg = tf.reduce_mean(1. - safe_div(dcg, 1e-6 + ideal_dcg))
        # ndcg = safe_div(tf.reduce_sum(ideal_dcg * safe_div(dcg, ideal_dcg_k)), tf.reduce_sum(ideal_dcg))
        return ndcg


def pirank_simple_loss(labels, logits, weights):
    '''
    Modeled after tensorflow_ranking/python/losses.py _loss_fn
    :param labels: True scores
    :param logits: Scores from the NN
    :param tau: Temperature parameter
    :return:
    '''
    tau = flag_dict["tau"]
    taustar = flag_dict["taustar"]
    ndcg_k = flag_dict["ndcg_k"]
    ste = flag_dict["ste"]

    with tf.name_scope("pirank_scope"):
        false_tensor = tf.convert_to_tensor(False)
        evaluation = tf.placeholder_with_default(false_tensor, ())

        temperature = tf.cond(evaluation,
                              false_fn=lambda: tf.convert_to_tensor(
                                  tau, dtype=tf.float32),
                              true_fn=lambda: tf.convert_to_tensor(
                                  1e-10, dtype=tf.float32)  # simulate hard sort
                              )

        is_label_valid = tfr.utils.is_label_valid(labels)
        labels = tf.where(is_label_valid, labels, tf.zeros_like(labels))
        logits = tf.where(is_label_valid, logits, -1e-6 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
        logits = tf.expand_dims(logits, 2, name="logits")
        labels = tf.expand_dims(labels, 2, name="labels")
        list_size = tf.shape(input=labels)[1]

        if ste:
            P_hat_backward = util.neuralsort(logits, temperature)
            P_hat_forward = util.neuralsort(logits, taustar)
            P_hat = P_hat_backward + tf.stop_gradient(P_hat_forward - P_hat_backward)
        else:
            P_hat = util.neuralsort(logits, temperature)
        P_hat = tf.identity(P_hat, name="P_hat")
        label_powers = tf.pow(2.0, tf.cast(labels, dtype=tf.float32), name="label_powers") - 1.0
        sorted_powers = tf.linalg.matmul(P_hat, label_powers)
        numerator = tf.reduce_sum(sorted_powers, axis=-1, name="dcg_numerator")
        position = tf.cast(tf.range(1, list_size + 1), dtype=tf.float32, name="dcg_position")
        denominator = tf.math.log(position + 1, name="dcg_denominator")
        dcg = numerator / denominator
        dcg = dcg[:, :ndcg_k]
        dcg = tf.reduce_sum(input_tensor=dcg, axis=1, keepdims=True, name="dcg")

        P_true = util.neuralsort(labels, 1e-10)
        ideal_sorted_labels = tf.linalg.matmul(P_true, labels)
        ideal_sorted_labels = tf.reduce_sum(ideal_sorted_labels, axis=-1,
                                            name="ideal_sorted_labels")
        numerator = tf.pow(2.0, tf.cast(ideal_sorted_labels, dtype=tf.float32),
                           name="ideal_dcg_numerator") - 1.0
        ideal_dcg = numerator / (1e-10 + denominator)
        ideal_dcg = ideal_dcg[:, :ndcg_k]
        ideal_dcg = tf.reduce_sum(ideal_dcg, axis=1, keepdims=True, name="dcg")
        ndcg = tf.reduce_sum(dcg) / (1e-10 + tf.reduce_sum(ideal_dcg))
        return 1. - ndcg


def get_eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
            tfr.metrics.RankingMetricKey.MRR,
            tfr.metrics.RankingMetricKey.ARP,
            tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
        ]
    })
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10, 15]
    })
    return metric_fns


def train_and_eval():
    """Train and Evaluate."""
    import os
    os.environ['IS_TEST'] = ''

    if flag_dict['num_train_steps'] > 0:
        features, labels = load_libsvm_data(flag_dict['train_path'],
                                            flag_dict['list_size'])
        train_input_fn, train_hook = get_train_inputs(features, labels, flag_dict['train_batch_size'])

        features_vali, labels_vali = load_libsvm_data(flag_dict['vali_path'],
                                                      flag_dict['list_size'])
        vali_input_fn, vali_hook = get_eval_inputs(features_vali, labels_vali)

    features_test, labels_test = load_libsvm_data(flag_dict['test_path'],
                                                  flag_dict['list_size'])
    test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)

    if flag_dict['optimizer'].lower() == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=flag_dict['learning_rate'])
    elif flag_dict['optimizer'].lower() == 'adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=flag_dict['learning_rate'])

    loss_fn = flag_dict['loss_fn']
    if loss_fn == 'pirank_simple_loss':
        loss_function = pirank_simple_loss
    elif loss_fn == 'neuralsort_permutation_loss':
        loss_function = neuralsort_permutation_loss
    elif loss_fn == 'pirank_arp_simple_loss':
        loss_function = pirank_arp_simple_loss
    elif loss_fn == 'lambda_rank_loss':
        loss_function = tfr.losses.make_loss_fn('pairwise_logistic_loss',
                                                lambda_weight=tfr.losses.create_ndcg_lambda_weight(topn=ndcg_k))
    else:
        loss_function = tfr.losses.make_loss_fn(loss_fn)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([minimize_op, update_ops])
        return train_op

    if _use_multi_head():
        primary_head = tfr.head.create_ranking_head(
            loss_fn=loss_function,
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn,
            name=_PRIMARY_HEAD)
        secondary_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(flag_dict['secondary_loss']),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn,
            name=_SECONDARY_HEAD)
        ranking_head = tfr.head.create_multi_ranking_head(
            [primary_head, secondary_head], [1.0, flag_dict['secondary_loss_weight']])
    else:
        ranking_head = tfr.head.create_ranking_head(
            loss_fn=loss_function,
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn)

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=flag_dict['group_size'],
            transform_fn=make_transform_fn(),
            ranking_head=ranking_head),
        config=tf.estimator.RunConfig(
            flag_dict['output_dir'], save_checkpoints_steps=1000))

    if flag_dict['num_train_steps'] > 0:
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            hooks=[train_hook],
            max_steps=flag_dict['num_train_steps'])
        # Export model to accept tf.Example when group_size = 1.
        if flag_dict['group_size'] == 1:
            vali_spec = tf.estimator.EvalSpec(
                input_fn=vali_input_fn,
                hooks=[vali_hook],
                steps=1,
                # exporters=tf.estimator.LatestExporter(
                #     "latest_exporter",
                #     serving_input_receiver_fn=make_serving_input_fn()),
                exporters=tf.estimator.BestExporter(
                    "best_exporter",
                    serving_input_receiver_fn=make_serving_input_fn()),
                start_delay_secs=0,
                throttle_secs=30)
        else:
            vali_spec = tf.estimator.EvalSpec(
                input_fn=vali_input_fn,
                hooks=[vali_hook],
                steps=1,
                start_delay_secs=0,
                throttle_secs=30)

        # if flag_dict['early_stopping']:
        #     iterator_initializer_hook = tf.estimator.experimental.stop_if_no_decrease_hook()
        # else:
        #
        tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

    # Evaluate on the test data.
    print('Testing')
    os.environ['IS_TEST'] = '1'
    estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        pass

    def __call__(self):
        train_and_eval()



def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


    if flag_dict["submit"]:
        # if flag_dict['submit']:
        # slurm params
        slurm_mem =  80
        slurm_timeout =  72
        slurm_partition = "learnlab"
        num_gpus = 1
        num_workers = 0
        num_nodes = 1
        print(flag_dict["submit"], flag_dict["exp"])
        logdir =  "./logs/" + flag_dict["exp"]

        executor = submitit.AutoExecutor(
            folder=os.path.join(logdir, "slurm"), slurm_max_num_timeout=3
        )
        executor.update_parameters(
            name=flag_dict["exp"],
            mem_gb=slurm_mem,
            timeout_min=slurm_timeout * 60,
            slurm_partition=slurm_partition,
            gpus_per_node=num_gpus,
            cpus_per_task=(num_workers + 1),
            tasks_per_node=1,
            nodes=num_nodes
        )

        job = executor.submit(Runner())
        print('Submitted job:', job.job_id)
    # If submit turned off just use what's in _call()
    else:
        Runner()()


if __name__ == "__main__":
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("vali_path")
    flags.mark_flag_as_required("test_path")
    flags.mark_flag_as_required("output_dir")

    # main()
    tf.compat.v1.app.run()
    # Goes inside main after