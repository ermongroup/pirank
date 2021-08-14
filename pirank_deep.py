from sacred import Experiment
from sacred.stflow import LogFileWriter
from sacred.observers import SqlObserver, MongoObserver
# Uncomment if using GPU
import setGPU
from tfdeterminism import patch
patch()
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_ranking as tfr
from neuralsort import tf as nstf
from neuralsort.tf import util
import time
import math
import six
from tensorflow.python import debug as tf_debug


ex = Experiment('PiRank')
hook_train = None
hook_vali = None
hook_test = None
# Uncomment these if you like to use TensorBoard debugging
# hook = tf_debug.TensorBoardDebugHook("localhost:7001")
# hook_train = hook_vali = hook_test = tf_debug.TensorBoardDebugHook("localhost:7000")
tf.enable_eager_execution()
tf.executing_eagerly()
tf.logging.set_verbosity(tf.logging.INFO)


# NeuralSort-related
flags = tf.app.flags
flags.DEFINE_float('tau', 5, 'Temperature')
flags.DEFINE_float('taustar', 1e-10, 'Temperature to use for trues and straight-through estimation')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to train, set 0 to just test')
flags.DEFINE_float('lr', 1e-4, 'initial learning rate')
# Training-related
flags.DEFINE_string("train_path", '/data/MSLR-WEB30K/Fold*/train.txt',
                    "Input file path used for training.")
flags.DEFINE_string("vali_path", '/data/MSLR-WEB30K/Fold*/vali.txt',
                    "Input file path used for validation.")
flags.DEFINE_string("test_path", '/data/MSLR-WEB30K/Fold*/test.txt',
                    "Input file path used for testing.")
flags.DEFINE_string("model_dir", None,
                    "Output directory for models.")
flags.DEFINE_integer("batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", None, "Number of steps for training.")
flags.DEFINE_integer("num_vali_steps", None, "Number of steps for validation.")
flags.DEFINE_integer("num_test_steps", None, "Number of steps for testing.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_string("optimizer", "Adam", "The optimizer for gradient descent.")
# TFR and architecture-related
flags.DEFINE_string("hidden_layers", "256,tanh,128,tanh,64,tanh",
                    "Hidden layers in the format 256,tanh,128,tanh,64,tanh")
flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("test_list_size", None, "List size used for testing. If not provided, same as train list size.")
flags.DEFINE_integer("vali_list_size", None, "List size used for validation. If not provided, same as train list size.")
# Deep PiRank-related
flags.DEFINE_integer("merge_block_size", None, "Block size used if merging, None if not merging.")
flags.DEFINE_integer('top_k', None, 'Use a different Top-k for merging than final NDCG@k for loss')
flags.DEFINE_boolean('straight_backprop', False, 'Backpropagate on scores only through NS operator')
flags.DEFINE_boolean('full_loss', False, 'Use the complete loss at the end of merge')
flags.DEFINE_string('tau_scheme', None, 'Which scheme to use for temperature going deeper (default: constant)')
flags.DEFINE_string('data_generator', None, 'Data generator (default: TFR\'s libsvm); use this for synthetic generation')
flags.DEFINE_integer('num_queries', 30000, 'Number of queries for synthetic data generator')
flags.DEFINE_integer('num_query_features', 10, 'Number of columns used as factors for each query by synthetic data generator')
flags.DEFINE_integer('actual_list_size', None, 'Size of actual list per query in synthetic data generation')
flags.DEFINE_boolean('with_opa', True, 'Include pairwise metric OPA')
# All PiRank-related
flags.DEFINE_string("loss_fn", "pirank_deep_loss",
                    "The loss function to use (either a TFR RankingLossKey, or loss function from the script).")
flags.DEFINE_boolean('ste', True, 'Whether to use the Straight-Through Estimator')
flags.DEFINE_integer('ndcg_k', 15, 'NDCG@k when using NS-NDCG loss')
# Sacred-related
flags.DEFINE_boolean('d', False, 'Debutaging for Sacred')
flags.DEFINE_boolean('n', False, 'Experiment name for Sacred')
flags.DEFINE_boolean('c', False, 'Comment for Sacred')
flags.DEFINE_boolean('m', False, 'MongoDB for Sacred')
flags.DEFINE_boolean('s', False, 'SQL for Sacred')
# NS
FLAGS = flags.FLAGS


# Put all TF flags in Sacred configuration. Equiv to:
# ex.add_config({param: getattr(FLAGS, param) for param in dir(FLAGS)})
@ex.config
def sacred_config():
    for param in dir(FLAGS):
        if param != 'learning_rate':
            locals()[param] = getattr(FLAGS, param)


@ex.capture
def synthetic_data_generator(path_seed, num_features, list_size, num_queries,
                             actual_list_size, num_query_features, seed):
    # path_seed is used to generate different queries for vali/test
    # _seed is provided by Sacred and used for query
    # https://sacred.readthedocs.io/en/stable/randomness.html
    assert num_query_features <= num_features / 2
    if not actual_list_size:
        actual_list_size = list_size
    assert list_size <= actual_list_size

    num_doc_features = num_features - num_query_features
    def inner_generator():
        qrnd = np.random.RandomState(path_seed)
        for _ in range(num_queries):
            order = qrnd.choice(range(num_doc_features), num_query_features,
                                 replace=False)
            query = qrnd.gamma(1, 1, num_query_features)

            X = qrnd.randn(list_size, num_doc_features)
            y = np.floor((query * X[:, order]).sum(axis=1))
            y = np.where(y < 0, 0, y)
            y = np.where(y > 5, 5, y)
            query_features = np.repeat(np.expand_dims(query, 0), list_size, 0)
            X = np.concatenate([query_features, X], axis=1)

            if list_size < actual_list_size:
                # Mimic behavior from TFR inner_generator
                idx = np.random.choice(X.shape[0], list_size, replace=False)
                X = X[idx, :]
                y = y[idx]

            # Adapted from _libsvm_generate
            features = {}
            for fid in range(num_features):
                features[str(fid + 1)] = np.expand_dims(X[:, fid], -1)
            labels = y
            yield features, labels

    return inner_generator


# TFR
@ex.capture
def input_fn(path, num_features, list_size, batch_size, num_epochs, data_generator, seed):
    '''
    Takes a libSVM LETOR dataset and turns it into tensor format.
    # It is a list of dictionaries, one per query-document pair, where
    # each dictionary is a mapping from a feature ID to a feature value.
    '''
    if data_generator:
        assert data_generator in globals()
        data_generator = globals()[data_generator]
    else:
        data_generator = tfr.data.libsvm_generator

    dataset = tf.data.Dataset.from_generator(
        data_generator(path, num_features, list_size, seed=seed),
        output_types=(
            {str(k): tf.float32 for k in range(1,num_features+1)},
            tf.float32
        ),
        output_shapes=(
            {str(k): tf.TensorShape([list_size, 1])
             for k in range(1,num_features+1)},
            tf.TensorShape([list_size])
        )
    )
    if data_generator == tfr.data.libsvm_generator:
        dataset = dataset.shuffle(1000, seed=seed).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    # cf. mnist_input.get_iterator()
    return dataset.make_one_shot_iterator().get_next()


# Here we formulate a scoring function using a feed forward network.
# The function takes the features of a single example (i.e., query-document pair)
# and produces a relevance score.
@ex.capture
def example_feature_columns(num_features):
    """Returns the example feature columns."""
    feature_names = [
        "%d" % (i + 1) for i in range(0, num_features)
    ]
    return {
        name: tf.feature_column.numeric_column(
            name, shape=(1,), default_value=0.0) for name in feature_names
    }


def parse_arch(a):
    # a = '256,tanh,128,tanh,64,tanh'
    a = a.lower().split(',')
    b = [int(a[i]) for i in range(0,len(a),2)]
    c = [a[i] for i in range(1,len(a),2)]
    assert len(b) == len(c)
    return list(zip(b, c))


# Build the scoring NN for LETOR data
@ex.capture
def make_score_fn(hidden_layers):
    def _score_fn(context_features, per_example_features, mode, params, config):
        """Defines the network to score a document."""
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(per_example_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for layer in parse_arch(hidden_layers):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer[0],
                activation=layer[1])

        # units=1 ensures permutation invariance
        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


@ex.capture
def neuralsort_permutation_loss(labels, logits, features, tau, taustar, ste):
    '''
    Modeled after tensorflow_ranking/python/losses.py _loss_fn
    :param labels: True scores
    :param logits: Scores from the NN
    :param tau: Temperature parameter
    :return:
    '''
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
        labels=P_true, logits=tf.log(P_hat + 1e-20), dim=2)
    losses = tf.reduce_mean(losses, axis=-1)
    loss = tf.reduce_mean(losses)
    return loss


@ex.capture
def pirank_deep_loss(labels, logits, features, tau, taustar, ndcg_k, top_k, list_size,
                     merge_block_size, ste, straight_backprop, full_loss, tau_scheme,
                     seed, depth=1):
    if not top_k:
        top_k = ndcg_k

    # Just here we want them the same
    assert top_k == ndcg_k

    assert merge_block_size >= top_k
    assert list_size % merge_block_size == 0
    assert list_size >= merge_block_size
    assert merge_block_size % top_k == 0

    div = merge_block_size // top_k
    with tf.name_scope("merge_nr_scope_depth_{}".format(depth)):
        is_label_valid = tfr.utils.is_label_valid(labels)
        labels = tf.where(is_label_valid, labels, tf.zeros_like(labels), name='labels')
        logits = tf.where(is_label_valid, logits, -1e-6 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=1, keepdims=True), name='logits')

        if list_size > merge_block_size:
            # Merge top_k logits from previous layers of merge
            logits, phat_tops = map(list(zip(*
                 [pirank_deep_loss(labels[:, i:(i + list_size // div)],
                                   logits[:, i:(i + list_size // div)],
                                   None, tau, taustar, ndcg_k, top_k,
                                   list_size // div, merge_block_size,
                                   ste, full_loss, depth=depth+1)
                  for i in range(0, list_size, list_size // div)])).__getitem__, [0, 1])
            logits = tf.concat(list(logits), 1, name='merged_logits')
            phat_tops = list(phat_tops)
            if not full_loss:
                labels = tf.concat(phat_tops, 1, name='merged_labels')

        # Get P_hat
        logits = tf.expand_dims(logits, 2, name="logits_exp")
        labels = tf.expand_dims(labels, 2, name="labels_exp")
        tau = tau ** depth if tau_scheme == 'square' else tau
        if ste or depth == 1:
            P_hat_backward = util.neuralsort(logits, tau)
            P_hat_backward = tf.identity(P_hat_backward, name="P_hat_backward")
            P_hat_forward = util.neuralsort(logits, taustar)
            P_hat = P_hat_backward + tf.stop_gradient(P_hat_forward - P_hat_backward)
        else:
            P_hat = util.neuralsort(logits, tau)
        P_hat = tf.identity(P_hat, name="P_hat")

        phat_top = P_hat[:, :top_k, :]
        if full_loss and list_size > merge_block_size:
            # Do the recursive product
            if phat_tops is not None:
                res = []
                for i, pt in enumerate(phat_tops):
                    l, h = i * top_k, (i+1) * top_k
                    res.append(tf.matmul(phat_top[:, :, l:h], pt))
                phat_top = tf.concat(res, 2)
                assert phat_top.shape[2] == list_size
        phat_top = tf.identity(phat_top, name='phat_top')

        if depth == 1:
            # Any deeper layers than last, return top_k
            label_powers = tf.pow(2.0, tf.cast(labels, dtype=tf.float32), name="label_powers") - 1.0
            sorted_powers = tf.matmul(phat_top, label_powers, name='sorted_powers')
            numerator = tf.reduce_sum(sorted_powers, axis=-1, name="dcg_numerator")
            position = tf.cast(tf.range(1, ndcg_k + 1), dtype=tf.float32, name="dcg_position")
            denominator = tf.math.log(position + 1, name="dcg_denominator")
            dcg = numerator / (1e-10 + denominator)
            dcg = tf.reduce_sum(input_tensor=dcg, axis=1, keepdims=True, name="dcg")
            labels = tf.squeeze(labels, 2)
            ideal_sorted_labels, _ = tf.nn.top_k(labels, k=ndcg_k, sorted=True)
            numerator = tf.pow(2.0, tf.cast(ideal_sorted_labels, dtype=tf.float32),
                               name="ideal_dcg_numerator") - 1.0
            ideal_dcg = numerator / (1e-10 + denominator)
            ideal_dcg = tf.reduce_sum(ideal_dcg, axis=1, keepdims=True, name="ideal_dcg")
            dcg = tf.where(tf.is_nan(dcg), tf.zeros_like(dcg), dcg)
            ideal_dcg = tf.where(tf.is_nan(ideal_dcg), tf.ones_like(ideal_dcg), ideal_dcg)
            ndcg = tf.reduce_sum(dcg) / (1e-10 + tf.reduce_sum(ideal_dcg))
            ndcg = tf.identity(ndcg, name='ndcg')
            return 1. - ndcg
        else:
            topk_logits = tf.matmul(phat_top, logits)
            if straight_backprop:
                topk_logits = tf.stop_gradient(topk_logits)
            topk_logits = tf.squeeze(topk_logits, 2, name="topk_logits")
            if not full_loss:
                topk_labels = tf.matmul(phat_top, labels)
                topk_labels = tf.squeeze(topk_labels, 2, name="topk_labels")
            return topk_logits, phat_top if full_loss else topk_labels


# TFR setup
@ex.capture
def eval_metric_fns(with_opa):
    """Returns a dict from name to metric functions.

    This can be customized as follows. Care must be taken when handling padded
    lists.

    def _auc(labels, predictions, features):
      is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
      clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
      clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
      return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
    metric_fns["auc"] = _auc

    Returns:
      A dict mapping from metric name to a metric function with above signature.
    """
    metric_fns = {}
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10, 15]
    })
    metric_fns.update({
        "metric/precision@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.PRECISION, topn=topn)
        for topn in [1, 3, 5, 10, 15]
    })
    metric_fns.update({
        "metric/%s" % metric: tfr.metrics.make_ranking_metric_fn(metric)
        for metric in [tfr.metrics.RankingMetricKey.NDCG, # Total NDCG
                       tfr.metrics.RankingMetricKey.MRR,
                       tfr.metrics.RankingMetricKey.ARP]
    })
    if with_opa:
        metric_fns.update({
            "metric/%s" % metric: tfr.metrics.make_ranking_metric_fn(metric)
            for metric in [tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY]
    })

    return metric_fns


@ex.capture
def get_estimator(hparams, optimizer, loss_fn, model_dir, ndcg_k, seed):
    """Create a ranking estimator.

    Args:
      hparams: (tf.contrib.training.HParams) a hyperparameters object.

    Returns:
      tf.learn `Estimator`.
    """
    if loss_fn in globals():
        loss_function = globals()[loss_fn]
    elif loss_fn == 'lambda_rank_loss':
        loss_function = tfr.losses.make_loss_fn('pairwise_logistic_loss',
                                   lambda_weight=tfr.losses.create_ndcg_lambda_weight(topn=ndcg_k),
                                   seed=seed)
    elif loss_fn == 'list_mle_loss':
        loss_function = tfr.losses.make_loss_fn('list_mle_loss',
                                                lambda_weight=tfr.losses.create_ndcg_lambda_weight(topn=ndcg_k),
                                                seed=seed)
    else:
        loss_function = tfr.losses.make_loss_fn(loss_fn, seed=seed)

    def _train_op_fn(loss_value):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss_value,
            global_step=tf.train.get_global_step(),
            learning_rate=hparams.learning_rate,
            optimizer=optimizer)

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=loss_function,
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=_train_op_fn)

    return tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=1,
            transform_fn=None,
            ranking_head=ranking_head),
        params=hparams,
        config=tf.estimator.RunConfig(model_dir=model_dir, tf_random_seed=seed))


# Automain is captured by Sacred
@ex.automain
@LogFileWriter(ex)
def run_experiment(learning_rate, train_path, vali_path, test_path,
                   num_epochs, num_train_steps, num_vali_steps, num_test_steps, model_dir,
                   list_size, vali_list_size, test_list_size, seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    if not model_dir and ex.current_run._id:
        model_dir = '/tmp/model_{}'.format(ex.current_run._id)
    hparams = tf.contrib.training.HParams(learning_rate=learning_rate)
    ranker = get_estimator(hparams, model_dir=model_dir)
    train_args = dict(input_fn=lambda: input_fn(train_path),
                      max_steps=num_train_steps,
                      hooks=[hook_train] if hook_train else None)
    if vali_path is not None:
        train_spec = tf.estimator.TrainSpec(**train_args)
        vali_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(vali_path,
                                                list_size=vali_list_size if vali_list_size else list_size),
                                      hooks=[hook_vali] if hook_vali else None)
    for epoch in range(num_epochs):
        print('Epoch {} of {}'.format(epoch + 1, num_epochs))
        print('Training and Validating')
        if vali_path is not None:
            tf.estimator.train_and_evaluate(ranker, train_spec, vali_spec)
        else:
            ranker.train(**train_args)
    print('Testing')
    if test_path is not None:
        ranker.evaluate(input_fn=lambda: input_fn(test_path, num_epochs=1,
                                              list_size=test_list_size if test_list_size else list_size),
                    steps=num_test_steps,
                    hooks=[hook_test] if hook_test else None)

