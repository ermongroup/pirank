import tensorflow as tf
import tensorflow_ranking as tfr 
from neuralsort.tf import util
import submitit
import os
import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)

hook_train = None
hook_vali = None
hook_test = None
# Uncomment these if you like to use TensorBoard debugging
# hook = tf_debug.TensorBoardDebugHook("localhost:7001")
# hook_train = hook_test = tf_debug.TensorBoardDebugHook("localhost:7000")
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


# NeuralSort-related
flags = tf.app.flags
flags.DEFINE_float('tau', 5, 'temperature (dependent meaning)')
flags.DEFINE_float('taustar', 1e-10, 'Temperature to use for trues (hard or soft sort)')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to train, set 0 to just test')
flags.DEFINE_float('lr', 1e-4, 'initial learning rate')
# Training-related
flags.DEFINE_string("train_path", './data/Fold1/train.txt',
                    "Input file path used for training.")
flags.DEFINE_string("vali_path", './data/Fold1/vali.txt',
                    "Input file path used for validation.")
flags.DEFINE_string("test_path", './data/Fold1/test.txt',
                    "Input file path used for testing.")
flags.DEFINE_string("model_dir", "/tmp",
                    "Output directory for models.")
flags.DEFINE_integer("batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", None, "Number of steps for training.")
flags.DEFINE_integer("num_vali_steps", None, "Number of steps for validation.")
flags.DEFINE_integer("num_test_steps", None, "Number of steps for testing.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_string("optimizer", "Adagrad", "The optimizer for gradient descent.")
# TFR and architecture-related
flags.DEFINE_string("hidden_layers", "256,tanh,128,tanh,64,tanh",
                    "Hidden layers in the format 256,tanh,128,tanh,64,tanh")
flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")
## PiRank-related
flags.DEFINE_string("loss_fn", "pirank_simple_loss",
                    "The loss function to use (either a TFR RankingLossKey, or loss function from the script).")
flags.DEFINE_boolean('ste', True, 'Whether to use the Straight-Through Estimator')
flags.DEFINE_integer('ndcg_k', 15, 'NDCG@k cutoff when using NS-NDCG loss')
# Sacred-related
flags.DEFINE_boolean('d', False, 'Debugging for Sacred')
flags.DEFINE_boolean('c', False, 'Comment for Sacred')
flags.DEFINE_boolean('n', False, 'Name for Sacred')
flags.DEFINE_boolean('m', False, 'MongoDB for Sacred')
flags.DEFINE_boolean('s', False, 'SQL for Sacred')
flags.DEFINE_boolean('submit', False, 'use cluster')
flags.DEFINE_string('exp', 'dummy', 'experiment name')
# NS
FLAGS = flags.FLAGS
if FLAGS.submit:
    FLAGS.exp = FLAGS.model_dir.split('/')[-1]
    print(FLAGS.model_dir.split('/'))
flag_dict = {}
for attr,flag_obj in tf.flags.FLAGS.__flags.items():
    flag_dict[attr] = getattr(FLAGS, attr)
print(flag_dict["loss_fn"])
# exit()
# # Put all TF flags in Sacred configuration. Equiv to:
# # ex.add_config({param: getattr(FLAGS, param) for param in dir(FLAGS)})
# @ex.config
# def sacred_config():
#     for param in dir(FLAGS):
#         if param != 'learning_rate':
#             locals()[param] = getattr(FLAGS, param)


# TFR
# TODO Maybe switch with _train_input_fn (bookmark T)
# @ex.capture
def input_fn(path):
    '''
    Takes a libSVM LETOR dataset and turns it into tensor format.
    # It is a list of dictionaries, one per query-document pair, where
    # each dictionary is a mapping from a feature ID to a feature value.
    '''
    num_features = flag_dict["num_features"]
    list_size = flag_dict["list_size"]
    batch_size = flag_dict["batch_size"]
    num_epochs = flag_dict["num_epochs"]

    train_dataset = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(path, num_features, list_size),
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

    # train_dataset = train_dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
    # train_dataset = train_dataset.repeat(num_epochs).batch(batch_size)
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    # cf. mnist_input.get_iterator()
    return train_dataset.make_one_shot_iterator().get_next()


# Here we formulate a scoring function using a feed forward network.
# The function takes the features of a single example (i.e., query-document pair)
# and produces a relevance score.
# @ex.capture
def example_feature_columns():
    num_features = flag_dict["num_features"]
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
    a = a.split(',')
    b = [int(a[i]) for i in range(0,len(a),2)]
    c = [a[i] for i in range(1,len(a),2)]
    assert len(b) == len(c)
    return list(zip(b, c))


# Build the scoring NN for LETOR data
# @ex.capture
def make_score_fn():
    hidden_layers = flag_dict["hidden_layers"]
    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a documents."""
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        # TODO Add activations = "tanh" or ["tanh", "tanh", "tanh"]
        for layer in parse_arch(hidden_layers):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer[0],
                activation=layer[1])

        logits = tf.layers.dense(cur_layer, units=1)
        return logits

    return _score_fn


# @ex.capture
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
def pirank_simple_loss(labels, logits, features):
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


# TFR setup
def eval_metric_fns():
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
        for metric in [tfr.metrics.RankingMetricKey.MRR,
                       tfr.metrics.RankingMetricKey.ARP,
                       tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY ]
    })

    return metric_fns


# @ex.capture
def get_estimator(hparams, model_dir):
    """Create a ranking estimator.

    Args:
      hparams: (tf.contrib.training.HParams) a hyperparameters object.

    Returns:
      tf.learn `Estimator`.
    """
    optimizer = flag_dict["optimizer"]
    loss_fn = flag_dict["loss_fn"]
    ndcg_k = flag_dict["ndcg_k"]

    if loss_fn == 'pirank_simple_loss':
        loss_function = pirank_simple_loss
    elif loss_fn == 'neuralsort_permutation_loss':
        loss_function = neuralsort_permutation_loss
    elif loss_fn == 'lambda_rank_loss':
        loss_function = tfr.losses.make_loss_fn('pairwise_logistic_loss',
                                   lambda_weight=tfr.losses.create_ndcg_lambda_weight(topn=ndcg_k))
    else:
        loss_function = tfr.losses.make_loss_fn(loss_fn)

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
        model_dir=model_dir)

class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        pass

    def __call__(self):

        learning_rate = flag_dict["learning_rate"]
        train_path = flag_dict["train_path"]
        vali_path = flag_dict["vali_path"]
        test_path = flag_dict["test_path"]
        num_epochs = flag_dict["num_epochs"]
        num_train_steps = flag_dict["num_train_steps"]
        num_test_steps = flag_dict["num_test_steps"]
        model_dir = flag_dict["model_dir"]

        import os
        os.environ['IS_TEST'] = ''
        # if not model_dir and ex.current_run._id:
        #     model_dir = '/tmp/model_{}'.format(ex.current_run._id)
        #     ex.current_run.config['model_dir'] = model_dir
        hparams = tf.contrib.training.HParams(learning_rate=learning_rate)
        ranker = get_estimator(hparams, model_dir=model_dir)
        #TODO Put back tensorboard stuff
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_path),
                                            max_steps=num_train_steps,
                                            hooks=[hook_train] if hook_train else None)
        vali_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(vali_path),
                                            hooks=[hook_vali] if hook_vali else None)
        for epoch in range(num_epochs):
            print('Epoch {} of {}'.format(epoch + 1, num_epochs))
            print('Training and Validating')
            tf.estimator.train_and_evaluate(ranker, train_spec, vali_spec)
        print('Testing')
        os.environ['IS_TEST'] = '1'
        ranker.evaluate(input_fn=lambda: input_fn(test_path),
                        steps=num_test_steps,
                        hooks=[hook_test] if hook_test else None)

# Automain is captured by Sacred
# @ex.automain
# @LogFileWriter(ex)
def main():
    if flag_dict["submit"]:
    # if FLAGS.submit:
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
    else:
        Runner()()

if __name__ == '__main__':

    main()