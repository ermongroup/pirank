# PiRank: Learning to Rank via Differentiable Sorting

This repository provides a reference implementation for learning PiRank-based models as described in the paper:

> PiRank: Learning to Rank via Differentiable Sorting   
> [Robin Swezey](https://www.linkedin.com/in/robin-m-e-swezey/), [Aditya Grover](https://aditya-grover.github.io), [Bruno Charron](https://scholar.google.com/citations?user=klbU-o4AAAAJ) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).  
> Paper: https://arxiv.org/abs/2012.06731

## Requirements

The codebase is implemented in Python 3.7. To install the necessary base requirements, run the following commands:

```
pip install -r requirements.txt
```

If you intend to use a GPU, modify _requirements.txt_ to install _tensorflow-gpu_ instead of _tensorflow_.

You will also need the NeuralSort implementation available [here](https://github.com/ermongroup/neuralsort). Make sure it is added to your PYTHONPATH.

## Datasets

PiRank was tested on the two following datasets:

- The MSLR WEB30K data can be found at [this address](https://www.microsoft.com/en-us/research/project/mslr/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fprojects%2Fmslr%2Fdownload.aspx).
- The Yahoo! C14 dataset can be found at [this address](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&guccounter=1). 

Additionally, the code is expected to work with any dataset stored in the standard LibSVM format used for LTR experiments.

## Scripts

There are two scripts for the code:

- _pirank_simple.py_ implements a simple depth-1 PiRank loss (_d=1_). It is used in the experiments of sections 4.1 (benchmark evaluation on MSLR-WEB30K and Yahoo! C14 datasets), 4.2.1 (effect of temperature parameter), and 4.2.2 (effect of training list size).

- _pirank_deep.py_ implements the deeper PiRank losses (_d>=1_). It is used for the experiments of section 4.2.3 and comes with a convenient synthetic data generator as well as more tuning options.

## Options

Options are handled by Sacred (see _Examples_ section below).

### _pirank_simple.py_ and _pirank_deep.py_

**PiRank-related:**

| Parameter | Default Value | Description |
|----------|:-------------:|------:|
| loss_fn | pirank_simple_loss | The loss function to use (either a TFR RankingLossKey, or loss function from the script) |
| ste | False | Whether to use the Straight-Through Estimator |
| ndcg_k | 15 | NDCG@k cutoff when using NS-NDCG loss |

**NeuralSort-related:**

| Parameter | Default Value | Description |
|----------|:-------------:|------:|
| tau | 5 | Temperature |
| taustar | 1e-10 | Temperature for trues and straight-through estimation.

**TensorFlow-Ranking and architecture-related:**

| Parameter | Default Value | Description |
|----------|:-------------:|------:|
| hidden_layers | "256,tanh,128,tanh,64,tanh" | Hidden layers for an example-wise feedforward network in the format _size,activation,...,size,activation_ |
| num_features | 136 | Number of features per document |
| list_size | 100 | List size used for training |
| group_size | 1 | Group size used in score function |

**Training-related:**

| Parameter | Default Value | Description |
|----------|:-------------:|------:|
| train_path | "/data/MSLR-WEB30K/Fold*/train.txt" | Input file path used for training |
| vali_path | "/data/MSLR-WEB30K/Fold*/vali.txt" | Input file path used for validation |
| test_path | "/data/MSLR-WEB30K/Fold*/test.txt" | Input file path used for testing |
| model_dir | None | Output directory for models |
| num_epochs | 200 | Number of epochs to train, set 0 to just test |
| lr | 1e-4 | initial learning rate |
| batch_size | 32 | The batch size for training |
| num_train_steps | None | Number of steps for training |
| num_vali_steps | None | Number of steps for validation |
| num_test_steps | None | Number of steps for testing |
| learning_rate | 0.01 | Learning rate for optimizer |
| dropout_rate | 0.5 | The dropout rate before output layer |
| optimizer | Adagrad | The optimizer for gradient descent |

**Sacred:**

In addition, you can use regular parameters from Sacred (such as -m for logging the experiment to MongoDB).

### _pirank_deep.py_ only

| Parameter | Default Value | Description |
|----------|:-------------:|------:|
| merge_block_size | None | Block size used if merging, None if not merging |
| top_k | None | Use a different Top-k for merging than final NDCG@k for loss |
| straight_backprop | False | Backpropagate on scores only through NS operator |
| full_loss | False | Use the complete loss at the end of merge |
| tau_scheme | None | Which scheme to use for temperature going deeper (default: constant) |
| data_generator | None | Data generator (default: TFR\s libsvm); use this for synthetic generation |
| num_queries | 30000 | Number of queries for synthetic data generator |
| num_query_features | 10 | Number of columns used as factors for each query by synthetic data generator |
| actual_list_size | None | Size of actual list per query in synthetic data generation |
| train_path | "/data/MSLR-WEB30K/Fold*/train.txt" | Input file path used for training; alternatively value of seed if using data generator |
| vali_path | "/data/MSLR-WEB30K/Fold*/vali.txt" | Input file path used for validation; alternatively value of seed if using data generator |
| test_path | "/data/MSLR-WEB30K/Fold*/test.txt" | Input file path used for testing; alternatively value of seed if using data generator |
| with_opa | True | Include pairwise metric OPA |

## Examples

_Run the benchmark experiment of section 4.1 with PiRank simple loss on MSLR-WEB30K_

```
cd pirank
python3 pirank_simple.py with loss_fn=pirank_simple_loss \
    ndcg_k=10 \
    tau=5 \
    list_size=80 \
    hidden_layers=256,relu,256,relu,128,relu,64,relu \
    train_path=/data/MSLR-WEB30K/Fold1/train.txt \
    vali_path=/data/MSLR-WEB30K/Fold1/vali.txt \
    test_path=/data/MSLR-WEB30K/Fold1/test.txt \
    optimizer=Adam \
    learning_rate=0.00001 \
    num_epochs=100 \
    batch_size=16 \
    model_dir=/tmp/model
```

_Run the benchmark experiment of section 4.1 with classic LambdaRank on MSLR-WEB30K_

```
cd pirank
python3 pirank_simple.py with loss_fn=lambda_rank_loss \
    ndcg_k=10 \
    tau=5 \
    list_size=80 \
    hidden_layers=256,relu,256,relu,128,relu,64,relu \
    train_path=/data/MSLR-WEB30K/Fold1/train.txt \
    vali_path=/data/MSLR-WEB30K/Fold1/vali.txt \
    test_path=/data/MSLR-WEB30K/Fold1/test.txt \
    optimizer=Adam \
    learning_rate=0.00001 \
    num_epochs=100 \
    batch_size=16 \
    model_dir=/tmp/model
```

_Run the scaling ablation experiment of section 4.2.3 using synthetic data generation (d=2)_

```
cd pirank
python3 pirank_deep.py with loss_fn=pirank_deep_loss \
    ndcg_k=10 \
    ste=True \
    merge_block_size=100 \
    tau=5 \
    taustar=1e-10 \
    tau_scheme=square \
    data_generator=synthetic_data_generator \
    actual_list_size=1000 \
    list_size=1000 \
    vali_list_size=1000 \
    test_list_size=1000 \
    full_loss=False \
    train_path=0 \
    vali_path=1 \
    test_path=2 \
    num_queries=1000 \
    num_features=25 \
    num_query_features=5 \
    hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
    optimizer=Adam \
    learning_rate=0.00001 \
    num_epochs=100 \
    batch_size=16
```

## Help

If you need help, reach out to [Robin Swezey](http://swezey.fr) or raise an issue.

## Citing

If you find PiRank useful in your research, please consider citing the following paper:

> @inproceedings{   
> swezey2020pirank,   
> title={PiRank: Learning to Rank via Differentiable Sorting},  
> author={Robin Swezey and Aditya Grover and Bruno Charron and Stefano Ermon},  
> year={2020},  
> url={},  
> }
