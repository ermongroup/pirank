METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_5 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_10 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_20 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 20 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_20 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_5 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_10 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_20 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 20 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_20 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_5 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_10 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_20 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 20 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_20 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_lambda_rank__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_lambda_rank__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.01__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.01__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.1__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.1__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_1__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_1__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pairwise_logistic__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pairwise_logistic_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pairwise_logistic__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_approx_ndcg__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn approx_ndcg_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_approx_ndcg__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_softmax__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn softmax_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_softmax__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_5 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_10 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_20 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 20 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01__topk_20 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_5 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_10 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_20 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 20 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1__topk_20 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_5 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_10 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_20 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 20 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1__topk_20 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_lambda_rank__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_lambda_rank__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.01__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 0.01 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.01__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.1__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 0.1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_0.1__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_1__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 1 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_neuralsort_permutation__lr_0.1__opt_Adagrad__tau_1__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pairwise_logistic__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pairwise_logistic_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_pairwise_logistic__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_approx_ndcg__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn approx_ndcg_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_approx_ndcg__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/final/metrics__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_softmax__lr_0.1__opt_Adagrad__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn softmax_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adagrad \
--learning_rate 0.1 \
--dropout_rate 0.6 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/final/model__arch_tfr__batch_16__data_yahoo__dropout_0.6__loss_softmax__lr_0.1__opt_Adagrad__tau_0__topk_0 \
--submit True

