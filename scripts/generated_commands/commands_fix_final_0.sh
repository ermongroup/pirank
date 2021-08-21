# Experiment fix_final, iteration 0
# {"num_train_steps": 100000, "data": ["mslr", "yahoo"], "topks": [3, 5, 10], "list_sizes": [100, 200], "batch_sizes": [16], "learning_rates": [0.001], "optimizers": ["Adam"], "dropout_rates": [0.3], "temperatures": [1000.0], "architectures": {"tfr": "1024,512,256"}, "losses": ["pirank_simple", "pirank_arp_simple", "neuralsort_permutation", "pairwise_logistic", "lambda_rank", "softmax", "approx_ndcg"]}


METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 3 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pirank_arp_simple_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pairwise_logistic_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 3 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 5 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 10 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn softmax_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn approx_ndcg_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 3 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pirank_arp_simple_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pairwise_logistic_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 3 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 5 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 10 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn softmax_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn approx_ndcg_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_100__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 3 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pirank_arp_simple_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pairwise_logistic_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 3 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 5 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 10 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn softmax_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn approx_ndcg_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 3 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 5 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adam__tau_1000.0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pirank_arp_simple_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pirank_arp_simple__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn neuralsort_permutation_loss \
--ndcg_k 0 \
--tau 1000.0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_neuralsort_permutation__lr_0.001__opt_Adam__tau_1000.0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn pairwise_logistic_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_pairwise_logistic__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 3 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_3 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 5 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_5 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 python3 pirank_simple_rebased.py \
--loss_fn lambda_rank_loss \
--ndcg_k 10 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_lambda_rank__lr_0.001__opt_Adam__tau_0__topk_10 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn softmax_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_softmax__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

METRICS_DIR=./experiments/fix_final/metrics__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 python3 pirank_simple_rebased.py \
--loss_fn approx_ndcg_loss \
--ndcg_k 0 \
--tau 0 \
--list_size 200 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/yahoo/set1.train.txt \
--vali_path  ./data/yahoo/set1.valid.txt \
--test_path  ./data/yahoo/set1.test.txt \
--num_features 700 \
--optimizer Adam \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 100000 \
--train_batch_size 16 \
--output_dir ./experiments/fix_final/model__L_200__arch_tfr__batch_16__data_yahoo__dropout_0.3__loss_approx_ndcg__lr_0.001__opt_Adam__tau_0__topk_0 \
--submit True

