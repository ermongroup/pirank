METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.001 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 10 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.001 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1000.0 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.001 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1000.0 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 1e-05 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 10 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 1e-05 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_10 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1000.0 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 1e-05 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1000.0 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 10 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1000.0 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.001 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1000.0 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 1e-05 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 10 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 1e-05 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_10 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1000.0 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 1e-05 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1000.0 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.001 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 10 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.001 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_10 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1000.0 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 0.001 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.001__opt_Adagrad__tau_1000.0 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 1e-05 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_10 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 10 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 1e-05 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_10 \
--submit True

METRICS_DIR=./experiments/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1000.0 python3 pirank_simple_rebased.py \
--loss_fn pirank_simple_loss \
--ndcg_k 10 \
--tau 1000.0 \
--list_size 100 \
--hidden_layer_dims 1024,512,256 \
--train_path ./data/Fold1/train.txt \
--vali_path  ./data/Fold1/vali.txt \
--test_path  ./data/Fold1/test.txt \
--num_features 136 \
--optimizer Adagrad \
--learning_rate 1e-05 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_1e-05__opt_Adagrad__tau_1000.0 \
--submit True

