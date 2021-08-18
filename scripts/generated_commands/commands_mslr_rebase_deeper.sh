!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01 python3 pirank_simple_rebased.py \
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
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1 python3 pirank_simple_rebased.py \
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
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.1 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.1__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01 python3 pirank_simple_rebased.py \
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
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1 python3 pirank_simple_rebased.py \
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
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.3 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.3__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01 python3 pirank_simple_rebased.py \
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
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.01)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1 python3 pirank_simple_rebased.py \
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
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_0.1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.1__opt_Adagrad__tau_1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.01)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_0.1)


!(export PYTHONPATH=/content:$PYTHONPATH ; cd pirank ; METRICS_DIR=./experiments/mslr_rebase_deeper/metrics__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1 python3 pirank_simple_rebased.py \
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
--learning_rate 0.01 \
--dropout_rate 0.6 \
--num_train_steps 50000 \
--train_batch_size 16 \
--output_dir ./experiments/mslr_rebase_deeper/model__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1 \
2>&1 | tee ./experiments/mslr_rebase_deeper/log__arch_tfr__batch_16__data_mslr__dropout_0.6__loss_pirank_simple__lr_0.01__opt_Adagrad__tau_1)


