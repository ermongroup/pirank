
METRICS_DIR=./tmp/metrics_mslr_pirank_simple_loss python3 pirank_simple.py --submit=True --loss_fn=pirank_simple_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/Fold1/train.txt \
--vali_path=./data/Fold1/vali.txt \
--test_path=./data/Fold1/test.txt \
--num_features=136 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_mslr_pirank_simple_loss
        
METRICS_DIR=./tmp/metrics_mslr_neuralsort_permutation_loss python3 pirank_simple.py --submit=True --loss_fn=neuralsort_permutation_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/Fold1/train.txt \
--vali_path=./data/Fold1/vali.txt \
--test_path=./data/Fold1/test.txt \
--num_features=136 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_mslr_neuralsort_permutation_loss
        
METRICS_DIR=./tmp/metrics_mslr_pairwise_logistic_loss python3 pirank_simple.py --submit=True --loss_fn=pairwise_logistic_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/Fold1/train.txt \
--vali_path=./data/Fold1/vali.txt \
--test_path=./data/Fold1/test.txt \
--num_features=136 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_mslr_pairwise_logistic_loss
        
METRICS_DIR=./tmp/metrics_mslr_lambda_rank_loss python3 pirank_simple.py --submit=True --loss_fn=lambda_rank_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/Fold1/train.txt \
--vali_path=./data/Fold1/vali.txt \
--test_path=./data/Fold1/test.txt \
--num_features=136 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_mslr_lambda_rank_loss
        
METRICS_DIR=./tmp/metrics_mslr_softmax_loss python3 pirank_simple.py --submit=True --loss_fn=softmax_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/Fold1/train.txt \
--vali_path=./data/Fold1/vali.txt \
--test_path=./data/Fold1/test.txt \
--num_features=136 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_mslr_softmax_loss
        
METRICS_DIR=./tmp/metrics_mslr_approx_ndcg_loss python3 pirank_simple.py --submit=True --loss_fn=approx_ndcg_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/Fold1/train.txt \
--vali_path=./data/Fold1/vali.txt \
--test_path=./data/Fold1/test.txt \
--num_features=136 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_mslr_approx_ndcg_loss
        
METRICS_DIR=./tmp/metrics_yahoo_pirank_simple_loss python3 pirank_simple.py --submit=True --loss_fn=pirank_simple_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/yahoo/set1.train.txt \
--vali_path=./data/yahoo/set1.valid.txt \
--test_path=./data/yahoo/set1.test.txt \
--num_features=700 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_yahoo_pirank_simple_loss
        
METRICS_DIR=./tmp/metrics_yahoo_neuralsort_permutation_loss python3 pirank_simple.py --submit=True --loss_fn=neuralsort_permutation_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/yahoo/set1.train.txt \
--vali_path=./data/yahoo/set1.valid.txt \
--test_path=./data/yahoo/set1.test.txt \
--num_features=700 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_yahoo_neuralsort_permutation_loss
        
METRICS_DIR=./tmp/metrics_yahoo_pairwise_logistic_loss python3 pirank_simple.py --submit=True --loss_fn=pairwise_logistic_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/yahoo/set1.train.txt \
--vali_path=./data/yahoo/set1.valid.txt \
--test_path=./data/yahoo/set1.test.txt \
--num_features=700 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_yahoo_pairwise_logistic_loss
        
METRICS_DIR=./tmp/metrics_yahoo_lambda_rank_loss python3 pirank_simple.py --submit=True --loss_fn=lambda_rank_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/yahoo/set1.train.txt \
--vali_path=./data/yahoo/set1.valid.txt \
--test_path=./data/yahoo/set1.test.txt \
--num_features=700 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_yahoo_lambda_rank_loss
        
METRICS_DIR=./tmp/metrics_yahoo_softmax_loss python3 pirank_simple.py --submit=True --loss_fn=softmax_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/yahoo/set1.train.txt \
--vali_path=./data/yahoo/set1.valid.txt \
--test_path=./data/yahoo/set1.test.txt \
--num_features=700 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_yahoo_softmax_loss
        
METRICS_DIR=./tmp/metrics_yahoo_approx_ndcg_loss python3 pirank_simple.py --submit=True --loss_fn=approx_ndcg_loss \
--ndcg_k=10 \
--tau=5 \
--list_size=100 \
--hidden_layers=256,relu,256,relu,128,relu,128,relu,64,relu,64,relu \
--train_path=./data/yahoo/set1.train.txt \
--vali_path=./data/yahoo/set1.valid.txt \
--test_path=./data/yahoo/set1.test.txt \
--num_features=700 \
--optimizer=Adam \
--learning_rate=0.00001 \
--num_epochs=100 \
--batch_size=16 \
--model_dir=./tmp/model_yahoo_approx_ndcg_loss
        