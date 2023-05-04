#train
python -u re_agcn_main.py --do_train --do_eval --task_name semeval --data_dir ./data/semeval/ --model_path ../../data/bert_model_large_uncased/  --dep_type local_global_graph --model_name RE_AGCN.SEVEVAL.BERT.L --do_lower_case  --IB True --beta1 1e-5 --beta2 1e-10 --tag large_ib_beta1_1e-5_beta2_1e-10 



 nohup python -u re_agcn_main.py --do_train --do_eval --task_name semeval --data_dir ./data/semeval/ --model_path ../../data/bert_model_large_uncased/  --dep_type local_global_graph --model_name RE_AGCN.SEVEVAL.BERT.L --do_lower_case  --IB True --beta1 1e-7 --beta2 1e-10 --tag large_ib_beta1_1e-7_beta2_1e-10_cuda_3_secondary --num_train_epoch 1000 >  large_ib_beta1_1e-7_beta2_1e-10_cuda_3.log &


num_train_epochs
