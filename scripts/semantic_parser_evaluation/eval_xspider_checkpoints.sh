#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python evaluate_text2sql_ckpts.py --batch_size 8 \
--device 0 \
--seed 52 \
--save_path {{ Your path to trained checkpoints }} \
--model_name_or_path google/mt5-large \
--eval_results_path ./eval_results/xspider_vanilla \
--mode eval \
--dev_filepath ./data/preprocessed_data/dev_spider_seq2seq.json \
--original_dev_filepath ./data/spider/dev.json \
--cross_dev_filepath ./data/dev_cspider_seq2seq.json \
--cross_original_dev_filepath ./data/Cspider/dev.json \
--cross_eval_dataset_name cspider \
--db_path ./database \
--cross_db_path ./database \
--num_beams 8 \
--num_return_sequences 8 \
--wandb_log \
--exp_name eval_mt5-large-text2sql-vanilla

# --save_path: Your path to saved model checkpoints