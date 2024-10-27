#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python evaluate_mschema2qa_text2sql_ckpts.py --batch_size 8 \
--device 0 \
--seed 42 \
--save_path {{ Your path to trained checkpoints }} \
--model_name_or_path google/mt5-large \
--eval_results_path ./eval_results/mschema2qa_base_32 \
--mode eval \
--dev_filepath ./data/mschema2qa/test.json \
--dataset_lang en \
--cross_dataset_lang ja \
--num_beams 8 \
--num_return_sequences 1 \
--wandb_log \
--exp_name eval_mschema2qa_base_32


# --save_path: Your path to saved model checkpoints