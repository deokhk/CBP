#!/bin/bash

# This will evaluate the model on Cspider and Vspider split in Xspider dataset.

MODEL_PATH=./models/mt5-large-xspider/best_model # Your path to best model checkpoint (after evaluation)
EVAL_RES_PATH=./eval_results/mspider_mt5-large-xspider # Path to save evaluation results

# Evaluate on "zh-full" split 
CUDA_VISIBLE_DEVICES=0 python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 52 \
    --save_path $MODEL_PATH \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path $EVAL_RES_PATH \
    --eval_file_name "eval_zh_full.txt" \
    --mode eval \
    --dev_filepath "./data/xspider/cspider/dev_cspider_seq2seq.json" \
    --original_dev_filepath "./data/xspider/cspider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

# Evaluate on "zh" split 
CUDA_VISIBLE_DEVICES=0 python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 52 \
    --save_path $MODEL_PATH \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path $EVAL_RES_PATH \
    --eval_file_name "eval_zh.txt" \
    --mode eval \
    --dev_filepath "./data/xspider/cspider/dev_cspider_zh_seq2seq.json" \
    --original_dev_filepath "./data/xspider/cspider/zh_dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

# Evaluate on "vi" split
CUDA_VISIBLE_DEVICES=0 python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 52 \
    --save_path $MODEL_PATH \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path $EVAL_RES_PATH \
    --eval_file_name "eval_vi.txt" \
    --mode eval \
    --dev_filepath "./data/xspider/vspider/dev_vspider_converted_seq2seq.json" \
    --original_dev_filepath "./data/xspider/vspider/dev_converted.json" \
    --db_path "./vspider_database" \
    --num_beams 8 \
    --num_return_sequences 8