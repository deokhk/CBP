#!/bin/bash


MODEL_PATH=./models/mt5-large-mschema2qa-CPT-seed32/best_model/ # Your path to the best model checkpoint
langs=("zh" "tr" "pl" "it" "fi" "fa" "ar" "ja" "es" "de")
for lang in ${langs[@]}; do
    CUDA_VISIBLE_DEVICES=0 python -u evaluate_single_ckpt.py \
        --batch_size 8 \
        --device "0" \
        --seed 42 \
        --save_path $MODEL_PATH \
        --model_name_or_path "google/mt5-large" \
        --eval_results_path $EVAL_RES_PATH \
        --eval_file_name "eval_${lang}.txt" \
        --mode eval \
        --dev_filepath "./data/mschema2qa/test.json" \
        --dataset_lang ${lang} \
        --dataset_type "mschema2qa" \
        --num_beams 8 \
        --num_return_sequences 1
done