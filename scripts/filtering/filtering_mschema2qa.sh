#!/bin/bash

# MODEL_PATH -> best checkpoint of trained vanilla semantic parser for filtering (mschema2qa)
# DATA_PATH -> path to the synthesized utterances

langs=("zh" "tr" "pl" "it" "fi" "fa" "ar" "ja" "es" "de")

for lang in ${LANGS[@]}; do
    DATA_PATH=./output/mschema2qa-mr2text-1K-50.16.3e-5.500/last_checkpoint/generated_predictions_${lang}_beam_4_from_train_spider_seq2seq_english.json
    MODEL_PATH=./models/mt5-mschema2qa-seed32/best_model
    CUDA_VISIBLE_DEVICES=0 python consistency_filter.py \
        --question_synthetic_data_path $DATA_PATH \
        --num_beams 8 \
        --batch_size 8 \
        --device "0" \
        --model_name_or_path $MODEL_PATH \
        --num_return_sequences 1 \
        --dataset_type mschema2qa
done