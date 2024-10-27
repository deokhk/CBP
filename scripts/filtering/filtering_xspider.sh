#!/bin/bash

# MODEL_PATH -> best checkpoint of trained vanilla semantic parser for filtering
# DATA_PATH -> path to the synthesized utterances

LANGS=("zh" "vi")

for lang in ${LANGS[@]}; do
    DATA_PATH=./output/xspider-mr2text-enko-50.16.3e-5.500/last_checkpoint/generated_predictions_${lang}_beam_4_from_train_spider_seq2seq_english.json
    ORIGINAL_DATA_PATH=./data/xspider/train_spider.json
    MODEL_PATH=./models/text2sql-mt5-large_baseline/best_model
    CUDA_VISIBLE_DEVICES=0 python consistency_filter.py \
        --question_synthetic_data_path $DATA_PATH \
        --sql_synthetic_data_path $ORIGINAL_DATA_PATH \
        --batch_size 8 \
        --num_beams 8 \
        --num_return_sequences 1 \
        --device "0" \
        --model_name_or_path $MODEL_PATH \
        --db_path "./database"
done