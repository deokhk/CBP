#!/bin/bash

LANG=zh
DATA_PATH=/home/deokhk/research/ZX-seq2seq/data/spider/train_spider_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/ZX-seq2seq/data/spider/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type xspider --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=vi
DATA_PATH=/home/deokhk/research/ZX-seq2seq/data/spider/train_spider_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/ZX-seq2seq/data/spider/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type xspider --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}
