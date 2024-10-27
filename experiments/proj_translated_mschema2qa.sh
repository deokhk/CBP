#!/bin/bash

LANG=ar
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=de
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=es
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=fa
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=fi
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=it
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=ja
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=pl
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=tr
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}

LANG=zh
DATA_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_${LANG}.json
SAVE_PATH=/home/deokhk/research/XSemPLR/dataset/mschema2qa/label_projected_reverse_${LANG}.json
CUDA_VISIBLE_DEVICES=1 python MT_align_values.py --data_type mschema2qa --translated_data_path ${DATA_PATH} --output_path ${SAVE_PATH}
