#!/bin/bash

echo Evaluating Mspider
TARGET_LANG=zh
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mspider_proposed/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train_spider_seq2seq_english.json
TARGET_LANG=vi

python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mspider_proposed/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train_spider_seq2seq_english.json

echo Evaluating Mschema2QA
TARGET_LANG=zh
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=ar
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=de
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=es
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=fa
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=fi
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=it
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=ja
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=pl
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=tr
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /mnt/hdd4/deokhk/Synthesized_Question_ACL2024_final/Mschema2QA/language_detected_filtered_generated_predictions_${TARGET_LANG}_beam_4_from_train.json
