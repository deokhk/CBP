#!/bin/bash


echo Evaluating Mschema2QA
TARGET_LANG=zh
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=ar
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=de
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=es
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=fa
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=fi
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=it
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=ja
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=pl
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json

TARGET_LANG=tr
python GEMBA_QE.py --target_lang $TARGET_LANG --data_path /home/deokhk/research/XLang-NL2SQL/output/test_genq_quality/mschema2qa/no_filtering/language_detected_generated_predictions_${TARGET_LANG}_beam_4_from_train.json
