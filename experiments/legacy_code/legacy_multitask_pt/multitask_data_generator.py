import json
import copy
import argparse
import random
import logging 
import os 

import numpy as np
from tqdm import tqdm 
from datasets import load_dataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


language_code_to_name = {
    "en": "English",
    "zh": "Chinese",
    "ru": "Russian",
    "vi": "Vietnamese",
}

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generating the ranked dataset.")
    parser.add_argument('--save_dir', type = str, default = "./data/multitask_ft",
                        help = 'Path to directory to save the generated dataset.')
    # Translation task
    parser.add_argument('--opus_translation_lang', type = str, default = "en-zh",
                        help = 'Language pair to use for translation task. (default: en-zh)')
    parser.add_argument('--translation_direction', type = str, choices=['source_to_target', 'target_to_source', 'both'], default = 'both',
                        help = 'translation direction of the translation task.')
   
    opt = parser.parse_args()
    return opt

def lista_contains_listb(lista, listb):
    for b in listb:
        if b not in lista:
            return 0
    
    return 1


def generate_translation_task_dataset(opt):

        
    opus_100 = load_dataset("opus100", opt.opus_translation_lang) # TODO: support more languages?
    source_lang = opt.opus_translation_lang.split("-")[0]
    target_lang = opt.opus_translation_lang.split("-")[1]

    opus_100_train = opus_100["train"]

    
    train_dataset = []
    for row in tqdm(opus_100_train, desc="Generating translation task dataset (train)"):
        if opt.translation_direction == "source_to_target" or opt.translation_direction == "both":
            task_prompt = f"Translate the following sentence into {language_code_to_name[target_lang]}: " 
            train_dataset.append(
                {
                    "input_sequence": task_prompt + row["translation"][source_lang],
                    "output_sequence": row["translation"][target_lang]
                }
            )
        if opt.translation_direction == "target_to_source" or opt.translation_direction == "both":
            task_prompt = f"Translate the following sentence into {language_code_to_name[source_lang]}: " 
            train_dataset.append(
                {
                    "input_sequence": task_prompt + row["translation"][target_lang],
                    "output_sequence": row["translation"][source_lang]
                }
            )
    
    dev_dataset = []
    opus_100_validation = opus_100["validation"]
    for row in tqdm(opus_100_validation, desc="Generating translation task dataset (val)"):
        if opt.translation_direction == "source_to_target" or opt.translation_direction == "both":
            task_prompt = f"Translate the following sentence into {language_code_to_name[target_lang]}: " 
            dev_dataset.append(
                {
                    "input_sequence": task_prompt + row["translation"][source_lang],
                    "output_sequence": row["translation"][target_lang]
                }
            )
        if opt.translation_direction == "target_to_source" or opt.translation_direction == "both":
            task_prompt = f"Translate the following sentence into {language_code_to_name[source_lang]}: " 
            dev_dataset.append(
                {
                    "input_sequence": task_prompt + row["translation"][target_lang],
                    "output_sequence": row["translation"][source_lang]
                }
            )
    # shuffle original dataset
    random.Random(42).shuffle(dev_dataset)
    logger.info("Translation task validation dataset size capped to 1K.")
    dev_dataset = dev_dataset[:1000]
    
    # Save dataset
    train_save_path = os.path.join(opt.save_dir, f"translation_train_{opt.opus_translation_lang}.json")
    with open(train_save_path, "w") as f:
        f.write(json.dumps(train_dataset, indent = 4, ensure_ascii = False))
        logger.info("Translation task dataset train split saved to {}".format(train_save_path))
        logger.info(f"Size: {len(train_dataset)}")

    dev_save_path = os.path.join(opt.save_dir, f"translation_val_{opt.opus_translation_lang}.json")
    with open(dev_save_path, "w") as f:
        f.write(json.dumps(dev_dataset, indent = 4, ensure_ascii = False))
        logger.info("Translation task dataset val split saved to {}".format(dev_save_path))
        logger.info(f"Size: {len(dev_dataset)}")

def generate_schema_and_value_prediction_dataset(opt):
    totto = load_dataset("totto")
    totto_train = totto["train"]
    totto_validation = totto["validation"]

    schema_task_prompt = "List schema items mentioned in the utterance: "
    value_task_prompt = "List values mentioned in the utterance: "
    for dataset, split in zip([totto_train, totto_validation], ["train", "val"]):
        sp_generated_dataset = []
        vp_generated_dataset = []
        for example in tqdm(dataset, desc=f"Generating sp/vp dataset ({split})"):
            table_name_original = example["table_page_title"] # We don't use section title here
            sentence = example["sentence_annotations"]["final_sentence"][0]
            column_names_original = []

            for cell in example["table"][0]:
                is_header = cell["is_header"]
                value = cell["value"]
                if is_header == True:
                    column_names_original.append(value)

            if len(column_names_original) != len(example["table"][0]):
                # ignore this example if the number of columns in the table header is not equal to the number of columns in the table body
                continue  
            
            schema_sequence = sentence +" | " + table_name_original + " : "
            column_info_list = []

            for column_name_original in column_names_original:
                column_info = table_name_original + "." + column_name_original
                column_info_list.append(column_info)

            schema_sequence += " , ".join(column_info_list)
            while "  " in schema_sequence:
                schema_sequence = schema_sequence.replace("  ", " ")
            

            # Find mentioned schema items and values
            mentioned_column_original_names = []
            mentioned_values = []

            highlighted_cell_indices = example["highlighted_cells"]

            # Since this is a single table example, we can assume that all highlighted cells are from the same table
            # Find the table name
            table_name = example["table_page_title"] # We don't use section title here

            # Find the highlighted column names
            is_valid_table = True  
            for cell_index in highlighted_cell_indices:
                mentioned_column_index = cell_index[1]
                if mentioned_column_index >= len(column_names_original):
                    is_valid_table = False
                    break
                if column_names_original[mentioned_column_index] in schema_sequence:
                    mentioned_column_original_names.append(column_names_original[mentioned_column_index])
            
            if not is_valid_table:
                continue # We ignore this example
            
            # Remove duplicate column names
            mentioned_column_original_names = list(set(mentioned_column_original_names))
            
            for cell_index in highlighted_cell_indices:
                highlighted_cell = example["table"][cell_index[0]][cell_index[1]]
                if highlighted_cell["value"] in sentence:
                    mentioned_values.append(highlighted_cell["value"])

            # Remove duplicate values
            mentioned_values = list(set(mentioned_values))
            
            # schema prediction task 
            o_t = table_name
            o_c = ""
            if len(mentioned_column_original_names) != 0:
                o_c = ", ".join(mentioned_column_original_names)
            else:   
                o_c = "<none>"

            output_sequence = "<table>" +  " " + o_t + " " + "<column>" + " " + o_c

            sp_generated_dataset.append(
                {
                    "input_sequence": schema_task_prompt + schema_sequence,
                    "output_sequence": output_sequence,
                    "mentioned_schema_items":{
                        "table_name": table_name,
                        "column_names": mentioned_column_original_names if len(mentioned_column_original_names) != 0 else ["<none>"]
                    }
                }
            )

            # value prediction task
            o_v = ""
            if len(mentioned_values) != 0:
                o_v = ", ".join(mentioned_values)
            else:
                o_v = "<none>"

            output_sequence = "<value>" + " " + o_v

            vp_generated_dataset.append(
                {
                    "input_sequence": value_task_prompt + schema_sequence,
                    "output_sequence": output_sequence,
                    "mentioned_values":{
                        "values": mentioned_values if len(mentioned_values) != 0 else ["<none>"]
                    }
                }
            )
        if split == "val":
            random.Random(42).shuffle(sp_generated_dataset)
            logger.info("Schema prediction dataset validation split size capped to 1K.")
            sp_generated_dataset = sp_generated_dataset[:1000]

            random.Random(42).shuffle(vp_generated_dataset)
            logger.info("Value prediction dataset validation split size capped to 1K.")
            vp_generated_dataset = vp_generated_dataset[:1000]

        # Save schema prediction dataset
        save_path = os.path.join(opt.save_dir, f"schema_prediction_{split}.json")
        with open(save_path, "w") as f:
            f.write(json.dumps(sp_generated_dataset, indent = 4, ensure_ascii = False))
            logger.info(f"Schema prediction dataset {split} split saved to {save_path}")
            logger.info(f"Size: {len(sp_generated_dataset)}")
        
        # Save value prediction dataset
        save_path = os.path.join(opt.save_dir, f"value_prediction_{split}.json")
        with open(save_path, "w") as f:
            f.write(json.dumps(vp_generated_dataset, indent = 4, ensure_ascii = False))
            logger.info(f"Value prediction dataset {split} split saved to {save_path}")
            logger.info(f"Size: {len(vp_generated_dataset)}")



def generate_seq2seq_dataset(opt):
    generate_translation_task_dataset(opt)
    generate_schema_and_value_prediction_dataset(opt)



if __name__ == "__main__":
    opt = parse_option()
    random.seed(42)
    
    generate_seq2seq_dataset(opt)
    # Save the dataset to the
