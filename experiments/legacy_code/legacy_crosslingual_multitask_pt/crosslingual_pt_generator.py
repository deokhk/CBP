import json
import copy
import argparse
import random
import logging 
import os 

import numpy as np
from tqdm import tqdm 
from datasets import load_dataset
from sql_metadata import Parser

from preprocessing import get_db_schemas, normalization, get_db_contents
from text2sql_data_generator import prepare_input_and_output

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


sql_keywords = ['select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', \
    'except', 'join', 'on', 'as', 'not', 'between', 'in', 'like', 'is', 'exists', 'max', 'min', \
        'count', 'sum', 'avg', 'and', 'or', 'desc', 'asc', 'order by', 'group by', 'distinct']
ops = ["=", "!=", ">", ">=", "<", "<="]


language_code_to_name = {
    "en": "English",
    "zh": "Chinese",
    "ru": "Russian",
    "vi": "Vietnamese",
}

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generating the ranked dataset.")

    # Generated (question, sql) path 
    parser.add_argument("--generated_text2sql_data_path", type = str, default = "./data/text2sql_data/train.json")

    # Other paths 
    parser.add_argument('--table_path', type = str, default = "./data/spider/tables.json")
    parser.add_argument('--save_dir', type = str, default = "./data/multitask_ft",
                        help = 'Path to directory to save the generated dataset.')

    parser.add_argument('--db_path', type = str, default = "./data/spider/database", 
                        help = "the filepath of database.")


    # Pretraining data generating options 

    parser.add_argument('--use_contents', action = 'store_true', default=True,
                        help = 'whether to add database contents in the input sequence.')
    parser.add_argument('--add_fk_info', action = 'store_true', default=True,
                    help = 'whether to add foreign key in the input sequence.')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--stepgen", action = "store_true",
                help = "Apply step-by-step generation for sql.")

    opt = parser.parse_args()
    return opt

def lista_contains_listb(lista, listb):
    for b in listb:
        if b not in lista:
            return 0
    
    return 1


def main(opt):

    # We need to generate each, repsectively. 
    with open(opt.generated_text2sql_data_path, "r") as f:
        generated_dataset = json.load(f)

    schema_task_prompt = "List schema items mentioned in the utterance: "
    value_task_prompt = "List values mentioned in the utterance: "
    
    # Load db schemas 
    all_db_infos = json.load(open(opt.table_path))
    db_schemas = get_db_schemas(all_db_infos)


    
    sp_prediction_dataset = []
    vp_prediction_dataset = []
    sql_generation_dataset = [ ]

    for data in tqdm(generated_dataset, desc=f"Generating cross-lingual datasets.."):
        question = data["question"] # We don't use section title here

        # We need to retrieve schema item..
        db_id = data["db_id"]
        
        sql = data["query"].strip()
        norm_sql = normalization(sql).strip()
        sql_tokens = norm_sql.split()

        preprocessed_data = {}
        preprocessed_data["question"] = question
        preprocessed_data["db_id"] = db_id

        preprocessed_data["sql"] = sql
        preprocessed_data["norm_sql"] = norm_sql
                
        preprocessed_data["db_schema"] = []
        preprocessed_data["pk"] = db_schemas[db_id]["pk"]
        preprocessed_data["fk"] = db_schemas[db_id]["fk"]
        preprocessed_data["table_labels"] = []
        preprocessed_data["column_labels"] = []
        

        # add database information (including table name, column name, ..., table_labels, and column labels)
        for table in db_schemas[db_id]["schema_items"]:
            db_contents = get_db_contents(
                question, 
                table["table_name_original"], 
                table["column_names_original"], 
                db_id, 
                opt.db_path
            )

            preprocessed_data["db_schema"].append({
                "table_name_original":table["table_name_original"],
                "table_name":table["table_name"],
                "column_names":table["column_names"],
                "column_names_original":table["column_names_original"],
                "column_types":table["column_types"],
                "db_contents": db_contents
            })

            # extract table and column classification labels
            if table["table_name_original"] in sql_tokens:  # for used tables
                preprocessed_data["table_labels"].append(1)
                column_labels = []
                for column_name_original in table["column_names_original"]:
                    if column_name_original in sql_tokens or \
                        table["table_name_original"]+"."+column_name_original in sql_tokens: # for used columns
                        column_labels.append(1)
                    else:
                        column_labels.append(0)
                preprocessed_data["column_labels"].append(column_labels)
            else:  # for unused tables and their columns
                preprocessed_data["table_labels"].append(0)
                preprocessed_data["column_labels"].append([0 for _ in range(len(table["column_names_original"]))])
        
        # Find schema items mentioned in the question 
        

        # Find values mentioned in the sql
        table_names_original, table_dot_column_names_original, column_names_original = [], [], []
        for table in db_schemas[db_id]["schema_items"]:
            table_name_original = table["table_name_original"]
            table_names_original.append(table_name_original)

            for column_name_original in ["*"]+table["column_names_original"]:
                table_dot_column_names_original.append(table_name_original+"."+column_name_original)
                column_names_original.append(column_name_original)
        
        not_value= table_names_original + column_names_original + table_dot_column_names_original + sql_keywords + ops + ["(", ")", ",", ";"]
        not_value_lowered = [x.lower() for x in not_value]
        parsed_sql = Parser(sql)
        sql_values = []
        for token in parsed_sql.tokens:
            if str(token).lower() not in not_value_lowered:
                sql_values.append(str(token).replace("'", "").replace("%","").strip())
        # Further post-processing
        sql_values_filtered = []

        # Here, we discard the values followed by limit
        # as this value is not explicitly mentioned in the question
        sql_limit_value = None 
        if parsed_sql.limit_and_offset is not None:
            sql_limit_value = str(parsed_sql.limit_and_offset[0])
        for v in sql_values:
            if (v.isnumeric() or (v.startswith("-") and v[1:].isnumeric())):
                if v != sql_limit_value:
                    sql_values_filtered.append(v)
            else:
                # if not numeric value, than the string must be included in the question
                if v in question:
                    sql_values_filtered.append(v)

        preprocessed_data["sql_values"] = sql_values_filtered
        
        table_ids = [idx for idx, _ in enumerate(preprocessed_data["db_schema"])]
                    
        # First, sql generation task 
        input_sequence, output_sequence = prepare_input_and_output(opt, preprocessed_data)
        
        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in preprocessed_data["db_schema"]:
            for column_name_original in ["*"] + table["column_names_original"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)
        
        sql_generation_dataset.append(
            {
                "db_id": data["db_id"],
                "input_sequence": input_sequence.strip(), 
                "output_sequence": output_sequence.strip(),
                "tc_original": tc_original
            }
        )

        # Preprare schema sequence, for schema prediction & value prediciton task.
        schema_sequence = ""
        for table_id in range(len(preprocessed_data["db_schema"])):
            table_name_original = preprocessed_data["db_schema"][table_id]["table_name_original"]
            # add table name
            schema_sequence += " | " + table_name_original + " : "
            
            column_info_list = []
            for column_id in range(len(preprocessed_data["db_schema"][table_id]["column_names_original"])):
                # extract column name
                column_name_original = preprocessed_data["db_schema"][table_id]["column_names_original"][column_id]
                db_contents = preprocessed_data["db_schema"][table_id]["db_contents"][column_id]
                # use database contents if opt.use_contents = True
                if opt.use_contents and len(db_contents) != 0:
                    column_contents = " , ".join(db_contents)
                    column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
                else:
                    column_info = table_name_original + "." + column_name_original

                column_info_list.append(column_info)
                
            # add column names
            schema_sequence += " , ".join(column_info_list)

        if opt.add_fk_info:
            for fk in preprocessed_data["fk"]:
                schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                    " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]
        
        # remove additional spaces in the schema sequence
        while "  " in schema_sequence:
            schema_sequence = schema_sequence.replace("  ", " ")


        # Second, schema prediction task
        # Find mentioned items in the given question 

        mentioned_schema_items = {}
        output_sequence = ""
        for table_id, table_label in zip(range(len(preprocessed_data["db_schema"])), preprocessed_data["table_labels"]):
            if table_label == 1:
                mentioned_columns = []
                table_name_original = preprocessed_data["db_schema"][table_id]["table_name_original"]
                output_sequence += "<table> " + table_name_original + " <column> "
                column_names_original = preprocessed_data["db_schema"][table_id]["column_names_original"]
                for column_id, column_label in zip(range(len(column_names_original)), preprocessed_data["column_labels"][table_id]):
                    if column_label == 1:
                        column_name_original = column_names_original[column_id]
                        mentioned_columns.append(column_name_original)
                        output_sequence += column_name_original + " , "
                mentioned_schema_items[table_name_original] = mentioned_columns
                output_sequence = output_sequence[:-2] + " "

        sp_prediction_dataset.append(
            {
                "input_sequence": (schema_task_prompt + question + schema_sequence).strip(),
                "output_sequence": output_sequence.strip(),
                "mentioned_schema_items": mentioned_schema_items
            }
        )

        # value prediction task
        
        o_v = ""
        mentioned_values = preprocessed_data["sql_values"]
        if len(mentioned_values) != 0:
            o_v = ", ".join(mentioned_values)
        else:
            o_v = "<none>"

        output_sequence = "<value>" + " " + o_v

        vp_prediction_dataset.append(
            {
                "input_sequence": (value_task_prompt + question + schema_sequence).strip(),
                "output_sequence": output_sequence.strip(),
                "mentioned_values":{
                    "values": mentioned_values if len(mentioned_values) != 0 else ["<none>"]
                }
            }
        )

    assert (len(sql_generation_dataset) == len(sp_prediction_dataset) == len(vp_prediction_dataset))       
    logger.info(f"Total size of cross-lingual pt generation dataset: {len(sql_generation_dataset)}")

    random.Random(42).shuffle(sql_generation_dataset)
    logger.info("Sql generation dataset validation split size capped to 1K.")
    sql_generation_train_dataset = sql_generation_dataset[1000:]
    sql_generation_dev_dataset = sql_generation_dataset[:1000]

    random.Random(42).shuffle(sp_prediction_dataset)
    logger.info("Schema prediction dataset validation split size capped to 1K.")
    sp_prediction_train_dataset = sp_prediction_dataset[1000:]
    sp_prediction_dev_dataset = sp_prediction_dataset[:1000]

    random.Random(42).shuffle(vp_prediction_dataset)
    logger.info("Value prediction dataset validation split size capped to 1K.")
    vp_prediction_train_dataset = vp_prediction_dataset[1000:]
    vp_prediction_dev_dataset = vp_prediction_dataset[:1000]

    # Save sql generation dataset 
    for split, dataset in zip(["train", "dev"], [sql_generation_train_dataset, sql_generation_dev_dataset]):
        save_path = os.path.join(opt.save_dir, f"sql_generation_{split}.json")
        with open(save_path, "w") as f:
            f.write(json.dumps(dataset, indent = 4, ensure_ascii = False))
            logger.info(f"Sql generation dataset {split} split saved to {save_path}")
            logger.info(f"Size: {len(dataset)}")

    # Save schema prediction dataset

    for split, dataset in zip(["train", "dev"], [sp_prediction_train_dataset, sp_prediction_dev_dataset]):
        save_path = os.path.join(opt.save_dir, f"schema_prediction_{split}.json")
        with open(save_path, "w") as f:
            f.write(json.dumps(dataset, indent = 4, ensure_ascii = False))
            logger.info(f"Schema prediction dataset {split} split saved to {save_path}")
            logger.info(f"Size: {len(dataset)}")

    # Save value prediction dataset
    for split, dataset in zip(["train", "dev"], [vp_prediction_train_dataset, vp_prediction_dev_dataset]):
        save_path = os.path.join(opt.save_dir, f"value_prediction_{split}.json")
        with open(save_path, "w") as f:
            f.write(json.dumps(dataset, indent = 4, ensure_ascii = False))
            logger.info(f"Value prediction dataset {split} split saved to {save_path}")
            logger.info(f"Size: {len(dataset)}")

if __name__ == "__main__":
    opt = parse_option()
    main(opt)
