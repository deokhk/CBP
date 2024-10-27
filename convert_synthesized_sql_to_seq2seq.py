import argparse 
import os 
import json 
import logging
from tqdm import tqdm 
from sql_metadata import Parser

from preprocessing import get_db_schemas, normalization, get_db_contents
from text2sql_data_generator import prepare_input_and_output

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


sql_keywords = ['select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', \
    'except', 'join', 'on', 'as', 'not', 'between', 'in', 'like', 'is', 'exists', 'max', 'min', \
        'count', 'sum', 'avg', 'and', 'or', 'desc', 'asc', 'order by', 'group by', 'distinct']
ops = ["=", "!=", ">", ">=", "<", "<="]


def main(args):
    logger.info("Loading synthesized SQL from {}".format(args.synthesized_sql_file))
    with open(args.synthesized_sql_file, "r") as f:
        synthesized_sql_dataset = json.load(f)

    # Load db schemas 
    all_db_infos = json.load(open(args.table_path))
    db_schemas = get_db_schemas(all_db_infos)

    sql_generation_dataset = []

    for data in tqdm(synthesized_sql_dataset, desc=f"Converting to Seq2seq format for question generation"):

        # We need to retrieve schema item..
        db_id = data["db_id"]
        
        question = "DUMMY" # dummy question, for compatibility with the original code
        sql = data["query"].strip()
        norm_sql = normalization(sql).strip()
        sql_tokens = norm_sql.split()

        preprocessed_data = {}
        preprocessed_data["question"] = question # dummy question, for compatibility with the original code
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
                args.db_path
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
        
        preprocessed_data["sql_values"] = [] # For compatibility with the original code                        
        # First, sql generation task 
        input_sequence, output_sequence = prepare_input_and_output(args, preprocessed_data)
        
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
    
    # Save 
    logger.info("Total {} data points".format(len(sql_generation_dataset)))
    orig_file_name = os.path.basename(args.synthesized_sql_file)
    save_file_name = "spider_" + orig_file_name.replace(".json", "_seq2seq.json")
    save_path = os.path.join(args.output_dir, save_file_name)
    logger.info("Saving to {}".format(save_path))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(sql_generation_dataset, f, indent=4)
    logger.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesized_sql_file", type=str, default="/home/deokhk/research/tensor2struct-public/experiments/sql2nl/data-synthetic/new-examples-no-question-6-128.json")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_seq2seq")

    parser.add_argument('--table_path', type = str, default = "./data/spider/tables.json")
    parser.add_argument('--db_path', type = str, default = "./database", 
                        help = "the filepath of database.")

    parser.add_argument('--use_contents', action = 'store_true', default=True,
                        help = 'whether to add database contents in the input sequence.')
    parser.add_argument('--add_fk_info', action = 'store_true', default=True,
                    help = 'whether to add foreign key in the input sequence.')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql. Legacy.")
    parser.add_argument("--stepgen", action = "store_true",
                help = "Apply step-by-step generation for sql. Legacy.")

    args = parser.parse_args()
    main(args)