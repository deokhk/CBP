# This file will generate the text2sql results using the chatGPT api.
# 1000 inference with gpt-3.5-turbo takes about 1 hour.
# It costs about 0.1 USD per 1000 inference.

import pathlib, os
import logging
import argparse 
import random 
import openai 
import json 
import time 
import requests 

from preprocessing import normalization
from utils.spider_metric.evaluator import EvaluateTool
from tqdm import tqdm 
from utils.text2sql_decoding_utils import get_cursor_from_path, execute_sql  
from func_timeout import func_set_timeout, FunctionTimedOut

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


logger = logging.getLogger(__name__)

def load_api_key(file_path):
    with open(file_path, "r") as f:
        api_key = f.read().strip()
    return api_key


def chatcompletion_with_retry(args, message, return_model_type=False):
    retries = 0
    MAX_RETRIES= 500
    RETRY_DELAY = 2 
    generated_context = "error"
    generated_model_name = "None"

    while retries < MAX_RETRIES:
        try:
            response = openai.ChatCompletion.create(model=args.model, 
                                                            messages=[
                                                                {"role": "user", "content": message},
                                                            ], temperature=0, max_tokens=512)
            generated_context = response["choices"][0]["message"]["content"]
            generated_model_name = response["model"]
            if return_model_type:
                return (generated_context, generated_model_name)
            else:
                return generated_context
        except:
            logger.info("Error in chat completion, retrying...")
            logger.warning("sleep")
            time.sleep(RETRY_DELAY)
            retries += 1




def postprocess_prediction(sql):
    # We append "SELECT" clause to the predicted SQL query 
    # and apply normalization (e.g. remove newlines, tabs, etc.)
    sql = "SELECT " + sql
    sql = sql.replace("\n", " ")
    sql = sql.replace("\t", " ")
    normalized_sql = normalization(sql)
    return normalized_sql

def main(args):

    logger.info(args)

    assert os.path.exists(args.openai_key_path), "Please put your OpenAI API key in the file: {}".format(args.openai_key_path)
    openai.api_key = load_api_key(args.openai_key_path)

    if os.path.exists(args.output_path) and not args.reprocess:
        logger.info("Output path already exists, quiting.")
        return -1 

    logger.info("Loading dataset from {}".format(args.preprocessed_eval_dataset_path))
    with open(args.preprocessed_eval_dataset_path) as f:
        dataset = json.load(f)
    
    # 1-shot 
    data = dataset[0]

    preprocessed_data = dict()
    preprocessed_data["question"] = data["question"]
    preprocessed_data["sql"] = data["sql"] # unused
    preprocessed_data["norm_sql"] = data["norm_sql"]
    preprocessed_data["db_id"] = data["db_id"]
    preprocessed_data["db_schema"] = []

    # record table & column labels
    preprocessed_data["table_labels"] = data["table_labels"]
    preprocessed_data["column_labels"] = data["column_labels"]
    preprocessed_data["sql_values"] = data["sql_values"]

    table_ids = [idx for idx, _ in enumerate(data["db_schema"])]
    
    for table_id in table_ids:
        new_table_info = dict()
        new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
        # record ids of used columns

        new_table_info["column_names_original"] = data["db_schema"][table_id]["column_names_original"]
        new_table_info["db_contents"] = data["db_schema"][table_id]["db_contents"]
        
        preprocessed_data["db_schema"].append(new_table_info)

    # record foreign keys
    preprocessed_data["fk"] = data["fk"]

    question = preprocessed_data["question"]
    query = preprocessed_data["norm_sql"]

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
            # use database contents if args.use_contents = True
            if args.use_contents and len(db_contents) != 0:
                column_contents = " , ".join(db_contents)
                column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
            else:
                column_info = table_name_original + "." + column_name_original

            column_info_list.append(column_info)
            
        # add column names
        schema_sequence += " , ".join(column_info_list)

    if args.add_fk_info:
        for fk in preprocessed_data["fk"]:
            schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]
    
    # remove additional spaces in the schema sequence
    while "  " in schema_sequence:
        schema_sequence = schema_sequence.replace("  ", " ")

    example_prompt = f"""
    ### Complete sqlite SQL query only and with no explanation \n ### Sqlite SQL tables, with their properties: \n# \n#
    {schema_sequence}# \n ### {question} \n {query}
    """

    output_dataset = []
    for data_id, data in tqdm(enumerate(dataset), desc="Generating prompts.."):
        preprocessed_data = dict()
        preprocessed_data["question"] = data["question"]
        preprocessed_data["sql"] = data["sql"] # unused
        preprocessed_data["norm_sql"] = data["norm_sql"]
        preprocessed_data["db_id"] = data["db_id"]
        preprocessed_data["db_schema"] = []

        # record table & column labels
        preprocessed_data["table_labels"] = data["table_labels"]
        preprocessed_data["column_labels"] = data["column_labels"]
        preprocessed_data["sql_values"] = data["sql_values"]

        table_ids = [idx for idx, _ in enumerate(data["db_schema"])]
        
        for table_id in table_ids:
            new_table_info = dict()
            new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
            # record ids of used columns

            new_table_info["column_names_original"] = data["db_schema"][table_id]["column_names_original"]
            new_table_info["db_contents"] = data["db_schema"][table_id]["db_contents"]
            
            preprocessed_data["db_schema"].append(new_table_info)

        # record foreign keys
        preprocessed_data["fk"] = data["fk"]

        question = preprocessed_data["question"]
        query = preprocessed_data["norm_sql"]

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
                # use database contents if args.use_contents = True
                if args.use_contents and len(db_contents) != 0:
                    column_contents = " , ".join(db_contents)
                    column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
                else:
                    column_info = table_name_original + "." + column_name_original

                column_info_list.append(column_info)
                
            # add column names
            schema_sequence += " , ".join(column_info_list)

        if args.add_fk_info:
            for fk in preprocessed_data["fk"]:
                schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                    " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]
        
        # remove additional spaces in the schema sequence
        while "  " in schema_sequence:
            schema_sequence = schema_sequence.replace("  ", " ")


        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in preprocessed_data["db_schema"]:
            for column_name_original in ["*"] + table["column_names_original"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)

        prompt = f"""
        ### Complete sqlite SQL query only and with no explanation \n ### Sqlite SQL tables, with their properties: \n# \n#
        {schema_sequence}# \n ### {question} \n SELECT
        """

        prompt = example_prompt + prompt
        output_dataset.append(
            {
                "db_id": data["db_id"],
                "query": query,
                "question": question,
                "prompt": prompt,
                "tc_original": tc_original
            }
        )

    save_file = open(args.output_path, 'w')

    # generate predictions
    predict_sqls = []
    for data in tqdm(output_dataset, desc="Generating predictions using chatGPT.."):
        pred_executable_sql = "sql placeholder"        
        db_id = data["db_id"]
        db_file_path = args.db_path + "/{}/{}.sqlite".format(db_id, db_id)

        cursor = get_cursor_from_path(db_file_path)
        (pred_sql, prediction_model_name) = chatcompletion_with_retry(args, data["prompt"], return_model_type=True)
        pred_sql = postprocess_prediction(pred_sql)
        pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
        
        try:
            # Note: execute_sql will be success for empty string
            assert len(pred_sql) > 0, "pred sql is empty!"

            results = execute_sql(cursor, pred_sql)
            # if the current sql has no execution error, we record and return it
            pred_executable_sql = pred_sql
            cursor.close()
            cursor.connection.close()
        except Exception as e:
            cursor.close()
            cursor.connection.close()
        except FunctionTimedOut as fto:
            del cursor

        data["predicted_sql"] = pred_executable_sql
        data["model"] = prediction_model_name

        # For printing purposes, we order the keys in the json file
        ordered_data = {
                "db_id": data["db_id"],
                "question": data["question"],
                "query": data["query"],
                "predicted_sql": pred_executable_sql,
                "model": prediction_model_name,
                "prompt": data["prompt"],
                "tc_original": data["tc_original"]
            }

        json.dump(ordered_data, save_file, ensure_ascii=False)

        predict_sqls.append(pred_executable_sql)
        save_file.write("\n")

    save_file.close()

    logger.info("Predictions saved to {}".format(args.output_path))
    logger.info("Now evaluating the generated predictions..")

    # Beware of the evluation orders!

    evaluator = EvaluateTool()
    evaluator.register_golds(args.original_eval_dataset_path, args.db_path)
    spider_metric_result = evaluator.evaluate(predict_sqls)

    logger.info('exact_match score: {}'.format(spider_metric_result["exact_match"]))
    logger.info('exec score: {}'.format(spider_metric_result["exec"]))

    # Save the evaluation results 
    # The result is between 0 and 1, where 1 is the best

    eval_result = dict()
    eval_result["EM"] = spider_metric_result["exact_match"]
    eval_result["EXEC"] = spider_metric_result["exec"]

    
    os.makedirs(os.path.dirname(args.eval_results_path), exist_ok = True)
    with open(args.eval_results_path, "w") as f:
        f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))

    logger.info("Evaluation results saved to {}".format(args.eval_results_path))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data generation arguments 
    parser.add_argument("--openai_key_path", type=str, default="/home/deokhk/research/ZX-seq2seq/openai_key_nlpserver2.txt", help="The path to the openai key file.")
    parser.add_argument("--model", type=str, 
                        help="Model to be used for generating context", default="gpt-3.5-turbo")
    parser.add_argument("--original_eval_dataset_path", type = str, default = "./data/Cspider/dev.json")
    parser.add_argument('--preprocessed_eval_dataset_path', type = str, default = "./data/preprocessed_data/preprocessed_dev_cspider.json",
                        help = 'filepath of the preprocessed input dataset.')
    parser.add_argument("--db_path", type=str, default="./database")
    
    parser.add_argument("--output_path", type=str,
                        help="Output path to store the predicted sql and related datas", default="./predictions/chatgpt_cspider_dev_prediction.jsonl")
    parser.add_argument("--eval_results_path", type=str,
                        help="Output path to store the evaluation results", default="./eval_results/text2sql/eval.json")
    parser.add_argument('--mode', type = str, default = "eval",
                        help = 'type of the input dataset, options: train, eval, test.')
    parser.add_argument('--use_contents', action = 'store_true',
                        help = 'whether to add database contents in the input sequence.')
    parser.add_argument('--add_fk_info', action = 'store_true',
                    help = 'whether to add foreign key in the input sequence.')

    parser.add_argument("--reprocess", action = 'store_true',
                        help="Whether to reprocess the input dataset, including chatgpt inference.")
    args = parser.parse_args()
    main(args)