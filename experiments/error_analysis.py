# Load a pretrained model and evaluate on the test set
# Save the predicted sql along with gold sql, with additional information


import argparse
import os
import json
import torch 
import shutil 
import logging 

from text2sql import _test

logger = logging.getLogger(__name__)

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for selecting the best ckpt.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size. Note that this is a effective batch size')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "./models/mt5-large-16-text2sql/best_model/",
                        help = 'save path of fine-tuned text2sql model.')
    parser.add_argument('--model_name_or_path', type=str, default= "mt5",
                        help="Type of model used for evaluation") # TODO: the name is confusing.. change it to model_type later.
    parser.add_argument('--eval_results_path', type = str, default = "./eval_results/text2sql",
                        help = 'the evaluation results of fine-tuned text2sql models.')
    parser.add_argument('--eval_file_name', type=str, default="eval_res.txt")

    parser.add_argument('--mode', type = str, default = "eval",
                        help='eval.')
    parser.add_argument('--dev_filepath', type = str, default = "./data/pre-processing/resdsql_test.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "./data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "./data/spider/database",
                        help = 'file path of database.')
    
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--output", type = str, default = "predicted_sql.txt")
    
    opt = parser.parse_args()

    return opt

    
if __name__ == "__main__":
    opt = parse_option()

    save_path = opt.save_path
    os.makedirs(opt.eval_results_path, exist_ok = True)

    dev_filepath = opt.dev_filepath
    original_dev_filepath = opt.original_dev_filepath

    logger.info("Start evaluating ckpt at: {}".format(opt.save_path))
    with open(opt.eval_results_path+f"/{opt.eval_file_name}", "w") as f:
        f.write("Evaluating...")
    
    em, exec = _test(opt)


    eval_result = dict()
    eval_result["ckpt"] = opt.save_path
    eval_result["EM"] = em
    eval_result["EXEC"] = exec

    with open(opt.eval_results_path+f"/{opt.eval_file_name}", "w") as f:
        f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))

    logger.info("Finish evaluating ckpt at: {}".format(opt.save_path))
    logger.info("EM: {}".format(em))
    logger.info("EXEC: {}".format(exec))    

