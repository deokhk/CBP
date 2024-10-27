import argparse
import os
import json
import torch 
import shutil 
import wandb 
import logging 

from text2sql import _test_mschema2qa

logger = logging.getLogger(__name__)

# By default, we evaluate


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for selecting the best ckpt.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size. Note that this is a effective batch size')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "./models/text2sql",
                        help = 'save path of fine-tuned text2sql models.')
    parser.add_argument('--model_name_or_path', type=str, default= "mt5",
                        help="Type of model used for evaluation")
    parser.add_argument('--eval_results_path', type = str, default = "./eval_results/text2sql",
                        help = 'the evaluation results of fine-tuned text2sql models.')
    parser.add_argument('--mode', type = str, default = "eval",
                        help='eval.')
    parser.add_argument('--dev_filepath', type = str, default = "./data/pre-processing/resdsql_test.json",
                        help = 'file path of test2sql dev set.')
    
    parser.add_argument("--dataset_lang", type=str, choices=["en", "ar", "de", "es", "fa", "fi", "it", "ja", "pl", "tr", "zh"], default="en")  
    parser.add_argument("--cross_dataset_lang", type=str, choices=["en", "ar", "de", "es", "fa", "fi", "it", "ja", "pl", "tr", "zh"], default="zh") 

    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--output", type = str, default = "predicted_sql.txt")
    
    parser.add_argument("--cross_eval_every_epoch", action="store_true", help="Enable for cross evaluation every epoch")

    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for wandb logging")
    parser.add_argument("--wandb_log", action="store_true", help="Enable for logging to wandb")

    parser.add_argument("--save_predictions", action="store_true", default=False)
    parser.add_argument("--save_predictions_path", type=str, default="./predictions/mschema2qa")

    opt = parser.parse_args()

    return opt

    
if __name__ == "__main__":
    opt = parse_option()
    
    if opt.wandb_log:
        wandb_exp_name = opt.exp_name if opt.exp_name is not None else "eval"
        wandb.init(
            project="ZX_seq2seq",
            name=wandb_exp_name,
        )

    print("Evaluating Mschema2QA")
    ckpt_names = os.listdir(opt.save_path)
    ckpt_names = sorted(ckpt_names, key = lambda x:eval(x.split("-")[1]))

    eval_lang = opt.dataset_lang
    cross_eval_lang = opt.cross_dataset_lang

    print("ckpt_names:", ckpt_names)

    save_path = opt.save_path
    os.makedirs(opt.eval_results_path, exist_ok = True)

    eval_results = []
    cross_eval_results = []
    for ckpt_name in ckpt_names:
        # if the current ckpt is being evaluated or has already been evaluated
        if "{}.txt".format(ckpt_name) in os.listdir(opt.eval_results_path):
            # is being evaluated
            with open(os.path.join(opt.eval_results_path, f"{ckpt_name}.txt"), "r") as f:
                if len(f.readlines()) == 1:
                    continue
            # has already been evaluated
            with open(os.path.join(opt.eval_results_path, f"{ckpt_name}.txt"), "r") as f:
                eval_result = json.load(f)
                eval_results.append(eval_result)
        # otherwise, we start evaluating the current ckpt
        else:
            logger.info("Start evaluating ckpt: {}".format(ckpt_name))
            with open(os.path.join(opt.eval_results_path, f"{ckpt_name}.txt"), "w") as f:
                f.write("Evaluating...")
            
            opt.save_path = os.path.join(save_path, ckpt_name)
            opt.dataset_lang = eval_lang
            em = _test_mschema2qa(opt)


            eval_result = dict()
            eval_result["ckpt"] = opt.save_path
            eval_result["EM"] = em

            with open(os.path.join(opt.eval_results_path, f"{ckpt_name}.txt"), "w") as f:
                f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))
            
            logger.info("ckpt name: {}".format(ckpt_name))

            eval_results.append(eval_result)
            logger.info(eval_results)

            if opt.wandb_log and not opt.cross_eval_every_epoch:
                step = int(eval(ckpt_name.split("-")[1]))
                wandb.log({"EM": eval_result["EM"], "STEP": step}, step)

            if opt.cross_eval_every_epoch:
                logger.info(f"Start testing ckpt on {opt.cross_dataset_lang}")
                with open(os.path.join(opt.eval_results_path, f"{ckpt_name}_{opt.cross_dataset_lang}.txt"), "w") as cf:
                    cf.write("Evaluating...")
                    
                    # Override the dev_filepath and original_dev_filepath
                    opt.dataset_lang = cross_eval_lang
                    cross_em = _test_mschema2qa(opt)


                    cross_eval_result = dict()
                    cross_eval_result["ckpt"] = opt.save_path
                    cross_eval_result["EM"] = cross_em

                    if opt.wandb_log:
                        step = int(eval(ckpt_name.split("-")[1]))
                    
                    logger.info(cross_eval_result)

                    cf.write(json.dumps(cross_eval_result, indent = 2, ensure_ascii = False))
                wandb.log({"EM": eval_result["EM"], f"{opt.cross_dataset_lang}_EM": cross_eval_result["EM"]}, step)
                cross_eval_results.append(cross_eval_result)

    for eval_result in eval_results:
        print("ckpt name:", eval_result["ckpt"])
        print("EM:", eval_result["EM"])
        print("-----------")
    
    if opt.cross_eval_every_epoch:
        for cross_eval_result in cross_eval_results:
            print("ckpt name:", cross_eval_result["ckpt"])
            print("EM:", cross_eval_result["EM"])
            print("-----------")
    

    em_list = [er["EM"] for er in eval_results]

    # find best EM ckpt
    best_em = 0.00
    best_em_idx = 0

    for idx, em in enumerate(em_list):
        if em > best_em:
            best_em = em
            best_em_idx = idx
            
    print("Best EM ckpt:", eval_results[best_em_idx])

    if opt.cross_eval_every_epoch:
        # Find best EM ckpt on cross-lingual dev set
        cross_best_em = 0
        cross_best_em_idx = 0

        for idx, cross_eval_result in enumerate(cross_eval_results):
            if cross_eval_result["EM"] > cross_best_em:
                cross_best_em = cross_eval_result["EM"] 
                cross_best_em_idx = idx

    # We only keep the checkpoint with best EM ckpt and best cross-lingual EM ckpt. Delete the others.
    
    if opt.cross_eval_every_epoch:
        for idx, eval_result in enumerate(eval_results):
            if (idx != best_em_idx) and (idx != cross_best_em_idx):
                shutil.rmtree(eval_result["ckpt"], ignore_errors = True)
    else:
        for idx, eval_result in enumerate(eval_results):
            if idx != best_em_idx:
                shutil.rmtree(eval_result["ckpt"], ignore_errors = True)
    

    best_ckpt_name = ckpt_names[best_em_idx]
    
    print(f"Deleted all ckpts except the best EM+EXEC ckpt : {best_ckpt_name}.")

    print(f"Now testing on cross-lingual dev sets using the best EM+EXEC ckpt : {best_ckpt_name}.")

    """
    Testing on cross-lingual dev sets using the best EM+EXEC ckpt
    """
    print("Start testing ckpt on cross-lingual dev set: {}".format(best_ckpt_name))
    with open(os.path.join(opt.eval_results_path, f"{best_ckpt_name}_cross.txt"), "w") as f:
        f.write("Evaluating on cross-lingual dev set...")
    
    opt.save_path = os.path.join(save_path, ckpt_names[best_em_idx])
    # Override the dev_filepath and original_dev_filepath
    opt.dataset_lang = cross_eval_lang
    em = _test_mschema2qa(opt)

    eval_result = dict()
    eval_result["ckpt"] = opt.save_path
    eval_result["EM"] = em

    print(f"Best ckpt cross-lingual evaluation result on {opt.cross_dataset_lang}: {eval_result}")
    with open(os.path.join(opt.eval_results_path, f"{best_ckpt_name}_{opt.cross_dataset_lang}.txt"), "w") as f:
        f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))
    
    # Rename best checkpoint to "best_model"
    renamed_best_ckpt_path = os.path.join(save_path, "best_model")
    shutil.move(opt.save_path, renamed_best_ckpt_path)
    print(f"Renamed best ckpt to {renamed_best_ckpt_path}")

    if opt.cross_eval_every_epoch:
        cross_best_ckpt_name = ckpt_names[cross_best_em_idx]
        opt.save_path = os.path.join(save_path, f"{cross_best_ckpt_name}")
        # Rename cross-lingual best checkpoint to "best_model_cross"
        renamed_cross_best_ckpt_path = os.path.join(save_path, "best_model_cross")
        shutil.move(opt.save_path, renamed_cross_best_ckpt_path)
        print(f"Renamed best cross-eval ckpt to {renamed_cross_best_ckpt_path}.")

    print("Done.")