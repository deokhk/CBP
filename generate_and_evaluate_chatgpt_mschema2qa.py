
import os 
import logging 
import argparse 
import openai 
import json 
import time
import random 
import sys 
import asyncio
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

import json 
import time 
import requests 
import tiktoken

from tqdm.asyncio import tqdm
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generate_and_evaluate_chatgpt_text2sql import load_api_key, chatcompletion_with_retry
from utils.mschema2qa_metric.evaluator import MSchema2QAEvaluateTool

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def construct_prompt(examples):
    prompt = ""
    for example in examples:
        question = example["question"]["en"]
        mr = example["mr"]["thingtalk"]["en"]
        prompt += f"Translate the following question into thingtalk QL:{question}\n"
        prompt += f"MR: {mr}\n"
    return prompt


def main(args):
    # Open the data file, and construct the prompt.
    with open(args.train_data_path, "r") as f:
        train_data = json.load(f)

    random.seed(42)
    random.shuffle(train_data)
    examples = train_data[:args.num_shots]
    prompt_example = construct_prompt(examples)

    assert os.path.exists(args.openai_key_path), "Please put your OpenAI API key in the file: {}".format(args.openai_key_path)
    OPENAI_API_KEY = load_api_key(args.openai_key_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    encoding = tiktoken.get_encoding("cl100k_base")

    eval_langs = examples[0]["question"].keys()
    eval_langs = [lang for lang in eval_langs if lang != "en"]
    logger.info("Evaluating on the following languages: {}".format(eval_langs))

    token_count = 0
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)
    
    logger.info(f"prompt length: {len(prompt_example)}")


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
    async def get_completion(prompt_dict, model_name, session, semaphore):
        async with semaphore:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_dict["prompt"]}],
                "temperature": 0,
                "max_tokens": 512
            }) as resp:

                response_json = await resp.json()

                pred_mr = response_json["choices"][0]['message']["content"]
                # Post-processing
                pred_mr = pred_mr.strip()
                return {
                    "gold_mr": prompt_dict["gold_mr"],
                    "pred_mr": pred_mr,
                    "model": model_name
                }

    async def get_completion_list(content_list, max_parallel_calls):
        semaphore = asyncio.Semaphore(value=max_parallel_calls)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10)) as session:
            return await tqdm.gather(*[get_completion(content, args.model_name, session, semaphore) for content in content_list])

    for lang in eval_langs:
        pred_mrs = []
        gold_mrs = []
        prompts = []
        token_count = 0
        save_path = os.path.join(args.output_dir, f"mschema2qa_{lang}_pred.jsonl")

        save_file = open(save_path, 'w')
        logger.info(f"Evaluating on language: {lang}")
        for datapoint in test_data:
            question = datapoint["question"][lang]
            mr = datapoint["mr"]["thingtalk"][lang]
            gold_mrs.append(mr)

            prompt = prompt_example
            prompt += f"Translate the following question into thingtalk QL:{question}\n"
            prompt += f"MR: "

            prompts.append({
                "prompt": prompt,
                "gold_mr": mr,
            })
            token_count += len(encoding.encode(prompt))

        logger.info(f"Total token count: {token_count}")
        cost= 0.50 * (token_count / 1e6) + 1.5 * ((token_count / 1e6) / 6)
        logger.info(f"Expected cost: {cost} USD")

        logger.info(f"Start inference for {lang}")
        results = asyncio.run(get_completion_list(prompts,  args.max_parallel_calls))

        # Save the results 
        for result in results:
            pred_mrs.append(result["pred_mr"])
            save_file.write(json.dumps(result) + "\n")

        evaluator = MSchema2QAEvaluateTool(args)
        metric_result = evaluator.evaluate(pred_mrs, gold_mrs)
        print('exact_match score: {}'.format(metric_result["exact_match"]))

        save_file.write("exact_match score: {}".format(metric_result["exact_match"]))
        save_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data generation arguments 
    parser.add_argument("--openai_key_path", type=str, default="/home/deokhk/research/ZX-seq2seq/openai_key_nlpserver2.txt", help="The path to the openai key file.")
    parser.add_argument("--model_name", type=str, 
                        help="Model to be used for generating context", default="gpt-3.5-turbo")
    parser.add_argument("--train_data_path", type=str, default="/home/deokhk/research/XSemPLR/dataset/mschema2qa/train.json")
    parser.add_argument("--test_data_path", type=str, default="/home/deokhk/research/XSemPLR/dataset/mschema2qa/test.json")
    
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--output_dir", type=str,
                        help="Output dir to store the predicted sql", default="/home/deokhk/research/ZX-seq2seq/predictions/chatgpt_mschema2qa_pred")
    
    # Parallelism arguments
    parser.add_argument("--max_parallel_calls", type=int, default=20, help="Maximum parallel calls for the API.")

    args = parser.parse_args()
    main(args)