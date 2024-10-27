# Translation quality assessment using GEMBA

import os 
import logging 
import argparse 
import openai 
import json 
import time
import re

import sys 
from tqdm import tqdm
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generate_and_evaluate_chatgpt_text2sql import load_api_key, chatcompletion_with_retry

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

language_codes = {
    "en": "English",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "es": "Spanish",
    "fa": "Farsi",
    "fi": "Finnish",
    "it": "Italian",
    "ja": "Japanese",
    "pl": "Polish",
    "tr": "Turkish"
}


def parse_numerical_answer(answer, min=None, max=None):
    # Taken from https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/prompt.py
    # get all numbers in a string
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 1:
        return int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    return None


def validate_stars(x):
    # Taken from https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba/prompt.py
    x = x.lower()
    # try to find all possible answers as sometimes it seems to be explaining itself
    possible_answers = set()

    # check if string x contains * characters
    if "*" in x:
        possible_answers.add(x.count("*"))
    if "★" in x:
        possible_answers.add(x.count("★"))

    x = f" {x} ".replace("\n", " ")
    # possible answers: "five stars", "5 stars", "five", "five starts: perfect translation", ...
    if " one " in x or "1 star" in x:
        possible_answers.add(1)
    if " two " in x or "2 star" in x:
        possible_answers.add(2)
    if " three " in x or "3 star" in x:
        possible_answers.add(3)
    if " four " in x or "4 star" in x:
        possible_answers.add(4)
    if " five " in x or "5 star" in x:
        possible_answers.add(5)

    numerical = parse_numerical_answer(x)
    if numerical is not None:
        possible_answers.add(numerical)

    if len(possible_answers) == 1:
        answer = possible_answers.pop()
        if 1 <= answer <= 5:
            return answer
    return None

def construct_prompt(examples, source_lang, target_lang):

    source_seg = examples["original_question"]
    target_seg = examples["generated_question"]

    prompt = f"""Score the following translation from {source_lang} to {target_lang} with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nStars:"""
    return prompt


def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    assert os.path.exists(args.openai_key_path), "Please put your OpenAI API key in the file: {}".format(args.openai_key_path)
    openai.api_key = load_api_key(args.openai_key_path)

    star_results = []
    quality_estimated_datapoints = []

    source_lang = language_codes[args.source_lang_code]
    target_lang = language_codes[args.target_lang_code]

    # Leave datapoints only with the target language 
    datapoints_with_target_lang = [datapoint for datapoint in data if args.target_lang_code in datapoint["detected_language"]]

    for datapoint in tqdm(datapoints_with_target_lang, desc="Evaluating translation quality on {}".format(target_lang)):
        prompt = construct_prompt(datapoint, source_lang, target_lang)

        (predictions, pred_model_name) = chatcompletion_with_retry(args, prompt, args.model)
        star_result = validate_stars(predictions.strip())

        if star_result is not None:
            star_results.append(star_result)
        datapoint["stars"] = star_result

        quality_estimated_datapoints.append(datapoint)

    original_file_dir = os.path.dirname(args.data_path)
    original_filename = os.path.basename(args.data_path).split(".")[0] # Remove file name extension
    save_path = os.path.join(original_file_dir, f"quality_estimated_{original_filename}.json")

    valid_count = len([star_res for star_res in star_results if isinstance(star_res, int)])
    valid_sum = 0
    for star_res in star_results:
        if isinstance(star_res, int):
            valid_sum += star_res
    
    average_rating = valid_sum / valid_count
            
    logger.info(f"Average rating: {average_rating}")
    
    one_star_count = len([star_res for star_res in star_results if star_res == 1])
    two_star_count = len([star_res for star_res in star_results if star_res == 2])
    three_star_count = len([star_res for star_res in star_results if star_res == 3])
    four_star_count = len([star_res for star_res in star_results if star_res == 4])
    five_star_count = len([star_res for star_res in star_results if star_res == 5])

    logger.info(f"1 star count: {one_star_count}")
    logger.info(f"2 star count: {two_star_count}")
    logger.info(f"3 star count: {three_star_count}")
    logger.info(f"4 star count: {four_star_count}")
    logger.info(f"5 star count: {five_star_count}")


    logger.info(f"Saving quality estimation info augmented datapoints to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(quality_estimated_datapoints, f, indent=4, ensure_ascii=False)
    
    save_path = os.path.join(original_file_dir, f"average_rating_{args.target_lang_code}.txt")
    logger.info("Saving average rating to: {}".format(save_path))
    with open(save_path, "w") as f:
        f.write(str(average_rating))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data generation arguments 
    parser.add_argument("--openai_key_path", type=str, default="/home/deokhk/research/ZX-seq2seq/openai_key_nlpserver2.txt", help="The path to the openai key file.")
    parser.add_argument("--model", type=str, 
                        help="Model to be used for generating context", default="gpt-3.5-turbo")
    parser.add_argument("--data_path", type=str)

    parser.add_argument("--source_lang_code", type=str, default="en")
    parser.add_argument("--target_lang_code", type=str, default="zh")

    args = parser.parse_args()
    main(args)