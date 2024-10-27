import argparse 
import json 
import os 
import logging 
import six
from google.cloud import translate_v2 as translate
from nltk.tokenize import word_tokenize

from tqdm import tqdm 


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/deokhk/research/ZX-seq2seq/translator-366508-8cf6d9b23c09.json"
translate_client = translate.Client()


def translate_text(translate_client, text, target="en"):
    """
    Given a text(string), translate it into a target language. (english by default)
    return translated version of the given text.
    """    
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]



def main(args):
    
    with open(args.original_dev_filepath, "r") as f:
        original_dev_data = json.load(f)
    
    translated_original_dev_data = []
    
    # For original dev data, we need to translate question only.
    for dev_original in tqdm(original_dev_data, desc=f"Translating original dataset to {args.target_lang}"):
        orginal_question = dev_original["question"]
        translated_question = translate_text(translate_client, orginal_question, target=args.target_lang)
        translated_original_dev_data.append(
            {
                "db_id": dev_original["db_id"],
                "original_question" : orginal_question,
                "translated_question": translated_question,
                "query": dev_original["query"]
            }
        )


    original_dev_dir = os.path.dirname(args.original_dev_filepath)
    original_dev_filename = os.path.basename(args.original_dev_filepath).split(".")[0] # Remove file name extension
    original_dev_save_path = os.path.join(original_dev_dir, f"question_translated_to_{args.target_lang}.json")
    logger.info(f"Saving translated original dev data to {original_dev_save_path}")
    with open(original_dev_save_path, "w") as f:
        json.dump(translated_original_dev_data, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dev_filepath", type=str, default="./data/Cspider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument("--target_lang", type=str, default="en",
                        help="The target language to translate. Default is english")

    args = parser.parse_args()
    main(args)