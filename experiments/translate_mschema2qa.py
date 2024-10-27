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


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/deokhk/research/pgrad_googleMT_Key.json"
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
    
    with open(args.mschema2qa_data_path, "r") as f:
        mschema2qa_dataset = json.load(f)
    
    translated_data = []
    
    for datapoint in tqdm(mschema2qa_dataset, desc=f"Translating original dataset to {args.target_lang}"):
        english_question = datapoint["question"]["en"]
        translated_question = translate_text(translate_client, english_question, target=args.target_lang)
        translated_data.append(
            {
                "mr": {
                    "thingtalk": {
                        "en": datapoint["mr"]["thingtalk"]["en"]
                    }
                },
                "translated_question": translated_question,
                "original_question": english_question
            }
        )


    data_dir = os.path.dirname(args.mschema2qa_data_path)
    save_path = os.path.join(data_dir, f"question_translated_to_{args.target_lang}.json")
    logger.info(f"Saving translated original dev data to {save_path}")
    with open(save_path, "w") as f:
        json.dump(translated_data, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mschema2qa_data_path", type=str, default="/home/deokhk/research/XSemPLR/dataset/mschema2qa/train.json",
                        help = 'File path for mschema2qa train path')
    parser.add_argument("--target_lang", type=str, default="en",
                        help="The target language to translate. Default is english")

    args = parser.parse_args()
    main(args)