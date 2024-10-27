# Following "XRICL: Cross-lingual Retrieval-Augmented In-Context Learning for Cross-lingual Text-to-SQL Semantic Parsing",
# We extract those queries without using Chinese cell value from Cspider dataset and mark those subsets as "zh" setting.

import argparse 
import os 
import json 
import logging 
from tqdm import tqdm 

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def contain_chinese_char(text):
    for char in text:
        if u'\u4e00' <= char <= u'\u9fff':
            return True
    return False


def main(args):
    with open(args.dev_path, "r") as f:
        dev = json.load(f)
    
    zh_dev = []
    for ori_eval in tqdm(dev, desc="Extracting queries without chinese cell values.."):
        contain_chinese = False 
        query_toks = ori_eval["query_toks"]
        for query_tok in query_toks:
            if contain_chinese_char(query_tok):
                contain_chinese = True 
                break
        if not contain_chinese:
            zh_dev.append(ori_eval)

    logger.info(f"Number of queries in original eval dataset: {len(dev)}")
    logger.info(f"Number of queries without chinese cell values: {len(zh_dev)}")



    # Save the extracted dataset
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(os.path.join(args.output_dir, "zh_dev.json"), "w") as f:
        json.dump(zh_dev, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved extracted dataset to {os.path.join(args.output_dir, 'zh_dev.json')}")
    

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_path", type = str, default = "./data/Cspider/dev.json")
    parser.add_argument("--output_dir", type=str,
                        help="Output path to store the extracted datasets", default="./data/Cspider/")

    args = parser.parse_args()
    main(args)
    