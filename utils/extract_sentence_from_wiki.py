# Extract sentences from wiki dump 
# Download wiki dump from https://dumps.wikimedia.org/backup-index.html
# Value given to extracted_wiki should be extrated wiki from bz2 file
# Check https://github.com/attardi/wikiextractor : Note that you should give --json --no-templates flag to the wikiextractor

# Usage: python utils/extract_sentence_from_wiki.py --extracted_wiki wikis/am/text --lang am --num_sentences 100000


import os
import argparse
import glob
import json
from tqdm import tqdm 

from datasets import Dataset, DatasetDict
from blingfire import text_to_sentences
# from amseg.amharicSegmenter import AmharicSegmenter # Temporary commented out 

sent_punct = []
word_punct = []
# segmenter = AmharicSegmenter(sent_punct,word_punct)

def get_files(data_dir):
    files = []
    for filename in glob.glob(f"{data_dir}/*"):
        if os.path.isdir(filename):
            files.extend(get_files(filename))
        else:
            files.append(filename)
    return files

def main(args):

    files = get_files(args.extracted_wiki)
    parsed_sentences = []
    num_dev_datapoints = 1000
    for file in tqdm(files, desc="Extracting sentences from wiki dumps.."):
        # load jsonl file 
        with open(file, "r") as f:
            json_list = list(f)
        # get the sentences
        sentences = []
        for json_str in json_list:
            result = json.loads(json_str)
            
            if args.lang == "am":
                # sentences = segmenter.tokenize_sentence(result["text"])
                pass # temporary comment out amseg as there's compatability issue with pytteserract
            else:
                sentences = text_to_sentences(result["text"]).split("\n")
                # filter out sentences that are too short (e.g. "")
            sentences = [s_ for s_ in sentences if len(s_) > 7]
            parsed_sentences.extend(sentences)
            if len(parsed_sentences) > args.num_sentences + num_dev_datapoints:
                parsed_sentences = parsed_sentences[:args.num_sentences + num_dev_datapoints]
                break
        if len(parsed_sentences) >= args.num_sentences + num_dev_datapoints:
            break
    
    train_sentences = parsed_sentences[:args.num_sentences]
    dev_sentences = parsed_sentences[args.num_sentences:]

    print(f"Number of train sentences: {len(train_sentences)}")
    print(f"Number of dev sentences: {len(dev_sentences)}")
    # save it as a dataset
    train_dict = {"sentence": train_sentences}
    dev_dict = {"sentence": dev_sentences}
    train_dataset = Dataset.from_dict(train_dict)
    dev_dataset = Dataset.from_dict(dev_dict)

    dataset = DatasetDict({"train": train_dataset, "dev": dev_dataset})

    # Upload to hub 
    dataset_name = f"{args.lang}_wiki_sentences_{args.num_sentences}"

    dataset.push_to_hub("deokhk/{}".format(dataset_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_wiki", type=str, default="wikis/am/text")
    parser.add_argument("--lang", type=str, default="am")
    parser.add_argument("--num_sentences", type=int, default=100000)

    args = parser.parse_args()
    main(args)
