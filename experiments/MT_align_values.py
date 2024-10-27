# Extract values from meaning representations and questions in mschema2qa dataset

import argparse 
import json 
#from transformers import AutoTokenizer, AutoModel
import transformers
import logging 
import re
import torch
import itertools 
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def extract_values_mschema2qa(input_string):
    pattern = r'"([^"]*)"'
    values = re.findall(pattern, input_string)

    # Trim values
    values = [value.strip() for value in values]
    return values

def extract_values_xspider(input_string):
    # Regular expression pattern to match substrings enclosed by single or double quotes
    
    pattern = r"['\"](.*?)['\"]"
    values = re.findall(pattern, input_string)

    # Trim values
    values = [value.strip() for value in values]
    return values

def tokenize_sentence(sent, tokenizer):

    tokenized = tokenizer.tokenize(sent)
    words = []
    word = ""
    for token in tokenized:
        if token.startswith("##"):
            word += token[2:]
        else:
            if word:
                words.append(word)
            word = token
    tokenized_sentence = " ".join(words)

    return tokenized_sentence


def find_match(value, sent_src, sent_tgt, align_words):
    src_to_tgt_dict = dict()

    for i, j in sorted(align_words):
        src_to_tgt_dict[sent_src[i]] = sent_tgt[j]

    matched_value = [src_to_tgt_dict.get(word, None) for word in value.split()]
    matched_value = [word for word in matched_value if word is not None]
    matched_value = " ".join(matched_value)

    return matched_value

def awesome_align(src, tgt, tokenizer, model):
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
    
    return (sent_src, sent_tgt, align_words)



def mschema2qa_label_projection(data, tokenizer, model):
    label_projected_data = []

    for datapoint in tqdm(data, desc="Projecting labels..", total=len(data)):
        mr = datapoint["mr"]["thingtalk"]["en"]
        src_question = datapoint["original_question"]
        tgt_question = datapoint["translated_question"]

        tokenized_src_question = tokenize_sentence(src_question, tokenizer)
        tokenized_tgt_question = tokenize_sentence(tgt_question, tokenizer)

        # Extract values from MR
        values = extract_values_mschema2qa(mr)
        (sent_src, sent_tgt, align_words) = awesome_align(tokenized_src_question, tokenized_tgt_question, tokenizer, model)

        proj_dict = {}
        for value in values:
            matched_value = find_match(value, sent_src, sent_tgt, align_words)
            proj_dict[value] = matched_value

        # Substitute values in the source mr
        for key, value in proj_dict.items():
            mr = mr.replace(key, value)

        label_projected_data.append({
            "mr": mr,
            "question": tgt_question
        })
    return label_projected_data

def xspider_label_projection(data, tokenizer, model):
    label_projected_data = []

    for datapoint in tqdm(data, desc="Projecting labels..", total=len(data)):
        src_question = datapoint["original_question"]
        tgt_question = datapoint["question"]
        mr = datapoint["query"]
        tokenized_src_question = tokenize_sentence(src_question, tokenizer)
        tokenized_tgt_question = tokenize_sentence(tgt_question, tokenizer)

        # Extract values from MR
        values = extract_values_xspider(mr)
        (sent_src, sent_tgt, align_words) = awesome_align(tokenized_src_question, tokenized_tgt_question, tokenizer, model)

        
        proj_dict = {}
        for value in values:
            matched_value = find_match(value, sent_src, sent_tgt, align_words)
            proj_dict[value] = matched_value

        # Substitute values in the source mr
        for key, value in proj_dict.items():
            mr = mr.replace(key, value)

        label_projected_datapoint = datapoint 
        label_projected_datapoint["query"] = mr
        label_projected_data.append(label_projected_datapoint)
    return label_projected_data


def main(args):

    logger.info("Loading model and tokenizers..")

    model = transformers.BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
    tokenizer = transformers.BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')

    with open(args.translated_data_path, "r") as f:
        data = json.load(f)
    
    logger.info("Projecting labels for {} data..".format(args.data_type))
    if args.data_type == "mschema2qa":
        label_projected_data = mschema2qa_label_projection(data, tokenizer, model)
    else:
        label_projected_data = xspider_label_projection(data, tokenizer, model)

    with open(args.output_path, "w") as f:
        json.dump(label_projected_data, f, indent=4, ensure_ascii=False)

    logger.info(f"Label projected data saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--translated_data_path", type=str, default="/home/deokhk/research/XSemPLR/dataset/mschema2qa/question_translated_to_ja.json",
                        help = 'File path for translated data')
    parser.add_argument("--align_method", type=str, choices=["awesome-align", "fast-align"], default="awesome-align")
    parser.add_argument("--output_path", type=str, default="label_projected_data.json",)

    parser.add_argument("--data_type", type=str, choices=["mschema2qa", "xspider"], required=True)
    args = parser.parse_args()
    main(args)