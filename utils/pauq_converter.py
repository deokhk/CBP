# Convert format of PAUQ dataset to Spider dataset.

import argparse
import os 
import json 




def main(args):
    with open(args.pauq_file_path, 'r') as f:
        pauq_dataset = json.load(f)
    
    pauq_converted = []
    for data in pauq_dataset:
        data['query'] = data['query']['ru']
        data['question'] = data['question']['ru']
        data['sql'] = data['sql']['ru']
        data['query_toks'] = data['query_toks']['ru']
        data['query_toks_no_value'] = data['query_toks_no_values']['ru']
        data['question_toks'] = data['question_toks']['ru']
        pauq_converted.append(data)
    
    save_path = os.path.join(os.path.dirname(args.pauq_file_path), (os.path.basename(args.pauq_file_path).split('.')[0] + '_converted.json'))
    with open(save_path, 'w') as f:
        f.write(json.dumps(pauq_converted, indent = 2, ensure_ascii = False))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pauq_file_path", type = str, default = "./data/pauq/pauq_dev.json")
    
    args = parser.parse_args()
    main(args)