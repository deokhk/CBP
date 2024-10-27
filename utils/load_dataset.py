import json
import os 
import random 
import numpy as np
import tqdm

from datasets import Dataset as hf_dataset, interleave_datasets
from torch.utils.data import Dataset
from typing import List 
from datasets import load_dataset

class ColumnAndTableClassifierDataset(Dataset):
    def __init__(
        self,
        dir_: str = None,
        use_contents: bool = True,
        add_fk_info: bool = True,
    ):
        super(ColumnAndTableClassifierDataset, self).__init__()

        self.questions: list[str] = []
        
        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []
        
        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []

            table_names_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []

            for table_id in range(len(data["db_schema"])):
                column_names_original_in_one_db.append(data["db_schema"][table_id]["column_names_original"])
                table_names_original_in_one_db.append(data["db_schema"][table_id]["table_name_original"])

                table_names_in_one_db.append(data["db_schema"][table_id]["table_name"])
                table_labels_in_one_db.append(data["table_labels"][table_id])

                column_names_in_one_db.append(data["db_schema"][table_id]["column_names"])
                column_labels_in_one_db += data["column_labels"][table_id]
                
                extra_column_info = ["" for _ in range(len(data["db_schema"][table_id]["column_names"]))]
                if use_contents:
                    contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(contents):
                        if len(content) != 0:
                            extra_column_info[column_id] += " , ".join(content)
                extra_column_info_in_one_db.append(extra_column_info)
            
            if add_fk_info:
                table_column_id_list = []
                # add a [FK] identifier to foreign keys
                for fk in data["fk"]:
                    source_table_name_original = fk["source_table_name_original"]
                    source_column_name_original = fk["source_column_name_original"]
                    target_table_name_original = fk["target_table_name_original"]
                    target_column_name_original = fk["target_column_name_original"]

                    if source_table_name_original in table_names_original_in_one_db:
                        source_table_id = table_names_original_in_one_db.index(source_table_name_original)
                        source_column_id = column_names_original_in_one_db[source_table_id].index(source_column_name_original)
                        if [source_table_id, source_column_id] not in table_column_id_list:
                            table_column_id_list.append([source_table_id, source_column_id])
                    
                    if target_table_name_original in table_names_original_in_one_db:
                        target_table_id = table_names_original_in_one_db.index(target_table_name_original)
                        target_column_id = column_names_original_in_one_db[target_table_id].index(target_column_name_original)
                        if [target_table_id, target_column_id] not in table_column_id_list:
                            table_column_id_list.append([target_table_id, target_column_id])
                
                for table_id, column_id in table_column_id_list:
                    if extra_column_info_in_one_db[table_id][column_id] != "":
                        extra_column_info_in_one_db[table_id][column_id] += " , [FK]"
                    else:
                        extra_column_info_in_one_db[table_id][column_id] += "[FK]"
            
            # column_info = column name + extra column info
            column_infos_in_one_db = []
            for table_id in range(len(table_names_in_one_db)):
                column_infos_in_one_table = []
                for column_name, extra_column_info in zip(column_names_in_one_db[table_id], extra_column_info_in_one_db[table_id]):
                    if len(extra_column_info) != 0:
                        column_infos_in_one_table.append(column_name + " ( " + extra_column_info + " ) ")
                    else:
                        column_infos_in_one_table.append(column_name)
                column_infos_in_one_db.append(column_infos_in_one_table)
            
            self.questions.append(data["question"])
            
            self.all_table_names.append(table_names_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_infos_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]

        return question, table_names_in_one_db, table_labels_in_one_db, column_infos_in_one_db, column_labels_in_one_db

class Text2SQLDataset(Dataset):
    def __init__(
        self,
        dir_: str,
        mode: str
    ):
        super(Text2SQLDataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            if self.mode in ["train", "eval"]:
                self.output_sequences.append(data["output_sequence"])
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")


    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_sequences[index], self.db_ids[index], self.all_tc_original[index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]



class MAtisDataset(Dataset):
    def __init__(
        self,
        dir_: str,
        data_lang: str,
        mode: str
    ):
        super(MAtisDataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            sql = data["mr"]["sql"]
            question = data["question"][data_lang]
            input_sequence = "Translate the following question into SQL:" + question
            self.input_sequences.append(input_sequence)

            if self.mode in ["train", "eval"]:
                self.output_sequences.append(sql)
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.input_sequences[index], self.output_sequences[index]
        elif self.mode == "test":
            return self.input_sequences[index]

class MSchema2QADataset(Dataset):
    def __init__(
        self,
        dir_: str,
        data_lang: str,
        mode: str
    ):
        super(MSchema2QADataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            query = data["mr"]["thingtalk"][data_lang]
            question = data["question"][data_lang]
            input_sequence = "Translate the following question into thingtalk QL: " + question
            self.input_sequences.append(input_sequence)

            if self.mode in ["train", "eval"]:
                self.output_sequences.append(query)
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.input_sequences[index], self.output_sequences[index]
        elif self.mode == "test":
            return self.input_sequences[index]

# For translate-train, with label projection
class TAPMSchema2QADataset(Dataset):
    def __init__(
        self,
        dir_: str,
        mode: str
    ):
        super(TAPMSchema2QADataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            query = data["mr"]
            question = data["question"]
            input_sequence = "Translate the following question into thingtalk QL: " + question
            self.input_sequences.append(input_sequence)

            if self.mode in ["train", "eval"]:
                self.output_sequences.append(query)
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.input_sequences[index], self.output_sequences[index]
        elif self.mode == "test":
            return self.input_sequences[index]

# For translate-train, without label projection
class TTMSchema2QADataset(Dataset):
    def __init__(
        self,
        dir_: str,
        mode: str
    ):
        super(TTMSchema2QADataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            query = data["mr"]["thingtalk"]["en"]
            question = data["translated_question"]
            input_sequence = "Translate the following question into thingtalk QL: " + question
            self.input_sequences.append(input_sequence)

            if self.mode in ["train", "eval"]:
                self.output_sequences.append(query)
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.input_sequences[index], self.output_sequences[index]
        elif self.mode == "test":
            return self.input_sequences[index]



class MultiMschema2QADataset(Dataset):
    def __init__(
        self,
        dir_: str,
        mode: str
    ):
        super(MSchema2QADataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        languages = dataset[0]["mr"]["thingtalk"].keys()
        for data in dataset:
            for data_lang in languages:
                query = data["mr"]["thingtalk"][data_lang]
                question = data["question"][data_lang]
                input_sequence = "Translate the following question into thingtalk QL: " + question
                self.input_sequences.append(input_sequence)

                if self.mode in ["train", "eval"]:
                    self.output_sequences.append(query)
                elif self.mode == "test":
                    pass
                else:
                    raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.input_sequences[index], self.output_sequences[index]
        elif self.mode == "test":
            return self.input_sequences[index]




class LanguagePredictionDataset(Dataset):
    def __init__(
        self,
        langs: List[str],
    ):
        super(LanguagePredictionDataset).__init__()
        
        lang_examples = []
        for idx, lang in enumerate(langs):
            # load dataset 
            lang_samples = load_dataset(f"deokhk/{lang}_wiki_sentences_1000000", split="train")["sentence"]
            lang_samples = lang_samples[:10000] # we only use the first 10000 samples for each language
            for sample in lang_samples:
                lang_examples.append(
                    {
                        "sentence": sample,
                        "label": idx
                    }
                )
        self.lang_examples = lang_examples

    
    def __len__(self):
        return len(self.lang_examples)
    
    def __getitem__(self, index):
        return self.lang_examples[index]["sentence"], self.lang_examples[index]["label"]

class ReconstructionDataset(Dataset):

    def __init__(self, 
                 langs: List[str],
                 tokenizer,
                 max_seq_length=512,
                 mask_rate=0.3,
    ):
        super(ReconstructionDataset).__init__()

        self.tokenizer = tokenizer 
        self.mask_rate = mask_rate
        self.max_seq_length = max_seq_length
        lang_examples = []
        for idx, lang in enumerate(langs):
            # load dataset 
            lang_samples = load_dataset(f"deokhk/{lang}_wiki_sentences_1000000", split="train")["sentence"]
            lang_samples = lang_samples[:30000] # we only use the first 30000 samples for each language. 50 epoch X 7000 =350K, 30K X 11 = 300K
            for sample in lang_samples:
                lang_examples.append(sample)

        self.lang_examples = lang_examples
        MASK_token_id = tokenizer.encode("<mask>")[0]
        self.MASK_token_id = MASK_token_id

        print(f"Generating datapoints...")
        self.features = self.get_features(self.lang_examples)
        print("Total datapoints:", len(self.features))

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


    def get_features(self, lang_samples):
        def masked_sequence(original_seq):
            masked_seq = original_seq[:]

            target_mask_len = int(len(masked_seq)*self.mask_rate)
            cur_mask_len = 0
            while cur_mask_len < target_mask_len:
                span_len = np.random.poisson(lam=3.5)
                cur_mask_len += span_len
                start_idx = np.random.choice(range(len(masked_seq)), size=1)[0]

                masked_seq = masked_seq[:start_idx] + [self.MASK_token_id] + masked_seq[start_idx+span_len:]
            return masked_seq
        
        features = []
        for sample in tqdm.tqdm(lang_samples):
            total_tokens = self.tokenizer.encode(sample)
            offset = 0
            while len(total_tokens[offset:]) > 0:
                labels = total_tokens[offset:offset+self.max_seq_length]
                input_ids = masked_sequence(labels)

                features.append(
                    {"input_text": self.tokenizer.decode(input_ids),
                    "label": self.tokenizer.decode(labels)}
                )
                offset = offset + self.max_seq_length


        return features


class Text2SQLWithReconDataset(Dataset):
    def __init__(
        self,
        text2sql_dataset,
        reconstruction_dataset
    ):
        super(Text2SQLWithReconDataset).__init__()
        self.text2sql_dataset = text2sql_dataset
        self.reconstruction_dataset = reconstruction_dataset

        self.recon_len = len(self.reconstruction_dataset)
    def __len__(self):
        return len(self.text2sql_dataset)

    def __getitem__(self, index):
        text2sql_input_seq = self.text2sql_dataset.input_sequences[index]
        text2sql_output_seq = self.text2sql_dataset.output_sequences[index]
        text2sql_db_id = self.text2sql_dataset.db_ids[index]
        text2sql_tc_original = self.text2sql_dataset.all_tc_original[index]
        
        # sample randomly from the reconstruction dataset
        idx = random.randint(0, self.recon_len-1)
        datapoint = self.reconstruction_dataset[idx]
        recon_input = datapoint["input_text"]
        recon_label = datapoint["label"]

        return text2sql_input_seq, text2sql_output_seq, text2sql_db_id, text2sql_tc_original, recon_input, recon_label


class Text2SQLWithLpAndReconDataset(Dataset):
    def __init__(
        self,
        text2sql_dataset,
        reconstruction_dataset,
        lp_dataset
    ):
        super(Text2SQLWithLpAndReconDataset).__init__()
        self.text2sql_dataset = text2sql_dataset
        self.reconstruction_dataset = reconstruction_dataset
        self.lp_dataset = lp_dataset

        self.recon_len = len(self.reconstruction_dataset)
        self.lp_len = len(self.lp_dataset)
    def __len__(self):
        return len(self.text2sql_dataset)

    def __getitem__(self, index):
        text2sql_input_seq = self.text2sql_dataset.input_sequences[index]
        text2sql_output_seq = self.text2sql_dataset.output_sequences[index]
        text2sql_db_id = self.text2sql_dataset.db_ids[index]
        text2sql_tc_original = self.text2sql_dataset.all_tc_original[index]
        
        # sample randomly from the reconstruction dataset
        idx = random.randint(0, self.recon_len-1)
        datapoint = self.reconstruction_dataset[idx]
        recon_input = datapoint["input_text"]
        recon_label = datapoint["label"]

        # sample randomly from the language prediction dataset
        lp_index = random.randint(0, self.lp_len-1)
        lp_input_seq, lp_label = self.lp_dataset[lp_index]

        return text2sql_input_seq, text2sql_output_seq, text2sql_db_id, text2sql_tc_original, lp_input_seq, lp_label, recon_input, recon_label,


class Text2SQLMultiPTDataset(Dataset):
    def __init__(
        self, 
        synthesized_dataset_paths: List[str]
    ):
        super(Text2SQLMultiPTDataset).__init__()
        
        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        for path in synthesized_dataset_paths:
            with open(path, 'r', encoding = 'utf-8') as f:
                dataset = json.load(f)
            
            for data in dataset:
                self.input_sequences.append(data["input_sequence"])
                self.output_sequences.append(data["output_sequence"])
    
    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return self.input_sequences[index], self.output_sequences[index]
            

class Mschema2QAMultiPTDataset(Dataset):
    def __init__(
        self, 
        synthesized_dataset_paths: List[str]
    ):
        super(Text2SQLMultiPTDataset).__init__()
        
        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        for path in synthesized_dataset_paths:
            with open(path, 'r', encoding = 'utf-8') as f:
                dataset = json.load(f)
            
            for data in dataset:
                query = data["mr"]["thingtalk"]["en"]
                question = data["generated_question"]

                input_sequence = "Translate the following question into thingtalk QL: " + question
                self.input_sequences.append(input_sequence)
                self.output_sequences.append(query)
    
    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return self.input_sequences[index], self.output_sequences[index]


class Mschema2QADatasetWithMultiPT(Dataset):
    def __init__(
        self,
        mschema2qa_dataset,
        multi_pt_dataset
    ):
        super(Mschema2QADatasetWithMultiPT).__init__()
        self.mschema2qa_dataset = mschema2qa_dataset
        self.multi_pt_dataset = multi_pt_dataset

        self.multi_pt_dataset_len = len(self.multi_pt_dataset)
    def __len__(self):
        return len(self.mschema2qa_dataset)
    
    def __getitem__(self, index):
        mschema2qa_input_seq = self.mschema2qa_dataset.input_sequences[index]
        mschema2qa_output_seq = self.mschema2qa_dataset.output_sequences[index]

        # sample randomly from the multi-pt dataset
        multi_pt_index = random.randint(0, self.multi_pt_dataset_len-1)
        multi_pt_input_seq, multi_pt_output_seq = self.multi_pt_dataset[multi_pt_index]

        return mschema2qa_input_seq, mschema2qa_output_seq, multi_pt_input_seq, multi_pt_output_seq

class Mschema2QADatasetWithTranslated(Dataset):
    def __init__(
        self,
        mschema2qa_dataset,
        tapm_dataset
    ):
        super(Mschema2QADatasetWithTranslated).__init__()
        self.mschema2qa_dataset = mschema2qa_dataset
        self.tapm_dataset = tapm_dataset

        self.tapm_dataset_len = len(self.tapm_dataset)
    def __len__(self):
        return len(self.mschema2qa_dataset)
    
    def __getitem__(self, index):
        mschema2qa_input_seq = self.mschema2qa_dataset.input_sequences[index]
        mschema2qa_output_seq = self.mschema2qa_dataset.output_sequences[index]

        # sample randomly from the multi-pt dataset
        idx = random.randint(0, self.tapm_dataset_len-1)
        tap_input_seq, tap_output_seq = self.tapm_dataset[idx]

        return mschema2qa_input_seq, mschema2qa_output_seq, tap_input_seq, tap_output_seq



class Text2SQLDatasetWithMultiPT(Dataset):
    def __init__(
        self,
        text2sql_dataset,
        multi_pt_dataset
    ):
        super(Text2SQLDatasetWithMultiPT).__init__()
        self.text2sql_dataset = text2sql_dataset
        self.multi_pt_dataset = multi_pt_dataset

        self.multi_pt_dataset_len = len(self.multi_pt_dataset)

    def __len__(self):
        return len(self.text2sql_dataset)

    def __getitem__(self, index):
        text2sql_input_seq = self.text2sql_dataset.input_sequences[index]
        text2sql_output_seq = self.text2sql_dataset.output_sequences[index]
        text2sql_db_id = self.text2sql_dataset.db_ids[index]
        text2sql_tc_original = self.text2sql_dataset.all_tc_original[index]

        # sample randomly from the multi-pt dataset
        multi_pt_index = random.randint(0, self.multi_pt_dataset_len-1)
        multi_pt_input_seq, multi_pt_output_seq = self.multi_pt_dataset[multi_pt_index]

        return text2sql_input_seq, text2sql_output_seq, text2sql_db_id, text2sql_tc_original, multi_pt_input_seq, multi_pt_output_seq


class Text2SQLDatasetWithTranslated(Dataset):
    def __init__(
        self,
        text2sql_dataset,
        translated_text2sql_dataset
    ):
        super(Text2SQLDatasetWithTranslated).__init__()
        self.text2sql_dataset = text2sql_dataset
        self.translated_text2sql_dataset = translated_text2sql_dataset 

        self.translated_text2sql_dataset_len = len(self.translated_text2sql_dataset)

    def __len__(self):
        return len(self.text2sql_dataset)

    def __getitem__(self, index):
        text2sql_input_seq = self.text2sql_dataset.input_sequences[index]
        text2sql_output_seq = self.text2sql_dataset.output_sequences[index]
        text2sql_db_id = self.text2sql_dataset.db_ids[index]
        text2sql_tc_original = self.text2sql_dataset.all_tc_original[index]

        # sample randomly from the translated text2sql dataset
        idx = random.randint(0, self.translated_text2sql_dataset_len-1)
        tt_input_seq, tt_output_seq, tt_db_id, tt_tc_original = self.translated_text2sql_dataset[idx]

        return text2sql_input_seq, text2sql_output_seq, text2sql_db_id, text2sql_tc_original, tt_input_seq, tt_output_seq, tt_db_id, tt_tc_original



class Text2SQLWithLPDataset(Dataset):
    def __init__(
        self,
        text2sql_dataset,
        lp_dataset
    ):
        super(Text2SQLWithLPDataset).__init__()
        self.text2sql_dataset = text2sql_dataset
        self.lp_dataset = lp_dataset

        self.lp_dataset_len = len(self.lp_dataset)
    def __len__(self):
        return len(self.text2sql_dataset)

    def __getitem__(self, index):
        text2sql_input_seq = self.text2sql_dataset.input_sequences[index]
        text2sql_output_seq = self.text2sql_dataset.output_sequences[index]
        text2sql_db_id = self.text2sql_dataset.db_ids[index]
        text2sql_tc_original = self.text2sql_dataset.all_tc_original[index]
        
        # sample randomly from the language prediction dataset
        lp_index = random.randint(0, self.lp_dataset_len-1)
        lp_input_seq, lp_label = self.lp_dataset[lp_index]

        return text2sql_input_seq, text2sql_output_seq, text2sql_db_id, text2sql_tc_original, lp_input_seq, lp_label






class Mschema2QAWithLPDataset(Dataset):
    def __init__(
        self,
        mschema2qa_dataset,
        lp_dataset
    ):
        super(Mschema2QAWithLPDataset).__init__()
        self.mschema2qa_dataset = mschema2qa_dataset
        self.lp_dataset = lp_dataset

        self.lp_dataset_len = len(self.lp_dataset)
    def __len__(self):
        return len(self.mschema2qa_dataset)

    def __getitem__(self, index):
        mschema2qa_input_seq = self.mschema2qa_dataset.input_sequences[index]
        mschema2qa_output_seq = self.mschema2qa_dataset.output_sequences[index]
        
        # sample randomly from the language prediction dataset
        lp_index = random.randint(0, self.lp_dataset_len-1)
        lp_input_seq, lp_label = self.lp_dataset[lp_index]

        return mschema2qa_input_seq, mschema2qa_output_seq, lp_input_seq, lp_label


class Mschema2QAWithReconDataset(Dataset):
    def __init__(
        self,
        mschema2qa_dataset,
        reconstruction_dataset
    ):
        super(Mschema2QAWithLPDataset).__init__()
        self.mschema2qa_dataset = mschema2qa_dataset
        self.reconstruction_dataset = reconstruction_dataset

        self.recon_len = len(self.reconstruction_dataset)
    def __len__(self):
        return len(self.mschema2qa_dataset)

    def __getitem__(self, index):
        mschema2qa_input_seq = self.mschema2qa_dataset.input_sequences[index]
        mschema2qa_output_seq = self.mschema2qa_dataset.output_sequences[index]
        
        # sample randomly from the reconstruction dataset
        idx = random.randint(0, self.recon_len-1)
        datapoint = self.reconstruction_dataset[idx]
        recon_input = datapoint["input_text"]
        recon_label = datapoint["label"]

        return mschema2qa_input_seq, mschema2qa_output_seq, recon_input, recon_label

class Mschema2QAWithLpAndReconDataset(Dataset):
    def __init__(
        self,
        mschema2qa_dataset,
        reconstruction_dataset,
        lp_dataset
    ):
        super(Mschema2QAWithLpAndReconDataset).__init__()
        self.mschema2qa_dataset = mschema2qa_dataset
        self.reconstruction_dataset = reconstruction_dataset
        self.lp_dataset = lp_dataset

        self.recon_len = len(self.reconstruction_dataset)
        self.lp_len = len(self.lp_dataset)
    def __len__(self):
        return len(self.mschema2qa_dataset)

    def __getitem__(self, index):
        mschema2qa_input_seq = self.mschema2qa_dataset.input_sequences[index]
        mschema2qa_output_seq = self.mschema2qa_dataset.output_sequences[index]
        
        # sample randomly from the reconstruction dataset
        idx = random.randint(0, self.recon_len-1)
        datapoint = self.reconstruction_dataset[idx]
        recon_input = datapoint["input_text"]
        recon_label = datapoint["label"]

        # sample randomly from the language prediction dataset
        lp_index = random.randint(0, self.lp_len-1)
        lp_input_seq, lp_label = self.lp_dataset[lp_index]

        return mschema2qa_input_seq, mschema2qa_output_seq, lp_input_seq, lp_label, recon_input, recon_label



class SynMSchema2QADataset(Dataset):
    def __init__(
        self,
        dir_: str,
        data_lang: str,
        mode: str
    ):
        super(SynMSchema2QADataset).__init__()
        
        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            query = data["mr"]["thingtalk"][data_lang]
            question = data["generated_question"]
            input_sequence = "Translate the following question into thingtalk QL: " + question
            self.input_sequences.append(input_sequence)

            if self.mode in ["train", "eval"]:
                self.output_sequences.append(query)
            elif self.mode == "test":
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode in ["train", "eval"]:
            return self.input_sequences[index], self.output_sequences[index]
        elif self.mode == "test":
            return self.input_sequences[index]




def load_multitask_dataset(mt_path, sp_path, vp_path, K, mode="train"):
    """
    We use a examples-proportional mixing strategy to combine the three datasets.
    """

    save_path = os.path.dirname(mt_path) + f"/interleaved_dataset_{K}"
    if os.path.exists(save_path):
        return hf_dataset.load_from_disk(save_path)
    else:
        mt_dataset = hf_dataset.from_json(mt_path)
        sp_dataset = hf_dataset.from_json(sp_path)
        vp_dataset = hf_dataset.from_json(vp_path)

        mt_count = len(mt_dataset)
        sp_count = len(sp_dataset)
        vp_count = len(vp_dataset)
        scaled_counts = [min(mt_count, K), min(sp_count, K), min(vp_count, K)]

        proportion = [count/sum(scaled_counts) for count in scaled_counts]
        dataset = interleave_datasets([mt_dataset, sp_dataset, vp_dataset], probabilities=proportion)

        if mode == "train":
            # We drop features that are not used in training.
            dataset = dataset.remove_columns(["mentioned_schema_items", "mentioned_values"])
        dataset.save_to_disk(save_path)
        
    return dataset 

    
def load_multitask_dataset_legacy(mt_path, sp_path, vp_path, K, mode="train"):
    """
    We use a examples-proportional mixing strategy to combine the three datasets.
    """
    mt_dataset = hf_dataset.from_json(mt_path)
    sp_dataset = hf_dataset.from_json(sp_path)
    vp_dataset = hf_dataset.from_json(vp_path)

    mt_count = len(mt_dataset)
    sp_count = len(sp_dataset)
    vp_count = len(vp_dataset)
    scaled_counts = [min(mt_count, K), min(sp_count, K), min(vp_count, K)]

    proportion = [count/sum(scaled_counts) for count in scaled_counts]
    dataset = interleave_datasets([mt_dataset, sp_dataset, vp_dataset], probabilities=proportion)

    if mode == "train":
        # We drop features that are not used in training.
        dataset = dataset.remove_columns(["mentioned_schema_items", "mentioned_values"])
        
    return dataset 