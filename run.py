import os
import json
import tqdm
import random
import numpy as np
import argparse
from safetensors import safe_open
from safetensors.torch import save_file, load_model

import torch
import evaluate 
import glob
import os 

from safetensors.torch import safe_open
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import MT5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator, AutoTokenizer, EarlyStoppingCallback
from models.mT5 import MT5ForConditionalGenerationWithAdapter, MT5ForCondGenWithLangAgnosticEncoderWithAdapter, MT5Config
from utils.extract_sentence_from_wiki import get_files

os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = None
lang_dict = None
task_dict = None
MASK_token_id = None
MIN_SAMPLE_NUM = 162870
metric = evaluate.load("sacrebleu")
detect_unused_params = False 



def parse_argument():
    global args, lang_dict, task_dict
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--mask_rate", default=0.3, type=float, help="masking rate for denoising pretraining")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--training_steps", type=int, default=100000)
    parser.add_argument("--eval_steps", type=int, default=5000)

    parser.add_argument("--do_adapter_train", action="store_true")
    parser.add_argument("--do_task_finetune", action="store_true")
    parser.add_argument("--do_predict", action="store_true", help="Whether to predict at the end of training")
    parser.add_argument("--do_synthesize", action="store_true", help="Whether to synthesize question with trained models")

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--exp_tag", type=str, default="test")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--adapter_training_checkpoint", type=str) 
    parser.add_argument("--pretrained_adapter_dir", type=str, default=None)

    parser.add_argument("--model_name_or_path", type=str, default=None, help="pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=1000000, help="maximum number of sentences to use")


    parser.add_argument("--local_rank", type=int, default=-1)
    ### Adapter Config
    parser.add_argument("--adapter_types", type=str, default="decoder-lang", help="example) encoder-task,decoder-lang")
    parser.add_argument("--langs", type=str, default=None, help="example) en,ko. Available languages in the pretrained adapter.")
    parser.add_argument("--partial_langs", type=str, default=None, help="example) en,am")
    parser.add_argument("--tasks", type=str, default=None, help="example) qa,sum")
    parser.add_argument("--task_lang", type=str, default=None, help="Which language adapter to use. Must belong to langs.")
    parser.add_argument("--adapter_layer_norm", type=str, default="false")
    parser.add_argument("--adapter_reduction_factor", type=int, default=2)
    parser.add_argument("--adapter_hidden_act", type=str, default="gelu")
    parser.add_argument("--frozen_list", type=str, default=None, help="all,emb,enc_ln,enc_attn,enc_ffn,dec_ln,dec_attn,dec_crossattn,dec_ffn")
    parser.add_argument("--freeze_option", type=str, choices=["encoder_only", "except_dec_cross_attention", "decoder_only", "decoder_only_with_dec_ffn_and_lm", "enc_final_layer_and_decoder", "enc_and_dec_ffn", "enc_and_dec_ffn_attn", "only_emb", "only_emb_and_ln", "no_freeze", "all"], default="dec_cross_attention_only")


    ### Text2MR dataset config 
    parser.add_argument("--train_file", type=str, default="data/preprocessed_data/train_spider_seq2seq_english.json")
    parser.add_argument("--valid_file", type=str, default="data/preprocessed_data/dev_spider_seq2seq_english.json")
    parser.add_argument("--without_schema", action="store_true", help="whether to not to use schema information")
    parser.add_argument("--inference_data_file", type=str, default="/home1/deokhk_1/research/ZX-seq2seq/data/preprocessed_data/dev_spider_seq2seq_english.json")
    parser.add_argument("--dataset_type", type=str, default="spider", choices=["spider", "mschema2qa"])
    parser.add_argument("--dataset_lang", type=str, default="en", choices=["en", "ko"])

    ### Generation config
    parser.add_argument("--generation_max_length", type=int, default=64)
    parser.add_argument("--generation_num_beams", type=int, default=8)
    parser.add_argument("--contrastive_generation", action="store_true", help="When set, generate with contrastive decoding with top_k=4 / penalty_alpha=0.6")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    
    ### language identity
    parser.add_argument("--without_language_identity", action='store_true', help="whether to remove language identity")
    parser.add_argument("--removal_type", type=str, default=None, choices=["mean_zero", "mean_eng", "batch_norm", "proj_eng", "proj_remove"])
    parser.add_argument("--language_identity_path", type=str, help="Path to the language mean and subspace projection matrix.", default="/home/deokhk/research/XLang-NL2SQL/language_identity")
    parser.add_argument("--remove_prob", type=float, default=1.0, help="Probability of removing language identity.")
    
    ### Other arguments
    parser.add_argument("--result_prefix", type=str, default="", help="Prefix for the synthesized file")
    args = parser.parse_args()

    if args.adapter_layer_norm == "true":
        args.adapter_layer_norm = True
    elif args.adapter_layer_norm == "false":
        args.adapter_layer_norm = False
    else:
        assert False

class AdapterConfig:
    def __init__(self, args):
        self.adapters = {"encoder": {"lang": False, "task": False},
                         "decoder": {"lang": False, "task": False}}
        for i in ["encoder", "decoder"]:
            for j in ["lang", "task"]:
                if f"{i}-{j}" in args.adapter_types:
                    print(f"### Use {i}-{j} ###")
                    self.adapters[i][j] = True

        self.lang_dict = dict(enumerate(args.langs.split(","))) if args.langs is not None else dict()
        self.task_dict = dict(enumerate(args.tasks.split(","))) if args.tasks is not None else dict()
        self.adapter_layer_norm = args.adapter_layer_norm
        self.adapter_reduction_factor = args.adapter_reduction_factor
        self.adapter_hidden_act = args.adapter_hidden_act

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DenoisingDataset(torch.utils.data.Dataset):
    
    def __init__(self, lang_samples, tokenizer, mode):
        super().__init__()
        self.tokenizer = tokenizer
        self.mode = mode

        print(f"Generate features...")
        self.features, self.unique_id_to_gold = self.get_features(lang_samples)
        print("total feature num:", len(self.features))

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def get_features(self, lang_samples):
        def masked_sequence(original_seq):
            masked_seq = original_seq[:]

            target_mask_len = int(len(masked_seq)*args.mask_rate)
            cur_mask_len = 0
            while cur_mask_len < target_mask_len:
                span_len = np.random.poisson(lam=3.5)
                cur_mask_len += span_len
                start_idx = np.random.choice(range(len(masked_seq)), size=1)[0]

                masked_seq = masked_seq[:start_idx] + [MASK_token_id] + masked_seq[start_idx+span_len:]
            return masked_seq

        unique_id = 0
        features = []
        unique_id_to_gold = dict()
        for lang, samples in lang_samples.items():
            lang_id = [k for k,v in lang_dict.items() if v == lang][0]
            for sample in tqdm.tqdm(samples):
                total_tokens = self.tokenizer.encode(sample)
                offset = 0
                while len(total_tokens[offset:]) > 0:
                    labels = total_tokens[offset:offset+args.max_seq_length]
                    input_ids = masked_sequence(labels)
                    attention_mask = [1]*len(input_ids)

                    unique_id_to_gold[unique_id] = {"input_text": self.tokenizer.decode(input_ids),
                                                    "label": self.tokenizer.decode(labels),
                                                    "lang": lang}

                    while len(input_ids) < args.max_seq_length:
                        input_ids.append(self.tokenizer.pad_token_id)
                        attention_mask.append(0)
                    while len(labels) < args.max_seq_length:
                        # Ignore padding tokens when computing loss
                        labels.append(-100)

                    assert len(input_ids) == args.max_seq_length, f"{len(input_ids)}"

                    feature = {"unique_id": unique_id,
                               "input_ids": input_ids,
                               "attention_mask": attention_mask,
                               "labels": labels,
                               "in_lang_ids": lang_id,
                               "out_lang_ids": lang_id}
                    features.append(feature)

                    offset = offset + args.max_seq_length
                    unique_id += 1

        return features, unique_id_to_gold




class SQL2TextDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, tokenizer, lang, task=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.without_schema = self.args.without_schema
        self.lang_id = [k for k,v in lang_dict.items() if v==lang][0]
        if task is not None:
            self.task_id = [k for k,v in task_dict.items() if v==task][0]

        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Generating features...")
        
        self.features = self.get_features(self.data)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def get_features(self, data):

        sql2text_input_seqs = []
        sql2text_output_seqs = []
        features = []

        input_seqs = [datapoint["input_sequence"] for datapoint in data]
        output_seqs = [datapoint["output_sequence"] for datapoint in data]

        for input_seq, output_seq in zip(input_seqs, output_seqs):
            sql2text_input_seq, sql2text_output_seq = self.preprocess(input_seq, output_seq)
            sql2text_input_seqs.append(sql2text_input_seq)
            sql2text_output_seqs.append(sql2text_output_seq)

        tokenized_inputs = self.tokenizer(
            sql2text_input_seqs, 
            padding="max_length", 
            return_tensors="pt",
            max_length=self.args.max_seq_length,
            truncation=True
        )

        tokenized_outputs = self.tokenizer(
            sql2text_output_seqs, 
            padding="max_length", 
            return_tensors="pt",
            max_length=256,
            truncation=True
        )

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]

        decoder_labels = tokenized_outputs["input_ids"]
        decoder_labels[decoder_labels == self.tokenizer.pad_token_id] = -100

        for enc_id, enc_attn_mask, dec_label in zip(
            encoder_input_ids, encoder_input_attention_mask, decoder_labels
        ):
            if task_dict is not None: 
                features.append(
                    {
                        "input_ids": enc_id,
                        "attention_mask": enc_attn_mask,
                        "labels": dec_label,
                        "in_lang_ids": self.lang_id,
                        "out_lang_ids": self.lang_id,
                        "task_ids": self.task_id
                    }
                )
            else:
                features.append(
                    {
                        "input_ids": enc_id,
                        "attention_mask": enc_attn_mask,
                        "labels": dec_label,
                        "in_lang_ids": self.lang_id,
                        "out_lang_ids": self.lang_id
                    }
                )

        return features

    def preprocess(self, input_seq, output_seq):
        # Take input_seq from text2sql dataset and output_seq from text2sql dataset
        # Preprocess the input_seq and output_seq to make it suitable for sql2text task
        
        input_seq = input_seq.split("Translate the following sequence into SQL:") # Remove prompt 
        input_seq = input_seq[1].strip() # Remove leading and trailing whitespaces

        input_seq_splitted = input_seq.split("|") # Split into question and schema
        question = input_seq_splitted[0].strip()
        schema = ("|".join(input_seq_splitted[1:])).strip()

        sql = output_seq 

        sql2text_input_seq = "Translate the following SQL to question:" + sql + " | "
        if not self.without_schema:
            sql2text_input_seq += schema

        sql2text_output_seq = question 
        return (sql2text_input_seq, sql2text_output_seq)


class Mr2TextMSchema2QADataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, tokenizer, adaper_lang, dataset_lang="en", task=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.lang_id = [k for k,v in lang_dict.items() if v==adaper_lang][0]
        self.dataset_lang = dataset_lang
        if task is not None:
            self.task_id = [k for k,v in task_dict.items() if v==task][0]

        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Generating features...")
        
        self.features = self.get_features(self.data)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def get_features(self, data):

        mr2text_input_seqs = []
        mr2text_output_seqs = []
        features = []

        for datapoint in data:
            question = datapoint["question"][self.dataset_lang]
            mr = datapoint["mr"]["thingtalk"][self.dataset_lang]
            mr2text_input_seq = "Translate the following MR to question: " + mr
            mr2text_input_seqs.append(mr2text_input_seq)
            mr2text_output_seqs.append(question)

        tokenized_inputs = self.tokenizer(
            mr2text_input_seqs, 
            padding="max_length", 
            return_tensors="pt",
            max_length=self.args.max_seq_length,
            truncation=True
        )

        tokenized_outputs = self.tokenizer(
            mr2text_output_seqs, 
            padding="max_length", 
            return_tensors="pt",
            max_length=256,
            truncation=True
        )

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]

        decoder_labels = tokenized_outputs["input_ids"]
        decoder_labels[decoder_labels == self.tokenizer.pad_token_id] = -100

        for enc_id, enc_attn_mask, dec_label in zip(
            encoder_input_ids, encoder_input_attention_mask, decoder_labels
        ):
            if task_dict is not None: 
                features.append(
                    {
                        "input_ids": enc_id,
                        "attention_mask": enc_attn_mask,
                        "labels": dec_label,
                        "in_lang_ids": self.lang_id,
                        "out_lang_ids": self.lang_id,
                        "task_ids": self.task_id
                    }
                )
            else:
                features.append(
                    {
                        "input_ids": enc_id,
                        "attention_mask": enc_attn_mask,
                        "labels": dec_label,
                        "in_lang_ids": self.lang_id,
                        "out_lang_ids": self.lang_id
                    }
                )

        return features




def compute_metrics(eval_preds):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


if __name__ == "__main__":
    parse_argument()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.exp_tag)
    os.makedirs(model_dir, exist_ok=True)

    config = os.path.join(model_dir, "config.json")
    if args.use_checkpoint or args.adapter_training_checkpoint or args.pretrained_adapter_dir:
        if args.pretrained_adapter_dir:
            config = os.path.join(os.getcwd(), f"{args.pretrained_adapter_dir}/config.json")
        if args.adapter_training_checkpoint and args.do_adapter_train:
            # When training adapters only, we allow to use checkpoint from previous training
            config = os.path.join("/".join(args.adapter_training_checkpoint.split("/")[:-1]), "config.json")
        with open(config, "r") as fin:
            arg_dict = json.load(fin)
            args.seed = arg_dict["seed"]
            if args.model_name_or_path is None:
                if "model_name" in arg_dict:
                    args.model_name_or_path = arg_dict["model_name"]
                elif "model_name_or_path" in arg_dict:
                    args.model_name_or_path = arg_dict["model_name_or_path"]
                else:
                    assert KeyError, "Config file does not have model_name_or_path!"

            args.max_seq_length = arg_dict["max_seq_length"]

            assert args.langs == arg_dict["langs"], "Language setting is different from the previous training."
            assert arg_dict["adapter_types"] in args.adapter_types, "Pretrained adapter type is not included in the current adapter types."
            if arg_dict["tasks"] is None and args.tasks is not None:
                print(f"Adding a new task adapter for {args.tasks}during training")
            if arg_dict["tasks"] is not None and args.tasks is not None:
                assert arg_dict["tasks"] in args.tasks, "Task setting is different from the previous training."
            args.adapter_layer_norm = arg_dict["adapter_layer_norm"]
            args.adapter_reduction_factor = arg_dict["adapter_reduction_factor"]
            args.adapter_hidden_act = arg_dict["adapter_hidden_act"]
    
    print(args)
    
    if args.langs is not None:
        lang_dict = dict(enumerate(args.langs.split(",")))
    if args.tasks is not None:
        task_dict = dict(enumerate(args.tasks.split(",")))
    
    if not args.do_adapter_train:
        assert args.task_lang in lang_dict.values()

    if not args.use_checkpoint and not args.adapter_training_checkpoint:
        config = os.path.join(model_dir, "config.json")
        with open(config, "w") as fout:
            json.dump(vars(args), fout, indent=1)
    adapter_config = AdapterConfig(args)

    seed_everything(args.seed)
    device = torch.device("cuda")
    
    tokenizer = MT5Tokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_dict = {'additional_special_tokens': ["<mask>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    MASK_token_id = tokenizer.encode("<mask>")[0]


    if args.partial_langs is not None:
        task_lang_id = [k for k,v in lang_dict.items() if v == args.partial_langs][0]
    else:
        task_lang_id = [k for k,v in lang_dict.items() if v == args.task_lang][0]
    if args.without_language_identity:

        assert args.langs.startswith("en"), "It should start with en, in order to keep an index for english as 0."
        detect_unused_params = True 

        model = MT5ForCondGenWithLangAgnosticEncoderWithAdapter.from_pretrained(args.model_name_or_path)
        # Load language means and subspaces, respectively.
        # There's no problem in this code. We only load what we need!
        
        files = get_files(args.language_identity_path)
        sorted_dict_list = sorted(lang_dict.items())
        lang_means = []
        lang_subspaces = []
        for (lang_id, lang) in sorted_dict_list:
            language_folders = os.listdir(args.language_identity_path)
            for folder_name in language_folders:
                if folder_name.startswith(lang):
                    mean_file_path = None 
                    subspace_file_path = None
                    language_path = os.path.join(args.language_identity_path, folder_name)
                    files = os.listdir(language_path)
                    for x in files:
                        if "mean" in x:
                            mean_file_path = os.path.join(language_path, x)
                        elif "subspace" in x:
                            subspace_file_path = os.path.join(language_path, x)
                    lang_mean = np.load(mean_file_path)
                    lang_means.append(lang_mean)

                    if args.removal_type == "proj_remove":
                        lang_subspace = np.load(subspace_file_path)
                        lang_subspaces.append(lang_subspace)

        lang_means = torch.Tensor(np.stack(lang_means, axis=0))
        if args.removal_type == "proj_remove":
            source_lang_subspace = torch.from_numpy(lang_subspaces[0])
            source_lang_subspace = torch.stack([source_lang_subspace], axis=0) # add batc dimension

            target_lang_subspace = torch.from_numpy(lang_subspaces[task_lang_id])
            target_lang_subspace = torch.stack([target_lang_subspace], axis=0) # add batc dimension
            model.add_language_identity(lang_means, source_lang_subspace, target_lang_subspace, args.removal_type, args.remove_prob, args.do_synthesize)
        else:
            model.add_language_identity(lang_means, None, None, args.removal_type, args.remove_prob, args.do_synthesize)
    else:
        detect_unused_params = True
        model = MT5ForConditionalGenerationWithAdapter.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    if args.do_adapter_train:
        ### Load Dataset
        samples = {"train": dict(), "valid": dict()}
        if args.partial_langs is not None:
            for lang in args.partial_langs.split(","):
                train_samples = load_dataset(f"deokhk/{lang}_wiki_sentences_1000000", split="train")["sentence"] 
                valid_samples = load_dataset(f"deokhk/{lang}_wiki_sentences_1000000", split="dev")["sentence"]

                train_samples = train_samples[:args.max_samples]
                samples["train"][lang] = train_samples
                samples["valid"][lang] = valid_samples
                print(f"lang:{lang}|#train samples:{len(train_samples)}")
                print(f"lang:{lang}|#valid samples:{len(valid_samples)}")

        print("Generate train features...")
        train_dataset = DenoisingDataset(samples["train"], tokenizer, "train")

        print("Generate valid features...")
        valid_dataset = DenoisingDataset(samples["valid"], tokenizer, "valid")

        model.config.adapter_hidden_act = adapter_config.adapter_hidden_act
        model.setup_adapter(adapter_config)
        model.freeze_components(frozen_list=["all"])
        model = model.to(device)

        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Calculate the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'Total parameters: {total_params}')
        print(f'Trainable parameters: {trainable_params}')        

        training_args = Seq2SeqTrainingArguments(
            output_dir=model_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_strategy="steps", 
            evaluation_strategy="steps", 
            max_steps=args.training_steps,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            logging_steps=50, 
            seed=args.seed,
            dataloader_num_workers= int(4*torch.cuda.device_count()),
            report_to="wandb",
            run_name=args.exp_tag,
            load_best_model_at_end=True,
            save_total_limit=2,
            ddp_find_unused_parameters=detect_unused_params,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=default_data_collator,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train(resume_from_checkpoint=args.use_checkpoint)

        # Save best model to checkpoint

        best_checkpoint_path = os.path.join(model_dir, "best_checkpoint")
        trainer.save_model(best_checkpoint_path)

        print(f"Training has been done. Save best checkpoint at: {best_checkpoint_path}")

    if args.do_task_finetune:
        assert args.task_lang in lang_dict.values()
        ### Load Dataset
        print(f"Using {args.dataset_type} dataset.")
        print("Generating train features...")
        train_file = args.train_file
        if args.dataset_type == "spider":
            train_dataset = SQL2TextDataset(args, train_file, tokenizer, args.task_lang, args.tasks)
        elif args.dataset_type == "mschema2qa":
            train_dataset = Mr2TextMSchema2QADataset(args, train_file, tokenizer, args.task_lang, args.dataset_lang, args.tasks)
            
        print("Generating valid features...")
        valid_file = args.valid_file
        if args.dataset_type == "spider":
            valid_dataset = SQL2TextDataset(args, valid_file, tokenizer, args.task_lang, args.tasks)
        elif args.dataset_type == "mschema2qa":
            valid_dataset = Mr2TextMSchema2QADataset(args, valid_file, tokenizer, args.task_lang, args.dataset_lang, args.tasks)

        assert args.pretrained_adapter_dir is not None
        pretrained_adapter_dir = os.path.join(os.getcwd(), args.pretrained_adapter_dir)
        adapter_checkpoint_file = [f for f in os.listdir(pretrained_adapter_dir) if f.endswith("safetensors")][0]
        adapter_checkpoints = {}
        with safe_open(os.path.join(pretrained_adapter_dir, adapter_checkpoint_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                adapter_checkpoints[key] = f.get_tensor(key)

        model.config.adapter_hidden_act = adapter_config.adapter_hidden_act
        model.setup_adapter(adapter_config)

        print("Loaded Adapter Checkpoints")
        model.load_state_dict(adapter_checkpoints, strict=False)


        
        if args.freeze_option == "except_dec_cross_attention":
            model.freeze_components(frozen_list=["emb", "enc_attn", "enc_ffn", "enc_final_lm", "dec_attn", "dec_ffn", "dec_final", "lm_head"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "encoder_only":
            model.freeze_components(frozen_list=["emb", "enc_attn", "enc_ffn", "enc_final_lm", "enc_lm"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "decoder_only":
            model.freeze_components(frozen_list=["emb", "dec_attn", "dec_crossattn", "dec_final", "lm_head"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "decoder_only_with_dec_ffn_and_lm":
            model.freeze_components(frozen_list=["emb", "dec_attn", "dec_ffn", "dec_crossattn", "dec_final", "dec_lm", "lm_head"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "enc_final_layer_and_decoder":
            model.freeze_components(frozen_list=["emb",  "enc_final_lm", "enc_final_block", "dec_attn","dec_crossattn", "dec_final", "lm_head"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "enc_and_dec_ffn":
            model.freeze_components(frozen_list=["emb", "enc_lm", "enc_final_lm", "dec_ffn", "dec_final", "dec_lm", "lm_head"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "enc_and_dec_ffn_attn":
            model.freeze_components(frozen_list=["emb", "enc_lm", "enc_final_lm", "dec_attn", "dec_ffn", "dec_final", "dec_lm", "lm_head"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "only_emb":
            model.freeze_components(frozen_list=["emb"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "only_emb_and_ln":
            model.freeze_components(frozen_list=["emb", "enc_lm", "enc_final_lm", "dec_lm", "dec_final"], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "no_freeze":
            model.freeze_components(frozen_list=[], freeze_lang=True, freeze_task=False)
        elif args.freeze_option == "all":
            model.freeze_components(frozen_list=["all"], freeze_lang=True, freeze_task=False)

        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Calculate the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'Total parameters: {total_params}')
        print(f'Trainable parameters: {trainable_params}')        
        model = model.to(device)

        finetuning_args = Seq2SeqTrainingArguments(
            output_dir=model_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_strategy="epoch", 
            evaluation_strategy="epoch", 
            seed=args.seed,
            logging_steps=10,
            dataloader_num_workers= int(4*torch.cuda.device_count()),
            save_total_limit=3,
            predict_with_generate=True,
            metric_for_best_model="bleu",
            run_name=args.exp_tag,
            ddp_find_unused_parameters=detect_unused_params
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=finetuning_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=default_data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if finetuning_args.predict_with_generate else None,
        )

        trainer.train(resume_from_checkpoint=args.use_checkpoint)

    if args.do_synthesize:
        assert args.task_lang in lang_dict.values()
        if args.without_language_identity:
            assert args.remove_prob == 1.0, "When predicting without language identity, remove_prob should be 1.0 for reproducible results."
        ### Load Dataset
        print("Generating inference features...")
        inference_file = args.inference_data_file
        if args.dataset_type == "spider":
            dataset = SQL2TextDataset(args, inference_file, tokenizer, args.task_lang, args.tasks)
        elif args.dataset_type == "mschema2qa":
            dataset = Mr2TextMSchema2QADataset(args, inference_file, tokenizer, args.task_lang, args.dataset_lang, args.tasks)

        # Load adapter checkpoint 

        assert args.pretrained_adapter_dir is not None
        pretrained_adapter_dir = os.path.join(os.getcwd(), args.pretrained_adapter_dir)
        adapter_checkpoint_file = [f for f in os.listdir(pretrained_adapter_dir) if f.endswith("safetensors")][0]
        adapter_checkpoints = {}
        with safe_open(os.path.join(pretrained_adapter_dir, adapter_checkpoint_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                adapter_checkpoints[key] = f.get_tensor(key)

        model.config.adapter_hidden_act = adapter_config.adapter_hidden_act
        model.setup_adapter(adapter_config)

        print(f"Loaded Adapter Checkpoints from {args.pretrained_adapter_dir}")
        model.load_state_dict(adapter_checkpoints, strict=False)
        model.to(device)

        inference_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.valid_batch_size,
            collate_fn=default_data_collator,
            num_workers= int(4*torch.cuda.device_count()),
            shuffle=False
        )
        # For each batch, generate predictions
        predictions = []
        original_questions = []
        for batch in tqdm.tqdm(inference_dataloader, desc="Synthesizing questions.."):
            batch = {k: v.to(device) for k, v in batch.items()}
            btask_ids=batch["task_ids"] if "task_ids" in batch else None
            with torch.no_grad():
                if args.contrastive_generation:
                    outputs = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        in_lang_ids=batch["in_lang_ids"],
                        out_lang_ids=batch["out_lang_ids"],
                        task_ids=btask_ids,
                        max_length=args.generation_max_length,
                        penalty_alpha=0.6,
                        top_k=4
                    )
                else:
                    outputs = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        in_lang_ids=batch["in_lang_ids"],
                        out_lang_ids=batch["out_lang_ids"],
                        task_ids=btask_ids,
                        max_length=args.generation_max_length,
                        num_beams=args.generation_num_beams,
                        repetition_penalty=args.repetition_penalty,
                    )
                outputs = outputs.to("cpu")
                outputs = np.where(outputs != -100, outputs, tokenizer.pad_token_id)
                outputs = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                outputs = [pred.strip() for pred in outputs]
                predictions.extend(outputs)

                batch["labels"] = batch["labels"].to("cpu")
                labels = np.where(batch["labels"] != -100, batch["labels"], tokenizer.pad_token_id)
                batch_orig_question = tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                batch_orig_question = [orig_q.strip() for orig_q in batch_orig_question]
                original_questions.extend(batch_orig_question)

        original_dataset = dataset.data

        # Save 
        prefix = args.result_prefix
        save_file_name = f"generated_predictions_{args.task_lang}_beam_{args.generation_num_beams}"
        if args.contrastive_generation:
            save_file_name = f"generated_predictions_{args.task_lang}_contrastive"
        
        if_name = os.path.basename(args.inference_data_file).split(".")[0]
        suffix = f"_from_{if_name}"
        save_file_path = prefix + save_file_name + suffix + ".json"

        save_path = os.path.join(args.model_name_or_path, save_file_path)
        outputs = [{"generated_question": pred, "original_question": orig_q} for pred, orig_q in zip(predictions, original_questions)]
        
        # Augmenting original_dataset with generated questions 
        augmented_dataset = []

        if args.dataset_type == "spider":
            for orig_datapoint, output in zip(original_dataset, outputs):
                original_question = output["original_question"]
                generated_question = output["generated_question"] # We don't use section title here
                
                input_sequence = orig_datapoint["input_sequence"]
                input_sequence = input_sequence.replace(original_question, generated_question)
                augmented_dataset.append(
                    {
                        "generated_question": generated_question,
                        "original_question": original_question,
                        "db_id": orig_datapoint["db_id"],
                        "input_sequence": input_sequence,
                        "output_sequence": orig_datapoint["output_sequence"],
                        "tc_original": orig_datapoint["tc_original"],
                    }
                )
        elif args.dataset_type == "mschema2qa":
            # mschema2qa
            for orig_datapoint, output in zip(original_dataset, outputs):
                original_question = output["original_question"]
                generated_question = output["generated_question"]
                mr = orig_datapoint["mr"]["thingtalk"][args.dataset_lang]
                augmented_dataset.append(
                    {
                        "generated_question": generated_question,
                        "original_question": original_question,
                        "mr": {"thingtalk": {args.dataset_lang: mr}},
                    }
                )
                

        with open(save_path, "w") as f:
            json.dump(augmented_dataset, f, indent=4, ensure_ascii=False)    

        print(f"Saved dataset augmented with synthetic question to: {save_path}")

        
