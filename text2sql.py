import os
import json
import torch
import argparse
import torch.optim as optim
import transformers
import wandb 
import torch.nn as nn
import math

from tqdm.auto import tqdm
from tokenizers import AddedToken
from accelerate import Accelerator
from sql_metadata import Parser
from preprocessing import sql_keywords, ops

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from mT5_grad_adv import MT5ForConditionalGenerationWithLP
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.mschema2qa_metric.evaluator import MSchema2QAEvaluateTool
from utils.load_dataset import Text2SQLDataset, MSchema2QADataset, TAPMSchema2QADataset, TTMSchema2QADataset, Text2SQLWithLPDataset, Mschema2QAWithLPDataset, MultiMschema2QADataset
from utils.load_dataset import Text2SQLMultiPTDataset, Text2SQLDatasetWithMultiPT, Mschema2QAMultiPTDataset, Mschema2QADatasetWithMultiPT, LanguagePredictionDataset
from utils.load_dataset import ReconstructionDataset, Text2SQLWithReconDataset, Mschema2QAWithReconDataset
from utils.load_dataset import Text2SQLWithLpAndReconDataset, Mschema2QAWithLpAndReconDataset 
from utils.load_dataset import Text2SQLDatasetWithTranslated, Mschema2QADatasetWithTranslated
from utils.text2sql_decoding_utils import decode_sqls



def list_of_strings(arg):
    return arg.split(',')


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--effective_batch_size', type = int, default = 8,
                        help = 'input batch size. Should be effective batch size!')
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_accumulation_step" steps.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 50,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--wandb_log', action="store_true", help="Enable for wandb logging")
    parser.add_argument('--model_name_or_path', type = str, default = "t5-3b",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='train, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "data/preprocessed_data/resdsql_train_spider.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database. ')
    parser.add_argument('--preprocessed_dataset_path', type = str)

    # Reconstruction dataset arguments 
    parser.add_argument("--reconstruction", action="store_true", help="Enable for training with reconstruction dataset")
    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.5)

    # Language prediction penalty arugments 
    parser.add_argument("--lp_penalty", action="store_true", help="Enable for training with language prediction penalty")
    parser.add_argument("--lp_penalty_weight", type=float, default=0.33)

    parser.add_argument("--lp_with_reconstruction", action="store_true", help="Enable for training with language prediction penalty and reconstruction dataset")

    # Train with translated dataset + source labeled dataset 
    parser.add_argument("--labeled_with_translated", action="store_true", help="Enable for training with translated dataset + source labeled dataset")
    parser.add_argument("--translated_dataset_path", type=str, default=None, help="Path of translated dataset. Only used for labeled_with_translated option.")

    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--output", type = str, default = "predicted_sql.txt",
                help = "save file of the predicted sqls.")
    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--dataset_type", type=str, choices=["spider", "mschema2qa"], default="spider")
    parser.add_argument("--dataset_lang", type=str, default="en") 

    parser.add_argument("--mschema2qa_translate_train", action="store_true", help="Enable for translated train set of mschema2qa")
    parser.add_argument("--mschema2qa_TAP", action="store_true", help="Enable for translated TAP of mschema2qa")
    parser.add_argument("--multilingual_training", action="store_true", help="Enable for multilingual training. By enable this, we will save model from epoch 0.")    


    parser.add_argument("--multilingual_pt", action="store_true", help="Enable for multilingual pretraining")
    parser.add_argument("--multi_pt_dataset_path_list", type=list_of_strings, help="Path of synthesized dataset for multilingual pretraining")
    parser.add_argument("--cpt_weight", type=float, default=1.0, help="Weight given to loss for multilingual pretraining")
    

    opt = parser.parse_args()

    return opt

def _train(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}",
        )

    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])


    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")

    if opt.dataset_type == "spider":
        train_dataset = Text2SQLDataset(
            dir_ = opt.train_filepath,
            mode = "train"
        )
    elif opt.dataset_type == "mschema2qa":
        if opt.mschema2qa_translate_train:
            train_dataset = TTMSchema2QADataset(
                dir_ = opt.train_filepath,
                mode = "train",
            )
        elif opt.mschema2qa_TAP:
            train_dataset = TAPMSchema2QADataset(
                dir_ = opt.train_filepath,
                mode = "train",
            )
        elif opt.multilingual_training:
            train_dataset = MultiMschema2QADataset(
                dir_ = opt.train_filepath,
                mode = "train",
            )
        else:
            train_dataset = MSchema2QADataset(
                dir_ = opt.train_filepath,
                data_lang = opt.dataset_lang,
                mode = "train",
            )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )


    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    if is_local_main_process:
        print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size

    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0

    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1
                
                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                with text2sql_tokenizer.as_target_tokenizer():
                    tokenized_outputs = text2sql_tokenizer(
                        batch_sqls, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                
                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = tokenized_outputs["attention_mask"]
                
                    model_outputs = model(
                        input_ids = encoder_input_ids,
                        attention_mask = encoder_input_attention_mask,
                        labels = decoder_labels,
                        decoder_attention_mask = decoder_attention_mask,
                        return_dict = True
                    )
                
                loss = model_outputs["loss"]
                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if is_local_main_process:
            if opt.multilingual_training or epoch>=30:
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()


def _train_with_multilingual_pt(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}",
        )


    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])


    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")

    if opt.dataset_type == "spider":
        train_dataset = Text2SQLDataset(
            dir_ = opt.train_filepath,
            mode = "train"
        )
        train_multipt_dataset = Text2SQLMultiPTDataset(
            synthesized_dataset_paths = opt.multi_pt_dataset_path_list
        )
        train_with_multipt_dataset = Text2SQLDatasetWithMultiPT(
            text2sql_dataset = train_dataset,
            multi_pt_dataset = train_multipt_dataset,
        )
    elif opt.dataset_type == "mschema2qa":
        train_dataset = MSchema2QADataset(
            dir_ = opt.train_filepath,
            data_lang = opt.dataset_lang,
            mode = "train",
        )
        train_multipt_dataset = Mschema2QAMultiPTDataset(
            synthesized_dataset_paths = opt.multi_pt_dataset_path_list
        )
        train_with_multipt_dataset = Mschema2QADatasetWithMultiPT(
            mschema2qa_dataset= train_dataset,
            multi_pt_dataset = train_multipt_dataset,
        )

    train_dataloader = DataLoader(
        train_with_multipt_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    if is_local_main_process:
        print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    # TODO: remove this!
    if is_local_main_process:
        print(f"train_dataset length: {len(train_dataset)}")
    
    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size

    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0
    
    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                if opt.dataset_type == "spider":
                    batch_cpt_inputs = [data[4] for data in batch]
                    batch_cpt_outputs = [data[5] for data in batch]
                elif opt.dataset_type == "mschema2qa":
                    batch_cpt_inputs = [data[2] for data in batch]
                    batch_cpt_outputs = [data[3] for data in batch]

                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                tokenized_cpt_inputs = text2sql_tokenizer(
                    batch_cpt_inputs,
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                if "bart" in opt.model_name_or_path:
                    tokenized_outputs = text2sql_tokenizer(
                        batch_sqls, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                    tokenized_cpt_outputs = text2sql_tokenizer(
                        batch_cpt_outputs, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                else:
                    with text2sql_tokenizer.as_target_tokenizer():
                        tokenized_outputs = text2sql_tokenizer(
                            batch_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                        tokenized_cpt_outputs = text2sql_tokenizer(
                            batch_cpt_outputs, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                
                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = decoder_attention_mask.to(device)
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss_text2sql = model_outputs["loss"]

                # cross-lingual pretraining loss 

                cpt_encoder_input_ids = tokenized_cpt_inputs["input_ids"]
                cpt_encoder_input_attention_mask = tokenized_cpt_inputs["attention_mask"]

                cpt_decoder_labels = tokenized_cpt_outputs["input_ids"]
                cpt_decoder_labels[cpt_decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                cpt_decoder_attention_mask = tokenized_cpt_outputs["attention_mask"]

                if torch.cuda.is_available():
                    cpt_encoder_input_ids = cpt_encoder_input_ids.to(device)
                    cpt_encoder_input_attention_mask = cpt_encoder_input_attention_mask.to(device)
                    cpt_decoder_labels = cpt_decoder_labels.to(device)
                    cpt_decoder_attention_mask = cpt_decoder_attention_mask.to(device)

                model_cpt_outputs = model(
                    input_ids = cpt_encoder_input_ids,
                    attention_mask = cpt_encoder_input_attention_mask,
                    labels = cpt_decoder_labels,
                    decoder_attention_mask = cpt_decoder_attention_mask,
                    return_dict = True
                )

                loss_cpt = model_cpt_outputs["loss"]

                loss = loss_text2sql + opt.cpt_weight * loss_cpt

                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if opt.dataset_type == "mschema2qa":
            if is_local_main_process and epoch>=30:
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)
        else:
            if is_local_main_process and epoch>=30:
                # Without pretraining, We only save model from epoch 30 - till it reaches top performance on source language (also save space either)
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()



def _train_with_lp_penalty(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}",
        )


    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])


    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")

    if opt.dataset_type == "spider":
        train_dataset = Text2SQLDataset(
            dir_ = opt.train_filepath,
            mode = "train"
        )
        train_lp_dataset = LanguagePredictionDataset(
            langs= ["en", "zh", "vi"]
        )
        multitask_train_dataset = Text2SQLWithLPDataset(
            text2sql_dataset = train_dataset,
            lp_dataset = train_lp_dataset,
        )
    elif opt.dataset_type == "mschema2qa":
        train_dataset = MSchema2QADataset(
            dir_ = opt.train_filepath,
            data_lang = opt.dataset_lang,
            mode = "train",
        )
        
        train_lp_dataset = LanguagePredictionDataset(
            langs= ["en", "ar", "de", "es", "fa", "fi", "it", "ja", "pl", "tr", "zh"]
        )
        
        multitask_train_dataset = Mschema2QAWithLPDataset(
            mschema2qa_dataset= train_dataset,
            lp_dataset = train_lp_dataset,
        )

    train_dataloader = DataLoader(
        multitask_train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    model_class = MT5ForConditionalGenerationWithLP if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    num_langs=3 if opt.dataset_type == "spider" else 11
    if is_local_main_process:
        print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path, num_langs)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    # TODO: remove this!
    if is_local_main_process:
        print(f"train_dataset length: {len(train_dataset)}")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size

    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model_decoder_start_token_id = model.config.decoder_start_token_id

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0

    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1
                grad_reverse_lambda = 2/(1+math.exp(-1*40*(train_step/num_training_steps)))-1

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                if opt.dataset_type == "spider":
                    batch_lp_inputs = [data[4] for data in batch]
                    batch_lp_labels = [data[5] for data in batch]
                elif opt.dataset_type == "mschema2qa":
                    batch_lp_inputs = [data[2] for data in batch]
                    batch_lp_labels = [data[3] for data in batch]

                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                tokenized_lp_inputs = text2sql_tokenizer(
                    batch_lp_inputs,
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                if "bart" in opt.model_name_or_path:
                        tokenized_outputs = text2sql_tokenizer(
                            batch_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                else:
                    with text2sql_tokenizer.as_target_tokenizer():
                        tokenized_outputs = text2sql_tokenizer(
                            batch_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                
                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = decoder_attention_mask.to(device)
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss_text2sql = model_outputs["loss"]

                # Language prediction loss 
                lp_input_ids = tokenized_lp_inputs["input_ids"]
                lp_attention_mask = tokenized_lp_inputs["attention_mask"]

                if torch.cuda.is_available():
                    lp_input_ids = lp_input_ids.to(device)
                    lp_attention_mask = lp_attention_mask.to(device)
                    lp_labels = torch.tensor(batch_lp_labels).to(device)

                model_lp_outputs = model(
                    input_ids = lp_input_ids,
                    attention_mask = lp_attention_mask,
                    labels=lp_labels,
                    task_id=1,
                    grad_reverse_lambda=grad_reverse_lambda
                )

                loss_lp = model_lp_outputs["loss"]

                loss = loss_text2sql + opt.lp_penalty_weight*loss_lp

                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if opt.dataset_type == "mschema2qa":
            if is_local_main_process and epoch>=30:
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)
        else:
            if is_local_main_process and epoch>=30:
                # Without pretraining, We only save model from epoch 30, since model with earlier epoch produced invalid sql, resulting in massive evaluation time
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()


def _train_with_reconstruction(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}_with_reconstruction",
        )


    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    special_tokens_dict = {'additional_special_tokens': ["<mask>"]}
    text2sql_tokenizer.add_special_tokens(special_tokens_dict)

    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")

    if opt.dataset_type == "spider":
        train_dataset = Text2SQLDataset(
            dir_ = opt.train_filepath,
            mode = "train"
        )
        train_recon_dataset = ReconstructionDataset(
            langs= ["en", "zh", "vi"],
            tokenizer=text2sql_tokenizer,
            max_seq_length=opt.max_seq_length,
            mask_rate=opt.mask_rate
        )
        multitask_train_dataset = Text2SQLWithReconDataset(
            text2sql_dataset = train_dataset,
            reconstruction_dataset = train_recon_dataset,
        )
    elif opt.dataset_type == "mschema2qa":
        train_dataset = MSchema2QADataset(
            dir_ = opt.train_filepath,
            data_lang = opt.dataset_lang,
            mode = "train",
        )
        
        train_recon_dataset = ReconstructionDataset(
            langs= ["en", "ar", "de", "es", "fa", "fi", "it", "ja", "pl", "tr", "zh"],
            tokenizer=text2sql_tokenizer,
            max_seq_length=opt.max_seq_length,
            mask_rate=opt.mask_rate
        )
        
        multitask_train_dataset = Mschema2QAWithReconDataset(
            mschema2qa_dataset= train_dataset,
            reconstruction_dataset = train_recon_dataset,
        )

    train_dataloader = DataLoader(
        multitask_train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    if is_local_main_process:
        print("initializing text2sql model.")

    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    # TODO: remove this!
    if is_local_main_process:
        print(f"train_dataset length: {len(train_dataset)}")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size


    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model_decoder_start_token_id = model.config.decoder_start_token_id

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0

    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                if opt.dataset_type == "spider":
                    batch_recon_inputs = [data[4] for data in batch]
                    batch_recon_labels = [data[5] for data in batch]
                elif opt.dataset_type == "mschema2qa":
                    batch_recon_inputs = [data[2] for data in batch]
                    batch_recon_labels = [data[3] for data in batch]

                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                if "bart" in opt.model_name_or_path:
                        tokenized_outputs = text2sql_tokenizer(
                            batch_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                else:
                    with text2sql_tokenizer.as_target_tokenizer():
                        tokenized_outputs = text2sql_tokenizer(
                            batch_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )

                tokenized_recon_inputs = text2sql_tokenizer(
                    batch_recon_inputs,
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )

                if "bart" in opt.model_name_or_path:
                    tokenized_recon_labels = text2sql_tokenizer(
                        batch_recon_labels, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 512,
                        truncation = True
                    )
                else:
                    with text2sql_tokenizer.as_target_tokenizer():
                        tokenized_recon_labels = text2sql_tokenizer(
                            batch_recon_labels, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 512,
                            truncation = True
                        )

                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = decoder_attention_mask.to(device)
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss_text2sql = model_outputs["loss"]

                # Reconstruction loss 
                recon_input_ids = tokenized_recon_inputs["input_ids"]
                recon_attention_mask = tokenized_recon_inputs["attention_mask"]

                recon_decoder_labels = tokenized_recon_labels["input_ids"]
                recon_decoder_labels[recon_decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                recon_decoder_attention_mask = tokenized_recon_labels["attention_mask"]


                if torch.cuda.is_available():
                    recon_encoder_input_ids = recon_input_ids.to(device)
                    recon_encoder_input_attention_mask = recon_attention_mask.to(device)
                    recon_decoder_labels = recon_decoder_labels.to(device)
                    recon_decoder_attention_mask = recon_decoder_attention_mask.to(device)


                model_recon_outputs = model(
                    input_ids = recon_encoder_input_ids,
                    attention_mask = recon_encoder_input_attention_mask,
                    labels = recon_decoder_labels,
                    decoder_attention_mask = recon_decoder_attention_mask,
                    return_dict = True
                )
                
                loss_recon = model_recon_outputs["loss"]

                loss = loss_text2sql + opt.reconstruction_loss_weight * loss_recon

                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if opt.dataset_type == "mschema2qa":
            if is_local_main_process and epoch>30:
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)
        else:
            if is_local_main_process and epoch>=30:
                # Without pretraining, We only save model from epoch 30, since model with earlier epoch produced invalid sql, resulting in massive evaluation time
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()



def _train_with_lp_and_reconstruction(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}_with_reconstruction",
        )


    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    special_tokens_dict = {'additional_special_tokens': ["<mask>"]}
    text2sql_tokenizer.add_special_tokens(special_tokens_dict)

    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")

    if opt.dataset_type == "spider":
        train_dataset = Text2SQLDataset(
            dir_ = opt.train_filepath,
            mode = "train"
        )
        train_lp_dataset = LanguagePredictionDataset(
            langs= ["en", "zh", "vi"]
        )
        train_recon_dataset = ReconstructionDataset(
            langs= ["en", "zh", "vi"],
            tokenizer=text2sql_tokenizer,
            max_seq_length=opt.max_seq_length,
            mask_rate=opt.mask_rate
        )
        multitask_train_dataset = Text2SQLWithLpAndReconDataset(
            text2sql_dataset = train_dataset,
            reconstruction_dataset = train_recon_dataset,
            lp_dataset = train_lp_dataset
        )
    elif opt.dataset_type == "mschema2qa":
        train_dataset = MSchema2QADataset(
            dir_ = opt.train_filepath,
            data_lang = opt.dataset_lang,
            mode = "train",
        )

        train_lp_dataset = LanguagePredictionDataset(
            langs= ["en", "ar", "de", "es", "fa", "fi", "it", "ja", "pl", "tr", "zh"]
        )

        train_recon_dataset = ReconstructionDataset(
            langs= ["en", "ar", "de", "es", "fa", "fi", "it", "ja", "pl", "tr", "zh"],
            tokenizer=text2sql_tokenizer,
            max_seq_length=opt.max_seq_length,
            mask_rate=opt.mask_rate
        )
        
        multitask_train_dataset = Mschema2QAWithLpAndReconDataset(
            mschema2qa_dataset= train_dataset,
            reconstruction_dataset = train_recon_dataset,
            lp_dataset = train_lp_dataset
        )

    train_dataloader = DataLoader(
        multitask_train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    model_class = MT5ForConditionalGenerationWithLP if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    num_langs=3 if opt.dataset_type == "spider" else 11
    if is_local_main_process:
        print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path, num_langs)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    # TODO: remove this!
    if is_local_main_process:
        print(f"train_dataset length: {len(train_dataset)}")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size

    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model_decoder_start_token_id = model.config.decoder_start_token_id

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0

    assert "bart" not in opt.model_name_or_path, "bart model is not supported for now."

    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1
                grad_reverse_lambda = 2/(1+math.exp(-1*40*(train_step/num_training_steps)))-1

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                if opt.dataset_type == "spider":
                    batch_lp_inputs = [data[4] for data in batch]
                    batch_lp_labels = [data[5] for data in batch]
                    batch_recon_inputs = [data[6] for data in batch]
                    batch_recon_labels = [data[7] for data in batch]
                elif opt.dataset_type == "mschema2qa":
                    batch_lp_inputs = [data[2] for data in batch]
                    batch_lp_labels = [data[3] for data in batch]
                    batch_recon_inputs = [data[4] for data in batch]
                    batch_recon_labels = [data[5] for data in batch]


                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                with text2sql_tokenizer.as_target_tokenizer():
                    tokenized_outputs = text2sql_tokenizer(
                        batch_sqls, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )

                tokenized_lp_inputs = text2sql_tokenizer(
                    batch_lp_inputs,
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )

                tokenized_recon_inputs = text2sql_tokenizer(
                    batch_recon_inputs,
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )

                with text2sql_tokenizer.as_target_tokenizer():
                    tokenized_recon_labels = text2sql_tokenizer(
                        batch_recon_labels, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 512,
                        truncation = True
                    )

                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = decoder_attention_mask.to(device)
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss_text2sql = model_outputs["loss"]


                # Language prediction loss 
                lp_input_ids = tokenized_lp_inputs["input_ids"]
                lp_attention_mask = tokenized_lp_inputs["attention_mask"]

                if torch.cuda.is_available():
                    lp_input_ids = lp_input_ids.to(device)
                    lp_attention_mask = lp_attention_mask.to(device)
                    lp_labels = torch.tensor(batch_lp_labels).to(device)

                model_lp_outputs = model(
                    input_ids = lp_input_ids,
                    attention_mask = lp_attention_mask,
                    labels=lp_labels,
                    task_id=1,
                    grad_reverse_lambda=grad_reverse_lambda
                )

                loss_lp = model_lp_outputs["loss"]

                # Reconstruction loss 
                recon_input_ids = tokenized_recon_inputs["input_ids"]
                recon_attention_mask = tokenized_recon_inputs["attention_mask"]

                recon_decoder_labels = tokenized_recon_labels["input_ids"]
                recon_decoder_labels[recon_decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                recon_decoder_attention_mask = tokenized_recon_labels["attention_mask"]


                if torch.cuda.is_available():
                    recon_encoder_input_ids = recon_input_ids.to(device)
                    recon_encoder_input_attention_mask = recon_attention_mask.to(device)
                    recon_decoder_labels = recon_decoder_labels.to(device)
                    recon_decoder_attention_mask = recon_decoder_attention_mask.to(device)


                model_recon_outputs = model(
                    input_ids = recon_encoder_input_ids,
                    attention_mask = recon_encoder_input_attention_mask,
                    labels = recon_decoder_labels,
                    decoder_attention_mask = recon_decoder_attention_mask,
                    return_dict = True
                )
                
                loss_recon = model_recon_outputs["loss"]

                loss = loss_text2sql + opt.lp_penalty_weight * loss_lp + opt.reconstruction_loss_weight * loss_recon

                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if opt.dataset_type == "mschema2qa":
            if is_local_main_process and epoch>=30:
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)
        else:
            if is_local_main_process and epoch>=30:
                # Without pretraining, We only save model from epoch 30, since model with earlier epoch produced invalid sql, resulting in massive evaluation time
                save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
                unwrapped_model.save_pretrained(save_directory = save_directory)
                print(f"Checkpoint saved at {save_directory}.")
                text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()


def _train_labeled_with_translated(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}",
        )


    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])


    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")

    if opt.dataset_type == "spider":
        train_dataset = Text2SQLDataset(
            dir_ = opt.train_filepath,
            mode = "train"
        )
        train_translated_dataset = Text2SQLDataset(
            dir_ = opt.translated_dataset_path,
            mode = "train"
        )
        labed_with_translated_train_dataset = Text2SQLDatasetWithTranslated(
            text2sql_dataset = train_dataset,
            translated_text2sql_dataset= train_translated_dataset
        )
    elif opt.dataset_type == "mschema2qa":
        train_dataset = MSchema2QADataset(
            dir_ = opt.train_filepath,
            data_lang = opt.dataset_lang,
            mode = "train",
        )
        train_translated_dataset = TAPMSchema2QADataset(
            dir_ = opt.translated_dataset_path,
            mode = "train",
        )
        labed_with_translated_train_dataset = Mschema2QADatasetWithTranslated(
            mschema2qa_dataset= train_dataset,
            tapm_dataset = train_translated_dataset,
        )

    train_dataloader = DataLoader(
        labed_with_translated_train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    if is_local_main_process:
        print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    if is_local_main_process:
        print(f"train_dataset length: {len(train_dataset)}")
    
    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.effective_batch_size) # Changed from opt.batch_size to opt.effective_batch_size

    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0
    
    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                if opt.dataset_type == "spider":
                    batch_translated_inputs = [data[4] for data in batch]
                    batch_translated_sqls = [data[5] for data in batch]
                elif opt.dataset_type == "mschema2qa":
                    batch_translated_inputs = [data[2] for data in batch]
                    batch_translated_sqls = [data[3] for data in batch]

                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                tokenized_translated_inputs = text2sql_tokenizer(
                    batch_translated_inputs,
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                if "bart" in opt.model_name_or_path:
                    tokenized_outputs = text2sql_tokenizer(
                        batch_sqls, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                    tokenized_translated_outputs = text2sql_tokenizer(
                        batch_translated_sqls, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                else:
                    with text2sql_tokenizer.as_target_tokenizer():
                        tokenized_outputs = text2sql_tokenizer(
                            batch_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                        tokenized_translated_outputs = text2sql_tokenizer(
                            batch_translated_sqls, 
                            padding = "max_length", 
                            return_tensors = 'pt',
                            max_length = 256,
                            truncation = True
                        )
                
                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = decoder_attention_mask.to(device)
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss_text2sql = model_outputs["loss"]

                # translated text2sql loss 

                translated_encoder_input_ids = tokenized_translated_inputs["input_ids"]
                translated_encoder_input_attention_mask = tokenized_translated_inputs["attention_mask"]

                translated_decoder_labels = tokenized_translated_outputs["input_ids"]
                translated_decoder_labels[translated_decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                translated_decoder_attention_mask = tokenized_translated_outputs["attention_mask"]

                if torch.cuda.is_available():
                    translated_encoder_input_ids = translated_encoder_input_ids.to(device)
                    translated_encoder_input_attention_mask = translated_encoder_input_attention_mask.to(device)
                    translated_decoder_labels = translated_decoder_labels.to(device)
                    translated_decoder_attention_mask = translated_decoder_attention_mask.to(device)

                model_translated_outputs = model(
                    input_ids = translated_encoder_input_ids,
                    attention_mask = translated_encoder_input_attention_mask,
                    labels = translated_decoder_labels,
                    decoder_attention_mask = translated_decoder_attention_mask,
                    return_dict = True
                )

                loss_translated_text2sql = model_translated_outputs["loss"]

                loss = loss_text2sql + loss_translated_text2sql

                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if is_local_main_process and epoch>=30:
            save_directory = os.path.join(opt.save_path, "checkpoint-{}".format(train_step)) 
            unwrapped_model.save_pretrained(save_directory = save_directory)
            print(f"Checkpoint saved at {save_directory}.")
            text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()


def _predict_spider(opt):

    import time
    start_time = time.time()
        
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, AutoTokenizer):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )


    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = model_class.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.to(device)
            encoder_input_attention_mask = encoder_input_attention_mask.to(device)

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            predict_sqls += decode_sqls(
                opt.db_path, 
                model_outputs, 
                batch_db_ids, 
                batch_inputs, 
                tokenizer, 
                batch_tc_original
            )
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    return predict_sqls


def switch_sql_to_english(sql, db_schema):

    table_map = dict()
    column_map = dict()

    for table_info in db_schema:
        table_name_original_english = table_info["table_name_original"] # table name in english
        table_name_target = table_info["table_name"] # table name in target language


        table_map[table_name_target] = table_name_original_english


        column_names_original_english = table_info["column_names_original"] # column name in english
        column_names_target = table_info["column_names"] # column name in target language

        for column_name_original, column_name_target in zip(column_names_original_english, column_names_target):
            column_name_original_with_table_name_original = table_name_original_english + "." + column_name_original
            column_name_with_table_name_target = table_name_target + "." + column_name_target
            
            column_map[column_name_target] =  column_name_original
            column_map[column_name_with_table_name_target] = column_name_original_with_table_name_original


    # sort by length of key 
    sorted_column_map = list(sorted(column_map.items(), key = lambda x: len(x[0]), reverse = True))
    sorted_table_map = list(sorted(table_map.items(), key = lambda x: len(x[0]), reverse = True))

    for col_target, col_original in sorted_column_map:
        sql = sql.replace(col_target, col_original)
    
    for table_target, table_original in sorted_table_map:
        sql = sql.replace(table_target, table_original)

    return sql


def _test_spider(opt):
    # Note : for test, we didn't apply acclerators due to complexity of inference
    set_seed(opt.seed)
    print(opt)

    predict_sqls = []
    predict_sqls = _predict_spider(opt)
    
    if opt.save_predictions:
        with open(opt.dev_filepath, "r") as f:
            dev_data = json.load(f)
        
        save_results = []
        for predict_sql, dev_datapoint in zip(predict_sqls, dev_data):
            gt_sql = dev_datapoint["output_sequence"].split("<sql>")[-1].strip()
            input_sequence = dev_datapoint["input_sequence"]
            save_results.append(
                {
                    "utterance": input_sequence,
                    "gt_mr": gt_sql,
                    "pred_mr": predict_sql
                }
            )

        os.makedirs(os.path.dirname(opt.save_predictions_path), exist_ok = True)
        with open(opt.save_predictions_path, 'w') as f:
            json.dump(save_results, f, ensure_ascii = False, indent=4)
            print("Predictions saved at {}".format(opt.save_predictions_path))
    
    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls)
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
    
        return spider_metric_result["exact_match"], spider_metric_result["exec"]



def _test_mschema2qa(opt):
    # Note : for test, we didn't apply acclerators due to complexity of inference
    set_seed(opt.seed)
    print(opt)

    predict_sqls = []

    import time
    start_time = time.time()
        
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, AutoTokenizer):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    dev_dataset = MSchema2QADataset(
        dir_ = opt.dev_filepath,
        data_lang = opt.dataset_lang,
        mode = opt.mode,
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = model_class.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    predict_sqls = []
    gold_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]

        if len(batch[0]) > 1:
            batch_outputs = [data[1] for data in batch]
            gold_sqls += batch_outputs
        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.to(device)
            encoder_input_attention_mask = encoder_input_attention_mask.to(device)

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 512,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            batch_size = model_outputs.shape[0]

            for batch_id in range(batch_size):
                pred_sequence = tokenizer.decode(model_outputs[batch_id, 0, :], skip_special_tokens = True)
                predict_sqls.append(pred_sequence)

    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    
    if opt.save_predictions:
        with open(opt.dev_filepath, "r") as f:
            dev_data = json.load(f)
        
        
        save_results = []
        for predict_mr, dev_datapoint in zip(predict_sqls, dev_data):
            gt_mr = dev_datapoint["mr"]["thingtalk"][opt.dataset_lang]
            question = dev_datapoint["question"][opt.dataset_lang]
            save_results.append(
                {
                    "utterance": question,
                    "gt_mr": gt_mr,
                    "pred_mr": predict_mr
                }
            )

        os.makedirs(os.path.dirname(opt.save_predictions_path), exist_ok = True)
        with open(opt.save_predictions_path, 'w') as f:
            json.dump(save_results, f, ensure_ascii = False, indent=4)
            print("Predictions saved at {}".format(opt.save_predictions_path))


    if opt.mode == "eval":
        # initialize evaluator
        # measure exact match 
        gold_sqls = [gold_sql.strip() for gold_sql in gold_sqls]
        predict_sqls = [pred_sql.strip() for pred_sql in predict_sqls]

        evaluator = MSchema2QAEvaluateTool(opt)
        metric_result = evaluator.evaluate(predict_sqls, gold_sqls)
        print('exact_match score: {}'.format(metric_result["exact_match"]))

        return metric_result["exact_match"]



if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        if opt.multilingual_pt:
            _train_with_multilingual_pt(opt)
        elif opt.lp_penalty:
            _train_with_lp_penalty(opt)
        elif opt.reconstruction:
            _train_with_reconstruction(opt)
        elif opt.lp_with_reconstruction:
            _train_with_lp_and_reconstruction(opt)
        elif opt.labeled_with_translated:
            _train_labeled_with_translated(opt)
        else:
            _train(opt)
    elif opt.mode in ["eval", "test"]:
        if opt.dataset_type == "spider":
            _test_spider(opt)
        elif opt.dataset_type == "mschema2qa":
            _test_mschema2qa(opt)
