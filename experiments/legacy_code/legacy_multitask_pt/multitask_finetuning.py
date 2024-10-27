import os
import json
import torch
import argparse
import torch.optim as optim
import transformers
import wandb 
import torch.nn as nn
import datasets 
import evaluate 
import logging 
import random 
import numpy as np 

from tqdm.auto import tqdm
from tokenizers import AddedToken
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM,
    default_data_collator, DataCollatorForSeq2Seq, SchedulerType,
    get_scheduler
)
from transformers.trainer_utils import set_seed
from utils.load_dataset import load_multitask_dataset, load_multitask_dataset_legacy
from utils.multitask_eval_utils import extract_schema_prediction_labels_batch, extract_value_prediction_labels_batch, batch_compute_f1




logger = get_logger(__name__)

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    

    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,
                        help = 'perform gradient descent per "gradient_accumulation_step" steps.')

    parser.add_argument('--learning_rate',type = float, default = 5e-5,
                        help = 'learning rate.')
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )


    parser.add_argument('--num_training_steps', type = int, default = 100000,
                        help = 'training steps.')
    parser.add_argument('--num_eval_steps', type=int, default=5000,
                        help='evaluate every num_eval_steps.')
    parser.add_argument('--logging_steps', type=int, default=20,
                        help='log every logging_steps.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    
    parser.add_argument('--save_path', type = str, default = "models/multitask_finetuned",
                        help = 'save path of the multitask-fine tuned model.')
    parser.add_argument('--wandb_log', action="store_true", help="Enable for wandb logging")
    parser.add_argument('--exp_name', default=None, type=str, help='Name of the experiment')
    parser.add_argument('--model_name_or_path', type = str, default = "t5-3b",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )


    parser.add_argument('--mt_train_filepath', type = str, default = "data/multitask_ft/translation_train.json",
                        help = 'file path of machine translation training set.')
    parser.add_argument('--sp_train_filepath', type = str, default = "data/multitask_ft/schema_prediction_train.json",
                        help = 'file path of schema prediction training set.')
    parser.add_argument('--vp_train_filepath', type = str, default = "data/multitask_ft/value_prediction_train.json",
                        help = 'file path of value prediction training set.')
    parser.add_argument('--mt_val_filepath', type = str, default = "data/multitask_ft/translation_val.json",
                    help = 'file path of machine translation training set.')
    parser.add_argument('--sp_val_filepath', type = str, default = "data/multitask_ft/schema_prediction_val.json",
                        help = 'file path of schema prediction training set.')
    parser.add_argument('--vp_val_filepath', type = str, default = "data/multitask_ft/value_prediction_val.json",
                        help = 'file path of value prediction training set.')
    parser.add_argument('--k', type = int, default = pow(2,16),
                        help = 'K used for examples-propotional sampling.')

    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--only_save_best_model", action="store_true",
                        help="Only save the best model instead of saving every checkpoint.")
    opt = parser.parse_args()

    return opt

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def train(opt):

    set_seed(opt.seed)
    accelerator = Accelerator()

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    is_local_main_process = accelerator.is_local_main_process

    logger.info(opt)


    if opt.wandb_log and is_local_main_process:
        wandb_exp_name = opt.exp_name if opt.exp_name else f"{opt.model_name_or_path}_pretrain_{opt.k}"
        wandb.init(
            project="ZX_seq2seq",
            name=wandb_exp_name,
            config=opt
        )


    with accelerator.main_process_first():
        train_dataset = load_multitask_dataset(opt.mt_train_filepath, opt.sp_train_filepath, opt.vp_train_filepath, opt.k)
        dev_mt_dataset = datasets.Dataset.from_json(opt.mt_val_filepath)
        dev_sp_dataset = datasets.Dataset.from_json(opt.sp_val_filepath)
        dev_vp_dataset = datasets.Dataset.from_json(opt.vp_val_filepath)


    tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(tokenizer, AutoTokenizer):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])


    # Temporarily set max_target_length for training.
    max_target_length = opt.max_target_length
    padding = "max_length" if opt.pad_to_max_length else False

    def preprocess_function(examples):
        # Here, we already append prompt prefix to the inputs, no need to set it in the tokenizer
        inputs = [ex for ex in examples["input_sequence"]]
        targets = [ex for ex in examples["output_sequence"]]
        model_inputs = tokenizer(inputs, max_length=opt.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and opt.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=opt.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not opt.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        tokenized_dev_mt = dev_mt_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=opt.preprocessing_num_workers,
            remove_columns=dev_mt_dataset.column_names,
            load_from_cache_file=not opt.overwrite_cache,
            desc="Running tokenizer on dev mt dataset",
        )
        tokenized_dev_sp = dev_sp_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=opt.preprocessing_num_workers,
            remove_columns=dev_sp_dataset.column_names,
            load_from_cache_file=not opt.overwrite_cache,
            desc="Running tokenizer on dev sp dataset",
        )
        tokenized_dev_vp = dev_vp_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=opt.preprocessing_num_workers,
            remove_columns=dev_vp_dataset.column_names,
            load_from_cache_file=not opt.overwrite_cache,
            desc="Running tokenizer on dev vp dataset",
        )

    accelerator.wait_for_everyone()

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    logger.info("initializing model..")
    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))


    # DataLoaders creation:
    label_pad_token_id = -100 if opt.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if opt.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if Accelerator.mixed_precision == 'fp16' else None,
        )

    num_workers = 4

    train_dataloader = DataLoader(
        tokenized_train, 
        batch_size = opt.per_device_train_batch_size, 
        shuffle = True,
        collate_fn = data_collator,
        num_workers = num_workers
    )

    dev_mt_dataloader = DataLoader(
        tokenized_dev_mt, 
        batch_size = opt.per_device_eval_batch_size, 
        shuffle = False,
        collate_fn = data_collator,
    )
    dev_sp_dataloader = DataLoader(
        tokenized_dev_sp, 
        batch_size = opt.per_device_eval_batch_size, 
        shuffle = False,
        collate_fn = data_collator,
    )
    dev_vp_dataloader = DataLoader(
        tokenized_dev_vp, 
        batch_size = opt.per_device_eval_batch_size, 
        shuffle = False,
        collate_fn = data_collator,
    )
    
    # Load metrics for evaluation
    sacrebleu_metric = evaluate.load("sacrebleu")


    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": opt.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=opt.learning_rate)

    lr_scheduler = get_scheduler(
        name=opt.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=opt.num_warmup_steps,
        num_training_steps=opt.num_training_steps*4, 
    )

    # For now, the library has an issue so we don't prepare lr_scheduler with accelerator.prepare
    # Check: https://github.com/huggingface/diffusers/issues/3954


    # Train!
    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * opt.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total optimization steps = {opt.num_training_steps}")
    logger.info(f"  Instantaneous batch size per device = {opt.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")

    model, optimizer, train_dataloader, dev_mt_dataloader, dev_sp_dataloader, dev_vp_dataloader, lr_scheduler  = accelerator.prepare(
        model, optimizer, train_dataloader, dev_mt_dataloader, dev_sp_dataloader, dev_vp_dataloader, lr_scheduler
    )

    best_bleu_normalized = 0
    best_table_f1 = 0
    best_column_f1 = 0
    best_value_f1 = 0
    best_combined_score = 0

    avg_loss = 0
    completed_steps = 0
    
    model.train()
    train_pbar = tqdm(total=opt.num_training_steps, disable=not is_local_main_process, desc="Training..")

    
    while completed_steps < opt.num_training_steps:
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            avg_loss += loss.item()

            loss = loss / opt.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % opt.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_pbar.update(1)
                completed_steps += 1

                if completed_steps % opt.logging_steps == 0 and is_local_main_process and accelerator.sync_gradients:
                    if opt.wandb_log:
                        wandb.log({"train loss": avg_loss / opt.gradient_accumulation_steps, "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=completed_steps)
                    else:
                        logger.info(f"At {completed_steps} training step, loss = {avg_loss / opt.gradient_accumulation_steps}.")
                    avg_loss = 0

            if completed_steps >= opt.num_training_steps:
                break
                    
            # Evaluation 
            if completed_steps % opt.num_eval_steps == 0:
                model.eval()

                if opt.val_max_target_length is None:
                    opt.val_max_target_length = opt.max_target_length

                gen_kwargs = {
                    "max_length": opt.val_max_target_length,
                    "num_beams": opt.num_beams
                }

                samples_seen = 0
                # Firstly, evaluate translation task performance 
                dev_mt_pbar = tqdm(total=len(dev_mt_dataloader), disable=not is_local_main_process, desc="Evaluating MT performance..")
                
                for step, batch in enumerate(dev_mt_dataloader):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not opt.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                        labels = accelerator.gather(labels).cpu().numpy()

                        if opt.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        # If we are in a multiprocess environment, the last batch has duplicates
                        if accelerator.num_processes > 1:
                            if step == len(dev_mt_dataloader) - 1:
                                decoded_preds = decoded_preds[: len(dev_mt_dataloader.dataset) - samples_seen]
                                decoded_labels = decoded_labels[: len(dev_mt_dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += len(decoded_labels)

                        sacrebleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)


                    if is_local_main_process:
                        dev_mt_pbar.update(1)

                
                if is_local_main_process:
                    # Compute metrics
                    metrics = sacrebleu_metric.compute()
                    if opt.wandb_log:
                        wandb.log({"eval_bleu": metrics["score"]}, step=completed_steps)
                    else:
                        logger.info(f"At {completed_steps} training step, eval_bleu = {metrics['score']}.")

                accelerator.wait_for_everyone()        

                samples_seen = 0
                # Secondly, evaluate schema prediction task performance
                dev_sp_pbar = tqdm(total=len(dev_sp_dataloader), disable=not is_local_main_process, desc="Evaluating schema prediction performance..")
                sp_predictions = []
                sp_references = []

                for step, batch in enumerate(dev_sp_dataloader):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not opt.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                        labels = accelerator.gather(labels).cpu().numpy()

                        if opt.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        # If we are in a multiprocess environment, the last batch has duplicates
                        if accelerator.num_processes > 1:
                            if step == len(dev_mt_dataloader) - 1:
                                decoded_preds = decoded_preds[: len(dev_mt_dataloader.dataset) - samples_seen]
                                decoded_labels = decoded_labels[: len(dev_mt_dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += len(decoded_labels)

                        sp_predictions += decoded_preds
                        sp_references += decoded_labels

                    if is_local_main_process:
                        dev_sp_pbar.update(1)

                
                if is_local_main_process:
                    # Compute metrics
                    batch_pred_table_labels, batch_pred_column_labels = extract_schema_prediction_labels_batch(sp_predictions)
                    batch_ref_table_labels, batch_ref_column_labels = extract_schema_prediction_labels_batch(sp_references)
                    
                    assert len(batch_pred_table_labels) == len(batch_ref_table_labels)
                    assert len(batch_pred_column_labels) == len(batch_ref_column_labels)

                    avg_tb_f1 = batch_compute_f1(batch_pred_table_labels, batch_ref_table_labels)
                    avg_col_f1 = batch_compute_f1(batch_pred_column_labels, batch_ref_column_labels)
                    
                    if opt.wandb_log:
                        wandb.log({"avg_tb_f1": avg_tb_f1, "avg_col_f1":avg_col_f1}, step=completed_steps)
                    else:
                        logger.info(f"At {completed_steps} training step, avg_tb_f1 = {avg_tb_f1}, avg_col_f1 = {avg_col_f1}.")

                accelerator.wait_for_everyone()                        


                samples_seen = 0
                # Finally, evaluate value prediction task performance
                dev_vp_pbar = tqdm(total=len(dev_vp_dataloader), disable=not is_local_main_process, desc="Evaluating value prediction performance..")
                vp_predictions = []
                vp_references = []

                for step, batch in enumerate(dev_vp_dataloader):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not opt.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                        labels = accelerator.gather(labels).cpu().numpy()

                        if opt.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        # If we are in a multiprocess environment, the last batch has duplicates
                        if accelerator.num_processes > 1:
                            if step == len(dev_mt_dataloader) - 1:
                                decoded_preds = decoded_preds[: len(dev_mt_dataloader.dataset) - samples_seen]
                                decoded_labels = decoded_labels[: len(dev_mt_dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += len(decoded_labels)

                        vp_predictions += decoded_preds
                        vp_references += decoded_labels

                    if is_local_main_process:
                        dev_vp_pbar.update(1)

                
                if is_local_main_process:
                    # Compute metrics
                    batch_pred_value_labels = extract_value_prediction_labels_batch(vp_predictions)
                    batch_ref_value_labels = extract_value_prediction_labels_batch(vp_references)
                    assert len(batch_pred_value_labels) == len(batch_ref_value_labels)

                    avg_value_f1 = batch_compute_f1(batch_pred_value_labels, batch_ref_value_labels)

                    if opt.wandb_log:
                        wandb.log({"avg_value_f1": avg_value_f1}, step=completed_steps)
                    else:
                        logger.info(f"At {completed_steps} training step, avg_value_f1 = {avg_value_f1}")

                accelerator.wait_for_everyone()                        


                if not opt.only_save_best_model:
                    logger.info(f"At {completed_steps} training step, save a checkpoint.")
                    os.makedirs(opt.save_path, exist_ok = True)

                    accelerator.wait_for_everyone()
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    if is_local_main_process:
                        save_directory = opt.save_path + "/checkpoint-{}".format(completed_steps)
                        unwrapped_model.save_pretrained(save_directory = save_directory)
                        logger.info(f"Checkpoint saved at {save_directory}.")
                        tokenizer.save_pretrained(save_directory = save_directory)

                # Save the evaluation result at current steps
                if is_local_main_process:
                    os.makedirs(opt.save_path, exist_ok = True)
                    # Open text file in write mode 
                    with open(opt.save_path + "/eval_result.txt", "a") as file1:
                        # Append evaluation result to the file
                        file1.write(f"Step: {completed_steps}, eval_bleu = {metrics['score']}, avg_tb_f1 = {avg_tb_f1}, avg_col_f1 = {avg_col_f1}, avg_value_f1 = {avg_value_f1}.\n")
                    
                    normalized_bleu = metrics["score"] / 100
                    if normalized_bleu > best_bleu_normalized:
                        best_bleu_normalized = normalized_bleu
                    if avg_tb_f1 > best_table_f1:
                        best_table_f1 = avg_tb_f1
                    if avg_col_f1 > best_column_f1:
                        best_column_f1 = avg_col_f1
                    if avg_value_f1 > best_value_f1:
                        best_value_f1 = avg_value_f1

                    cur_combined_score = normalized_bleu*3 + avg_tb_f1 + avg_col_f1 + avg_value_f1 # Equal weighting b/w translation task and schema/value prediction task

                    if cur_combined_score > best_combined_score:
                        best_combined_score = cur_combined_score
                        # Save best model checkpoint
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_directory = opt.save_path + "/best_model"
                        unwrapped_model.save_pretrained(save_directory = save_directory)
                        logger.info(f"Best model saved at {save_directory}.")
                        tokenizer.save_pretrained(save_directory = save_directory)

                    if opt.wandb_log:
                        wandb.log({"best_bleu": best_bleu_normalized*100, "best_table_f1":best_table_f1,
                                   "best_column_f1": best_column_f1, "best_value_f1": best_value_f1}, 
                                   step=completed_steps)

                model.train()



    wandb.finish()


if __name__ == "__main__":
    opt = parse_option()
    train(opt)
