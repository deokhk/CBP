# Extract language identity vector
# Either be mean vector or subspace 

import os 
import argparse 
import torch
import logging
import datasets, transformers
import numpy as np 

from accelerate.logging import get_logger
from tqdm import tqdm 
from datasets import load_dataset
from models.mT5 import MT5ForConditionalGeneration
from transformers import MT5Tokenizer, AutoTokenizer, default_data_collator
from run import seed_everything
from torch.utils.data import DataLoader
from accelerate import Accelerator


logger = get_logger(__name__)


def main(args):
    seed_everything(args.seed)

    accelerator = Accelerator()

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    tokenizer = MT5Tokenizer.from_pretrained(args.model_name, is_fast=True)
    logger.info(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name) 
    
    logger.info(f"Loading encoder from model {args.model_name}...")
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    model_encoder = model.get_encoder()

    logger.info("Tokenizing...")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", max_length=args.max_seq_length, truncation=True)

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        encoding_dataset = tokenized_dataset["train"]
        encoding_dataset = encoding_dataset.remove_columns(["sentence"])

        encoding_dataset = encoding_dataset.select(range(min(args.max_samples, len(encoding_dataset))))
        logger.info(f"Tokenized dataset shape: {encoding_dataset.shape}")
    
    # Effective batch size would be args.per_device_eval_batch_size * num_gpus
    dataloader = DataLoader(
        encoding_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False,
        collate_fn= default_data_collator,
        drop_last = False
    )
    
    
    device = accelerator.device
    # initialize model
    
    model_encoder, dataloader = accelerator.prepare(model_encoder, dataloader)
    model_encoder.eval()

    hidden_states_list = []
    model.eval()
    samples_seen = 0
    token_count = 0

    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Encoding..")
    for step, batch in enumerate(progress_bar):
        with torch.no_grad():
            outputs = model_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            hidden_states = outputs["last_hidden_state"]
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            attention_mask = torch.flatten(batch["attention_mask"])
            # Select only the non-padded tokens
            hidden_states = hidden_states[attention_mask == 1]
        hidden_states = accelerator.gather(hidden_states)
        # If we are in a multiprocess environment, the last batch has duplicates 
        if accelerator.num_processes > 1:
            if step == len(dataloader) - 1:
                hidden_states = hidden_states[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += hidden_states.shape[0]
        hidden_states_list.append(hidden_states.detach().to("cpu").numpy())
        token_count += hidden_states.shape[0]
        if token_count > args.max_token_count:
            break
    print("Token count: ", token_count)
    hidden_states = np.concatenate(hidden_states_list, axis=0)
    hidden_states = hidden_states[:args.max_token_count]
    logger.info(f"Hidden states shape: {hidden_states.shape}")

    if accelerator.is_local_main_process:
        # Get the language-centroid vector 
        language_mean = np.mean(hidden_states, axis=0)

        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, f"mean.npy"), language_mean)
        logger.info(f"Saved language mean to {args.output_dir}/hidden_states.npy")

        if args.compute_subspace:

            u, s_mat, v_h = np.linalg.svd(hidden_states)
            # Get the language subspace
            squared_s = np.square(s_mat)
            total_variance = np.sum(squared_s)
            variance_sum = 0
            subspace_dim = 0
            for idx, sq_s in enumerate(squared_s.tolist()):
                variance_sum += sq_s
                # Decide subspace dimension, so that the subspace captures at least 90% of the variance
                if variance_sum / total_variance >= 0.9:
                    subspace_dim = idx
                    break
            language_subspace = v_h[:subspace_dim, :]
            
            np.save(os.path.join(args.output_dir, f"subspace_{subspace_dim}.npy"), language_subspace)
            logger.info(f"Saved language-specific subspace to {args.output_dir}/subspace_{subspace_dim}.npy")
            logger.info(f"Subspace dimension: {subspace_dim}. Done!")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/mt5-large")
    parser.add_argument("--dataset_name", type=str, default="deokhk/en_wiki_sentences_1000000")

    parser.add_argument("--per_device_eval_batch_size", type=int, default=64, help="Encoding Batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--max_token_count", type=int, default=1000000)
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples(sentences) to encode")
    parser.add_argument("--output_dir", type=str, default="language_identity/am_wiki_sentences_100000")

    parser.add_argument("--compute_subspace", action="store_true", help="Compute language-specific subspace in addition to mean vector")
    args = parser.parse_args()
    main(args)