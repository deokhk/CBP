import torch
import os
import argparse 
from safetensors import safe_open
from safetensors.torch import save_file, load_model
from models.mT5 import MT5ForConditionalGeneration
from transformers import MT5Tokenizer


def main(args):
    print("Using checkpoints at ", args.checkpoint_dir)
    adapter_checkpoints = {}
    files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith("safetensors")]
    for file in files:
        with safe_open(f"{args.checkpoint_dir}/{file}", framework="pt", device="cpu") as f:
            for key in f.keys():
                if "adapter" in key.lower() and f"lang.{args.lang}" in key.lower():
                    adapter_checkpoints[key] = f.get_tensor(key)

    checkpoint_parent_dir = "/".join(args.checkpoint_dir.split("/")[:-1])
    save_fname = args.output_adapter_file_name + ".safetensors"
    save_path = os.path.join(checkpoint_parent_dir, save_fname)

    save_file(adapter_checkpoints, save_path)
    print(f"Saved adapter checkpoints to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True, help="Adapter language. en, zh")
    parser.add_argument("--output_adapter_file_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
