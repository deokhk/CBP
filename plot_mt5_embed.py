#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import MT5Tokenizer, MT5EncoderModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

LANGS = ["en", "zh", "ko", "ar"]

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_text_column(example):
    """Automatically detect a column that contains sentence text."""
    for k in ["text", "sentence", "content"]:
        if k in example:
            return k
    # Fallback: pick the first string-type column
    for k, v in example.items():
        if isinstance(v, str):
            return k
    raise ValueError("No text column found. Please specify with --text_column.")

def load_sentences(lang: str, n: int, text_column_override: str = None, seed: int = 42) -> List[str]:
    """Load n sentences from HuggingFace dataset for a given language."""
    ds_name = f"deokhk/{lang}_wiki_sentences_1000000"
    ds = load_dataset(ds_name, split="train")

    # Detect text column
    if text_column_override is None:
        text_col = pick_text_column(ds[0])
    else:
        text_col = text_column_override
        if text_col not in ds.column_names:
            raise ValueError(f"Column '{text_col}' not found in {ds_name}. Available: {ds.column_names}")

    # Sample using shuffle+select (avoids numpy int64 indexing issues)
    k = min(n, len(ds))
    ds_sample = ds.shuffle(seed=seed).select(range(k))
    return list(ds_sample[text_col])

@torch.no_grad()
def encode_sentence_representations(
    sentences: List[str],
    tokenizer: MT5Tokenizer,
    model: MT5EncoderModel,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "mean",   # "mean" or "last"
) -> np.ndarray:
    """
    Encode sentences using the mT5 encoder.
    pooling:
      - "mean": mean pooling over non-padding tokens using attention_mask
      - "last": hidden state of the last non-padding token
    """
    reps = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"]                 # (B, T)
        bsz = h.size(0)

        if pooling == "mean":
            # Mean over valid (non-pad) tokens
            mask_f = mask.unsqueeze(-1).to(h.dtype)  # (B, T, 1)
            summed = (h * mask_f).sum(dim=1)         # (B, H)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            vecs = summed / lengths.to(h.dtype)      # (B, H)
        else:
            # "last" = last non-padding token
            last_idx = mask.sum(dim=1) - 1
            vecs = h[torch.arange(bsz, device=device), last_idx]

        reps.append(vecs.cpu().numpy())

    return np.concatenate(reps, axis=0)

# def plot_2d(
#     X_2d: np.ndarray,
#     labels: List[str],
#     out_png: str,
#     title: str = "mT5 encoder (last-token) 2D",
#     x_label: str = "Dim 1",
#     y_label: str = "Dim 2",
# ):
#     """Visualize 2D embeddings and save as PNG."""
#     plt.figure(figsize=(8, 6), dpi=150)

#     # Fixed color map for languages
#     color_map = {
#         "en": "C0",  # blue
#         "zh": "C1",  # orange
#         "ko": "C2",  # green
#         "ar": "C3",  # red
#         "fi": "C4",  # purple
#     }

#     for lang in sorted(set(labels), key=lambda x: LANGS.index(x) if x in LANGS else x):
#         idx = [i for i, l in enumerate(labels) if l == lang]
#         plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=16, alpha=0.75, label=lang, c=color_map.get(lang, None))

#     plt.legend(title="Language", markerscale=1.2)
#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.tight_layout()
#     plt.savefig(out_png)
#     plt.close()


# def plot_2d(
#     X_2d: np.ndarray,
#     labels: List[str],
#     out_png: str,
#     title: str = "mT5 encoder (last-token) 2D",
#     x_label: str = "Dim 1",
#     y_label: str = "Dim 2",
# ):
#     """Visualize 2D embeddings and save as PNG. Also plot per-language mean as a star."""
#     plt.figure(figsize=(8, 6), dpi=150)

#     # Fixed color map for languages
#     color_map = {
#         "en": "C0",  # blue
#         "zh": "C1",  # orange
#         "ko": "C2",  # green
#         "ar": "C3",  # red
#         "fi": "C4",  # purple (unused here but kept for consistency)
#     }

#     # Scatter points per language
#     langs_sorted = sorted(set(labels), key=lambda x: LANGS.index(x) if x in LANGS else x)
#     labels_arr = np.array(labels)

#     for lang in langs_sorted:
#         idx = np.where(labels_arr == lang)[0]
#         plt.scatter(
#             X_2d[idx, 0], X_2d[idx, 1],
#             s=16, alpha=0.75, label=lang,
#             c=color_map.get(lang, None)
#         )

#     # Overlay language means (stars)
#     for lang in langs_sorted:
#         idx = np.where(labels_arr == lang)[0]
#         if len(idx) == 0:
#             continue
#         mean_xy = X_2d[idx].mean(axis=0)
#         plt.scatter(
#             mean_xy[0], mean_xy[1],
#             marker="*", s=220,
#             c=color_map.get(lang, None),
#             edgecolors="k", linewidths=0.9,
#             zorder=5
#         )

#     # Build two legends: clusters and means
#     from matplotlib.lines import Line2D
#     cluster_handles = [Line2D([0], [0], marker='o', linestyle='',
#                               color=color_map.get(l, 'k'), label=l, markersize=6, alpha=0.9)
#                        for l in langs_sorted]
#     mean_handles = [Line2D([0], [0], marker='*', linestyle='',
#                            markerfacecolor=color_map.get(l, 'k'), markeredgecolor='k',
#                            label=f"{l} mean", markersize=12)
#                     for l in langs_sorted]

#     # First legend: clusters
#     leg1 = plt.legend(handles=cluster_handles, title="Language (points)", loc="best")
#     plt.gca().add_artist(leg1)
#     # Second legend: means
#     plt.legend(handles=mean_handles, title="Per-language mean (★)", loc="upper right")

#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.tight_layout()
#     plt.savefig(out_png)
#     plt.close()


def plot_2d(
    X_2d: np.ndarray,
    labels: List[str],
    out_png: str,
    title: str = "mT5 encoder (last-token) 2D",
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    show_means: bool = False,
):
    """Visualize 2D embeddings and save as PNG.
    If show_means=True, plot per-language mean stars and show corresponding legend.
    Otherwise, show language cluster legend only.
    """
    plt.figure(figsize=(8, 6), dpi=150)

    # Fixed color map for languages
    color_map = {
        "en": "C0",
        "zh": "C1",
        "ko": "C2",
        "ar": "C3",
        "fi": "C4",
    }

    langs_sorted = sorted(set(labels), key=lambda x: LANGS.index(x) if x in LANGS else x)
    labels_arr = np.array(labels)

    # Scatter points
    for lang in langs_sorted:
        idx = np.where(labels_arr == lang)[0]
        plt.scatter(
            X_2d[idx, 0], X_2d[idx, 1],
            s=16, alpha=0.75,
            label=lang if not show_means else None,
            c=color_map.get(lang, None)
        )

    if show_means:
        # Overlay means and build legend
        from matplotlib.lines import Line2D
        mean_handles = []
        for lang in langs_sorted:
            idx = np.where(labels_arr == lang)[0]
            if len(idx) == 0:
                continue
            mean_xy = X_2d[idx].mean(axis=0)
            plt.scatter(
                mean_xy[0], mean_xy[1],
                marker="*", s=220,
                c=color_map.get(lang, None),
                edgecolors="k", linewidths=0.9,
                zorder=5
            )
            mean_handles.append(
                Line2D([0], [0], marker="*", linestyle="",
                       markerfacecolor=color_map.get(lang, "k"),
                       markeredgecolor="k", label=f"{lang} mean", markersize=12)
            )
        plt.legend(handles=mean_handles, title="Per-language mean (★)", loc="best")
    else:
        plt.legend(title="Language (points)", markerscale=1.2, loc="best")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="mT5 encoder representation 2D viz by language (PCA or t-SNE)")
    parser.add_argument("--model_name", type=str, default="google/mt5-large")
    parser.add_argument("--langs", nargs="+", default=LANGS, help="List of language codes")
    parser.add_argument("--samples_per_lang", type=int, default=100)
    parser.add_argument("--text_column", type=str, default=None, help="Column name for sentence text (optional)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./out_embed")

    # Method switch
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="pca",
                        help="Dimensionality reduction method: 'pca' or 'tsne'.")
    parser.add_argument(
        "--pooling", type=str, choices=["mean", "last"], default="mean",
        help="Sentence pooling strategy: 'mean' over non-pad tokens or 'last' token.")   
    parser.add_argument("--show_means", action="store_true",
                        help="If set, plot per-language mean stars (★) instead of default language legends.")


    # Preprocessing
    parser.add_argument("--scale", action="store_true", help="Standardize features before reduction (recommended).")

    # PCA options
    parser.add_argument("--whiten", action="store_true", help="Use PCA whitening (PCA only).")

    # t-SNE options
    parser.add_argument("--perplexity", type=float, default=40.0, help="t-SNE perplexity.")
    parser.add_argument("--learning_rate", type=float, default=200.0, help="t-SNE learning rate.")
    parser.add_argument("--n_iter", type=int, default=2000, help="t-SNE iterations.")
    parser.add_argument("--metric", type=str, default="euclidean", help="t-SNE distance metric.")
    parser.add_argument("--tsne_init", type=str, choices=["random", "pca"], default="random",
                        help="t-SNE init strategy.")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    print("[Info] Loading tokenizer/model...")
    tokenizer = MT5Tokenizer.from_pretrained(args.model_name)
    model = MT5EncoderModel.from_pretrained(args.model_name).to(device)
    model.eval()

    all_reps = []
    all_labels = []

    # Loop over languages
    for lang in args.langs:
        print(f"[Info] Loading sentences for {lang}...")
        sents = load_sentences(lang, args.samples_per_lang, args.text_column, args.seed)

        print(f"[Info] Encoding {lang} ({len(sents)} sentences)...")
        reps = encode_sentence_representations(
            sents, tokenizer, model, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        )
        all_reps.append(reps)
        all_labels.extend([lang] * reps.shape[0])

    X = np.vstack(all_reps)
    labels = all_labels

    # Optional standardization (often helps both PCA and t-SNE)
    if args.scale:
        print("[Info] Standardizing features (mean=0, var=1)...")
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X)
    else:
        X_proc = X

    # Dimensionality reduction
    if args.method == "pca":
        print("[Info] Running PCA to 2D...")
        pca = PCA(n_components=2, whiten=args.whiten, random_state=args.seed)
        X_2d = pca.fit_transform(X_proc)

        # Save raw embeddings to CSV
        csv_path = os.path.join(args.out_dir, "mt5_lasttoken_representations.csv")
        print(f"[Info] Saving representations to {csv_path}")
        header = ",".join(["language"] + [f"h{i}" for i in range(X.shape[1])])
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row, lang in zip(X, labels):
                f.write(lang + "," + ",".join(map(str, row.tolist())) + "\n")

        # Save 2D coordinates
        csv2d_path = os.path.join(args.out_dir, "mt5_lasttoken_pca2d.csv")
        print(f"[Info] Saving 2D PCA coordinates to {csv2d_path}")
        with open(csv2d_path, "w", encoding="utf-8") as f:
            f.write(f"# explained_variance_ratio: {pca.explained_variance_ratio_.tolist()}\n")
            f.write("language,x,y\n")
            for (x, y), lang in zip(X_2d, labels):
                f.write(f"{lang},{x},{y}\n")

        # Plot
        suffix = "means" if args.show_means else "points"
        png_path = os.path.join(args.out_dir, f"mt5_{args.pooling}_pca_{suffix}.png")
        title = f"mT5 encoder rep ({args.pooling}) PCA | langs={','.join(args.langs)}"
        plot_2d(X_2d, labels, png_path, title=title,
                x_label="PC 1", y_label="PC 2", show_means=args.show_means)

    else:  # tsne
        print("[Info] Running t-SNE to 2D...")
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
            metric=args.metric,
            init=args.tsne_init,
            random_state=args.seed,
            verbose=1,
        )
        X_2d = tsne.fit_transform(X_proc)

        # Save raw embeddings to CSV
        csv_path = os.path.join(args.out_dir, "mt5_lasttoken_representations.csv")
        print(f"[Info] Saving representations to {csv_path}")
        header = ",".join(["language"] + [f"h{i}" for i in range(X.shape[1])])
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row, lang in zip(X, labels):
                f.write(lang + "," + ",".join(map(str, row.tolist())) + "\n")

        # Save 2D coordinates
        csv2d_path = os.path.join(args.out_dir, "mt5_lasttoken_tsne2d.csv")
        print(f"[Info] Saving 2D t-SNE coordinates to {csv2d_path}")
        with open(csv2d_path, "w", encoding="utf-8") as f:
            f.write("language,x,y\n")
            for (x, y), lang in zip(X_2d, labels):
                f.write(f"{lang},{x},{y}\n")

        # Plot
        suffix = "means" if args.show_means else "points"
        png_path = os.path.join(args.out_dir, f"mt5_{args.pooling}_tsne_{suffix}.png")
        title = f"mT5 encoder rep ({args.pooling}) t-SNE | langs={','.join(args.langs)}"
        plot_2d(X_2d, labels, png_path, title=title,
                x_label="t-SNE dim 1", y_label="t-SNE dim 2", show_means=args.show_means)

    print("[Done]")

if __name__ == "__main__":
    main()
