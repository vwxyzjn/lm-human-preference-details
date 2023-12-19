import multiprocessing
import os
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from datasets import load_dataset
from huggingface_hub import HfApi
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()


"""
poetry run python lm_human_preference_details/tldr_dataset.py
poetry run python lm_human_preference_details/tldr_dataset.py \
    --base-model=EleutherAI/pythia-160m \
    --max-sft-response-length=53 \
    --max-rm-response-length=169
"""


@dataclass
class Args:
    base_model: str = "gpt2"  # EleutherAI/pythia-160m
    max_sft_response_length: int = 48  # 53
    max_rm_response_length: int = 153  # 169
    hf_entity: str = None


@dataclass
class TaskQueryHParams:
    length: int = 512
    format_str: Optional[
        str
    ] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"  # if underlying dataset yields dicts, can format arbitrarily
    truncate_field: Optional[str] = "post"
    truncate_text: Optional[str] = "\n"
    padding: Optional[str] = " "  # empty spaces
    pad_side: Optional[str] = "left"


def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]


def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    return hparams.padding * hparams.length


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    query_tokens = encoder.encode(format_str.format(**query_info))
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encoder.encode(format_str.format(**query_info))

    query_token = _ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    oai_h = TaskQueryHParams()
    if isinstance(oai_h.padding, str):
        oai_h.padding = tokenizer.encode(oai_h.padding)
    else:
        oai_h.padding = tokenizer.pad_token_id
    pprint(oai_h)
    sft_ds = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")

    def process_query_data(x):
        # the `x['summary']` in `vwxyzjn/summarize_from_feedback_tldr_3_filtered`
        # DOES NOT HAVE a leading space so we are adding the leading space and
        # `<|endoftext|>` token
        reference_response = f" {x['summary']}<|endoftext|>"
        return {
            **process_query(x, encoder=tokenizer, hparams=oai_h),
            "reference_response": reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=args.max_sft_response_length,
                truncation=True,
            ),
            "reference_response_token_len": len(tokenizer.encode(reference_response)),
        }

    sft_ds = sft_ds.map(process_query_data, load_from_cache_file=False, num_proc=multiprocessing.cpu_count())
    sft_ds.push_to_hub(
        f"{args.hf_entity}/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_{args.base_model.split('/')[-1]}_{args.max_sft_response_length}"
    )

    label_ds = load_dataset("openai/summarize_from_feedback", "comparisons")

    def process_response_data(x):
        # the `x['summaries'][0]['text']` in `openai/summarize_from_feedback` `comaprisons`
        # DOES HAVE a leading space so we are just adding the `<|endoftext|>` token
        response0 = f"{x['summaries'][0]['text']}<|endoftext|>"
        response1 = f"{x['summaries'][1]['text']}<|endoftext|>"
        response0_policy = x["summaries"][0]["policy"]
        response1_policy = x["summaries"][1]["policy"]
        policies = "--".join(sorted([response0_policy, response1_policy]))
        return {
            **process_query(x["info"], encoder=tokenizer, hparams=oai_h),
            "response0": response0,
            "response0_token": tokenizer.encode(
                response0, padding="max_length", max_length=args.max_rm_response_length, truncation=True
            ),
            "response0_token_len": len(tokenizer.encode(response0)),
            "response1": response1,
            "response1_token": tokenizer.encode(
                response1, padding="max_length", max_length=args.max_rm_response_length, truncation=True
            ),
            "response1_token_len": len(tokenizer.encode(response1)),
            "response0_policy": response0_policy,
            "response1_policy": response1_policy,
            "policies": policies,
        }

    label_ds = label_ds.map(process_response_data, load_from_cache_file=False, num_proc=multiprocessing.cpu_count())
    label_ds.push_to_hub(
        f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{args.base_model.split('/')[-1]}_{args.max_rm_response_length}"
    )

    os.makedirs("dataset_visuals", exist_ok=True)
    # visualize token length distribution
    num_subplots = len(sft_ds) + len(label_ds) * 2
    print(f"{num_subplots=}")
    fig, axs = plt.subplots(3, 3, figsize=(16, 16))
    axs = axs.flatten()
    for i, key in enumerate(sft_ds.keys()):
        df = sft_ds[key].to_pandas()
        axs[i].hist(df["reference_response_token_len"], bins=100)
        axs[i].set_title(f"{key} split: reference response token length\nmax_length={max(df['reference_response_token_len'])}")
    offset = len(sft_ds)
    for i, key in enumerate(label_ds.keys()):
        df = label_ds[key].to_pandas()
        axs[2 * i + offset].hist(df["response0_token_len"], bins=100)
        axs[2 * i + offset].set_title(f"{key} split: response0 token length\nmax_length={max(df['response0_token_len'])}")
        axs[2 * i + offset + 1].hist(df["response1_token_len"], bins=100)
        axs[2 * i + offset + 1].set_title(f"{key} split: response1 token length\nmax_length={max(df['response1_token_len'])}")
    fig.suptitle(f"{args.base_model} Tokenizer: Token length distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/token_len.png")

    # visualize confidence distribution
    fig, axs = plt.subplots(len(label_ds), 1, figsize=(8, 8))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, key in enumerate(label_ds.keys()):
        df = label_ds[key].to_pandas()
        axs[i].hist(df["extra.confidence"])
        axs[i].set_title(f"{key} split: confidence distribution")
    fig.suptitle("Confidence distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/confidence.png")

    # visualize policies used
    fig, axs = plt.subplots(1, len(label_ds), figsize=(8, 12))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, key in enumerate(label_ds.keys()):
        df = label_ds[key].to_pandas()
        cat = pd.concat([df["response0_policy"], df["response1_policy"]], axis=0)
        cat.hist(ax=axs[i], xrot=90, orientation="horizontal")
        axs[i].set_title(f"{key} split: policy distribution")
    fig.suptitle("Policy distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/policies.png")

    # visualize compairson distribution
    fig, axs = plt.subplots(1, len(label_ds), figsize=(24, 30))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, key in enumerate(label_ds.keys()):
        df = label_ds[key].to_pandas()
        df["policies"].hist(ax=axs[i], xrot=90, orientation="horizontal")
        axs[i].set_title(f"{key} split: policy comparison distribution")
    fig.suptitle("Policy comparison distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/policy_comparisons.png")

    # upload the `dataset_visuals`

    api.upload_folder(
        folder_path="dataset_visuals",
        path_in_repo="dataset_visuals",
        repo_id=f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{args.base_model.split('/')[-1]}_{args.max_rm_response_length}",
        repo_type="dataset",
    )
    # upload current file
    print(f"{__file__=}")
    api.upload_file(
        path_or_fileobj=__file__,
        path_in_repo="create_dataset.py",
        repo_id=f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_{args.base_model.split('/')[-1]}_{args.max_rm_response_length}",
        repo_type="dataset",
    )
