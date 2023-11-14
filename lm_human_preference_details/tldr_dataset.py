from dataclasses import dataclass
from typing import Dict, Optional, Union

from datasets import load_dataset
from rich.pretty import pprint
from transformers import AutoTokenizer


@dataclass
class TaskQueryHParams:
    length: int = 512
    dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered"
    format_str: Optional[
        str
    ] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"  # if underlying dataset yields dicts, can format arbitrarily
    truncate_field: Optional[str] = "post"
    truncate_text: Optional[str] = "\n"
    padding: Optional[Union[str, int]] = 50257
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    max_response_length = 48
    oai_h = TaskQueryHParams()
    if isinstance(oai_h.padding, str):
        oai_h.padding = tokenizer.encode(oai_h.padding)
    else:
        oai_h.padding = [oai_h.padding]
    pprint(oai_h)
    dataset = load_dataset(oai_h.dataset)

    def process_query_data(x):
        # with an extra leading space to account for the space between the query and response
        reference_response = f" {x['summary']}<|endoftext|>"
        return {
            **process_query(x, encoder=tokenizer, hparams=oai_h),
            "reference_response": reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=max_response_length,
                truncation=True,
            ),
        }

    dataset = dataset.map(process_query_data, load_from_cache_file=False)
    push_result = dataset.push_to_hub("vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing")
    print(push_result)

    label = load_dataset("openai/summarize_from_feedback", "comparisons")

    def process_response_data(x):
        # with an extra leading space to account for the space between the query and response
        response0 = f" {x['summaries'][0]['text']}<|endoftext|>"
        response1 = f" {x['summaries'][1]['text']}<|endoftext|>"
        return {
            **process_query(x["info"], encoder=tokenizer, hparams=oai_h),
            "response0": response0,
            "response0_token": tokenizer.encode(
                response0, padding="max_length", max_length=max_response_length, truncation=True
            ),
            "response1": response1,
            "response1_token": tokenizer.encode(
                response1, padding="max_length", max_length=max_response_length, truncation=True
            ),
        }

    label = label.map(process_response_data, load_from_cache_file=False)
    push_result = label.push_to_hub("vwxyzjn/summarize_from_feedback_oai_preprocessing")
