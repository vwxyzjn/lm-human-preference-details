import random
import re

import ftfy
from datasets import load_dataset


# bookcorpus dataset, modified from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/datasets/books.py
def books_generator(mode, seed=0, shuffle=False):
    dataset = load_dataset("bookcorpus", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    while True:
        for _, data in enumerate(dataset):
            text = data["text"]
            yield text


# Cnn_dailymail dataset, modified from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/datasets/cnndm.py
def clean_up_start(text):
    text = re.split(r"\(CNN\) +--", text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1] + text[100:]
    text = re.sub(r"^and \w+\n", "", text)
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    return text.strip()


def cnndm_generator(mode, seed=0, shuffle=False):
    dataset = load_dataset("cnn_dailymail", version="3.0.0", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    for _, data in enumerate(dataset):
        original_text = data["article"]
        text = clean_up_start(original_text)
        text = ftfy.fix_text(text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.split("@highlight")[0].strip()

        yield text


# TL;DR dataset, modified from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/datasets/tldr.py
def tldr_generator(mode, seed=0, shuffle=False):
    dataset = load_dataset("webis/tldr-17", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    for _, data in enumerate(dataset):
        text = data["content"]
        yield text


# TL;DR filtered dataset, modified from
# https://github.com/openai/summarize-from-feedback/tree/700967448d10004279f138666442bf1497d0e705#reddit-tldr-dataset
def tldr_filtered_generator(split, seed=0, shuffle=False):
    assert split in ["test", "train", "valid"]

    data = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")[split]
    if shuffle:
        random.seed(seed)
        dataset.shuffle(seed)

    for item in data:
        yield dict(reference=item["summary"], **{k: v for (k, v) in item.items() if k != "summary"})
        # yield f"SUBREDDIT: r/{item['subreddit']}\n\nTITLE: {item['title']}\n\nPOST: {item['post']}\n\nTL;DR:"


# for testing only
def dummy_generator(mode, seed=0, shuffle=False):
    while True:
        yield "dummy"


DATASET = {
    "books": books_generator,
    "cnndm": cnndm_generator,
    "tldr": tldr_generator,
    "tldr_3_filtered": tldr_filtered_generator,
    "dummy": dummy_generator,
}


from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional, Union

import numpy as np
import torch


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, float):
        return np.array(x)
    raise ValueError(f"Unexpected type {type(x)}")


# from summarize_from_feedback.utils import hyperparams
PADDING_TOKEN = -1


@dataclass
class TaskQueryHParams:
    length: int = None
    dataset: str = None
    format_str: Optional[str] = None  # if underlying dataset yields dicts, can format arbitrarily
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[str] = None  # defaults to repeated spaces
    pad_side: Optional[str] = None


@dataclass
class TaskResponseHParams:
    ref_format_str: Optional[str] = None  # if underlying dataset yields dicts, can format arbitrarily
    length: int = None
    # Truncate response at the first occurrence of this token when sampling.
    truncate_token: Optional[int] = None


@dataclass
class TaskHParams:
    query: TaskQueryHParams = field(default_factory=TaskQueryHParams)
    response: TaskResponseHParams = field(default_factory=TaskResponseHParams)


# Has endoftext potentially, random stuff after
SampledTokens = NewType("SampledTokens", torch.LongTensor)
SampledTokenList = NewType("SampledTokenList", List[int])
# Has only the actual sample + padding tokens
ProcessedTokens = NewType("ProcessedTokens", torch.LongTensor)
ProcessedTokenList = NewType("ProcessedTokenList", List[int])


class ResponseEncoder:
    def __init__(self, H: TaskResponseHParams, encoder, padding_token=PADDING_TOKEN):
        self.H = H
        self.encoder = encoder
        self.padding_token = padding_token

    def process_responses(self, unprocessed_tokens: SampledTokens) -> ProcessedTokens:
        assert unprocessed_tokens.size(-1) == self.H.length
        if self.H.truncate_token is not None:
            assert self.padding_token is not None
            trunc_idxs = first_true_indices(unprocessed_tokens == self.H.truncate_token).unsqueeze(-1)
            new_size = [1] * (len(unprocessed_tokens.size()) - 1) + [self.H.length]
            idxs = torch.arange(self.H.length, device=unprocessed_tokens.device).view(*new_size)
            return torch.masked_fill(unprocessed_tokens, idxs > trunc_idxs, self.padding_token)
        else:
            return unprocessed_tokens

    def encode_response(self, text: str, allow_truncate: bool = False) -> ProcessedTokenList:
        tokens = self.encoder.encode(text)
        if allow_truncate:
            tokens = tokens[: self.H.length - (0 if self.H.truncate_token is None else 1)]
        if self.H.truncate_token is not None:
            tokens = tokens + [self.H.truncate_token]
        if self.padding_token is None:
            assert len(tokens) == self.H.length
            return tokens
        assert len(tokens) <= self.H.length, f"Response too long (limit {self.H.length}): {text}"
        return tokens + [self.padding_token] * (self.H.length - len(tokens))

    def decode_response(self, processed_response_tokens: ProcessedTokenList) -> str:
        tokens = [x for x in processed_response_tokens if x != self.padding_token]
        if self.H.truncate_token is not None:
            if tokens[-1] == self.H.truncate_token:
                tokens = tokens[:-1]
            else:
                assert len(tokens) == self.H.length
        return self.encoder.decode(tokens)

    def decode_responses(self, processed_response_tokens: Union[ProcessedTokens, np.ndarray]):  # -> array of array of ... str:
        def _decode_responses_list(l):
            if isinstance(l[0], (int, np.int64)):
                return self.decode_response(l)
            return [_decode_responses_list(ll) for ll in l]

        return _decode_responses_list(to_numpy(processed_response_tokens))


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
    if hparams.padding is not None:
        return encoder.encode(hparams.padding)
    return encoder.encode(" ") * hparams.length


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    # breakpoint()
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

    return dict(query_token=_ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence))


if __name__ == "__main__":
    gen = tldr_filtered_generator("train")
    for i in range(10):
        d = next(gen)
        from transformers import AutoTokenizer

        encoder = AutoTokenizer.from_pretrained("gpt2")

        q = process_query(
            d,
            encoder=encoder,
            hparams=TaskQueryHParams(
                length=512,
                dataset="tldr_3_filtered",
                format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
                truncate_field="post",
                truncate_text="\n",
                padding=None,
                pad_side="left",
            ),
        )
        print("===start")
        print(d, len(q))
        print("===", encoder.decode(q["tokens"]))
        print("===end")
