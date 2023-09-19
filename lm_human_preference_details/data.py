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


# for testing only
def dummy_generator(mode, seed=0, shuffle=False):
    while True:
        yield "dummy"


DATASET = {
    "books": books_generator,
    "cnndm": cnndm_generator,
    "tldr": tldr_generator,
    "dummy": dummy_generator,
}
