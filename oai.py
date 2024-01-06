# you can download the CSV from https://wandb.ai/costa-huang/tldr_summarize/runs/gb2dian5

import asyncio
import random
from dataclasses import dataclass

import pandas as pd
import tyro
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

limiter = AsyncLimiter(1000, 60)


@dataclass
class Args:
    csv_path: str = "trained_response.csv"
    max_samples: int = 64


# client = OpenAI()
async_client = AsyncOpenAI()


template = r"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.
### Post:
{{post}}
### Summary A:
{{summarya}}
### Summary B:
{{summaryb}}
### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""


async def process_text(post, summary_a, summary_b, i):
    text = template.replace("{{post}}", post)
    text = text.replace("{{summarya}}", summary_a)
    text = text.replace("{{summaryb}}", summary_b)  # Ensure this split logic is correct for your data

    async with limiter:
        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
            ],
        )
        r = response.choices[0].message.content
        try:
            comparison = r.split("Comparison:")[1].split("Preferred:")[0].strip()
            preferred = r.split("Preferred:")[1].strip()
            return comparison, preferred, i
        except:
            print(f"error in {i}")
            return "", random.choice(["A", "B"]), i


async def main(args: Args):
    num_trails = 2
    for j in range(num_trails):
        print(j)
        tasks = []
        df = pd.read_csv(args.csv_path)
        df["explanation"] = [None for _ in range(len(df))]
        df["preferred"] = [None for _ in range(len(df))]
        df["shuffled_index"] = [None for _ in range(len(df))]
        r = range(min(args.max_samples, len(df)))
        if args.max_samples == -1:
            r = range(len(df))
        for i in r:
            post = df["query"].iloc[i].strip()
            # shuffled the index to avoid GPT4's preference bias in the content's order
            shuffled_index = random.randint(0, 1)
            df.at[i, "shuffled_index"] = shuffled_index
            summaries = [
                df["postprocessed_response"].iloc[i].strip(),
                df["reference_responses"].iloc[i].split("<|endoftext|>")[0].strip(),
            ]
            summary_a = summaries[shuffled_index]
            summary_b = summaries[1 - shuffled_index]
            task = asyncio.create_task(process_text(post, summary_a, summary_b, i))
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks)

        for _, (comparison, preferred, i) in enumerate(results):
            df.at[i, "explanation"] = comparison
            preferred_label = (
                "ours"
                if (df.at[i, "shuffled_index"] == 0 and preferred == "A")
                or (df.at[i, "shuffled_index"] == 1 and preferred == "B")
                else "reference"
            )
            df.at[i, "preferred"] = preferred_label

        print(df["preferred"].value_counts())
        df.to_csv(f"{args.csv_path}_judged.csv")
        # return df


if __name__ == "__main__":
    args = tyro.cli(Args)
    asyncio.run(main(args))
