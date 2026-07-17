#!/usr/bin/env python3
"""Convert FlashRAG QA splits to the Search-R1/Search-G1 parquet schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets


DATASET_ALIASES = {
    "nq": "nq",
    "natural_questions": "nq",
    "hotpot": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
}

PROMPT = """Answer the given question.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, directly provide the answer inside <answer> and </answer>, without detailed illustrations.
For example, <answer> Beijing </answer>.
Question: {question}
"""


def choose_split(dataset: datasets.DatasetDict, requested: str) -> str:
    if requested == "auto":
        for candidate in ("test", "dev", "validation", "train"):
            if candidate in dataset:
                return candidate
    if requested in dataset:
        return requested
    fallbacks = {
        "test": ("dev", "validation"),
        "dev": ("validation", "test"),
        "validation": ("dev", "test"),
    }
    for candidate in fallbacks.get(requested, ()):
        if candidate in dataset:
            return candidate
    raise ValueError(f"split {requested!r} is unavailable; found {list(dataset.keys())}")


def gold_answers(example: dict) -> list[str]:
    value = example.get("golden_answers", example.get("answers", example.get("answer", [])))
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict):
        value = value.get("text", [])
    return [str(answer) for answer in value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_ALIASES))
    parser.add_argument("--split", default="auto",
                        choices=("auto", "train", "test", "dev", "validation"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="RUC-NLPIR/FlashRAG_datasets")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--calibration-output")
    parser.add_argument("--calibration-size", type=int, default=0)
    parser.add_argument("--calibration-offset", type=int, default=0,
                        help="Start index of a question-disjoint calibration slice.")
    args = parser.parse_args()

    config = DATASET_ALIASES[args.dataset]
    dataset_dict = datasets.load_dataset(args.source, config)
    split = choose_split(dataset_dict, args.split)
    source = dataset_dict[split]

    def convert(example: dict, index: int) -> dict:
        question = str(example["question"]).strip()
        if question and not question.endswith("?"):
            question += "?"
        return {
            "data_source": config,
            "prompt": [{"role": "user", "content": PROMPT.format(question=question)}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": gold_answers(example)},
            },
            "extra_info": {"split": split, "index": index},
        }

    converted = source.map(
        convert,
        with_indices=True,
        remove_columns=source.column_names,
        num_proc=args.num_proc,
        desc=f"Preparing {config}/{split}",
    )

    policy_data = converted
    if args.calibration_output:
        if args.calibration_size <= 0:
            raise ValueError("--calibration-size must be positive with --calibration-output")
        start = args.calibration_offset
        end = start + args.calibration_size
        if start < 0 or end > len(converted):
            raise ValueError(f"calibration slice [{start}:{end}] exceeds {len(converted)} rows")
        calibration = converted.select(range(start, end))
        remaining = []
        if start:
            remaining.append(converted.select(range(0, start)))
        if end < len(converted):
            remaining.append(converted.select(range(end, len(converted))))
        if not remaining:
            raise ValueError("calibration split cannot consume the full policy-training split")
        policy_data = remaining[0] if len(remaining) == 1 else datasets.concatenate_datasets(remaining)
        calibration_output = Path(args.calibration_output)
        calibration_output.parent.mkdir(parents=True, exist_ok=True)
        calibration.to_parquet(str(calibration_output))
        print(f"wrote {len(calibration)} calibration examples to {calibration_output}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_data.to_parquet(str(output))
    print(f"wrote {len(policy_data)} policy examples to {output}")


if __name__ == "__main__":
    main()
