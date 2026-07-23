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
    parser.add_argument("--validation-output")
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--validation-offset", type=int, default=0,
                        help="Start index of a question-disjoint model-selection slice.")
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

    holdouts = []
    for name, output_name, size, offset in (
        ("calibration", args.calibration_output, args.calibration_size, args.calibration_offset),
        ("validation", args.validation_output, args.validation_size, args.validation_offset),
    ):
        if output_name is None:
            if size:
                raise ValueError(f"--{name}-output is required with --{name}-size")
            continue
        if size <= 0:
            raise ValueError(f"--{name}-size must be positive with --{name}-output")
        start = offset
        end = start + size
        if start < 0 or end > len(converted):
            raise ValueError(f"{name} slice [{start}:{end}] exceeds {len(converted)} rows")
        holdouts.append((start, end, name, Path(output_name)))

    holdouts.sort()
    for previous, current in zip(holdouts, holdouts[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"{previous[2]} slice [{previous[0]}:{previous[1]}] overlaps "
                f"{current[2]} slice [{current[0]}:{current[1]}]"
            )

    destinations = [("policy", Path(args.output))]
    destinations.extend((name, destination) for _, _, name, destination in holdouts)
    resolved = {}
    for name, destination in destinations:
        key = destination.expanduser().resolve()
        if key in resolved:
            raise ValueError(
                f"{name} output and {resolved[key]} output resolve to the same path: {key}"
            )
        resolved[key] = name

    for start, end, name, destination in holdouts:
        subset = converted.select(range(start, end))
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset.to_parquet(str(destination))
        print(f"wrote {len(subset)} {name} examples to {destination}")

    held_out_indices = {
        index
        for start, end, _, _ in holdouts
        for index in range(start, end)
    }
    policy_indices = [index for index in range(len(converted)) if index not in held_out_indices]
    if not policy_indices:
        raise ValueError("calibration and validation slices cannot consume the full training split")
    policy_data = converted.select(policy_indices)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_data.to_parquet(str(output))
    print(f"wrote {len(policy_data)} policy examples to {output}")


if __name__ == "__main__":
    main()
