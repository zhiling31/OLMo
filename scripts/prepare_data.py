"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment
import pandas as pd

def extract_triple_quoted(text):
    start_marker = r"```python"
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""  # 未找到起始标记
    start_idx += 9  # 跳过起始三引号
    end_marker = "```"
    end_idx = text.find(end_marker, start_idx)  # 从start_idx开始查找结束标记
    return text[start_idx:end_idx] if end_idx != -1 else ""




log = logging.getLogger(__name__)


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    dataset = ds.load_dataset("bunyaminergen/Stable-Code-Python-SFT", split="train")
    print("dataset",dataset,type(dataset),dataset.column_names)
    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=dataset.column_names,
        num_proc=opts.num_proc,  # type: ignore
    )


    subset = dataset.select(range(10))  # 仅加载前10条数据[1,7](@ref)

    # 打印所有字段的前10行
    for i in range(10):
        print(f"Row {i+1}:")
        print(f"  messages: {subset[i]['messages']}")
        print(f"  input_ids: {subset[i]['input_ids'][:10]}...")  # 仅展示部分 token
        print(f"  label_mask: {subset[i]['label_mask']}")
        print(f"  n_labels: {subset[i]['n_labels']}\n")

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples")
    print(222,dataset)
    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    log.info(f"Total tokens: {total_tokens:,d}")
    log.info(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_ids_file = np.memmap(
        str(output_dir / "input_ids.npy"), dtype=np.uint32, mode="w+", shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        str(output_dir / "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
    )
    offset = 0
    for ex in track(dataset):
        ex_len = len(ex["input_ids"])  # type: ignore
        input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
        label_mask_file[offset : offset + ex_len] = ex["label_mask"]  # type: ignore
        offset += ex_len
    input_ids_file.flush()
    label_mask_file.flush()

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]
    print(1111111111,example)
    example["messages"] = [{'content':example["instruction"], 'role': 'user'}, {'content':extract_triple_quoted(example["output"]), 'role': 'assistant'}]
    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Tulu V2 dataset")
    parser.add_argument("output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default="d:/OLMo/olmo_data/tokenizers/allenai_dolma2.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=4096)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=100257)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=100277)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
