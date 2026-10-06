import json
import random
import gzip

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path,
        seq_length,
        shuffle,
    ):
        self.seq_length = seq_length

        # 1. 读取数据
        documents = []

        if dataset_path.endswith(".gz"):
            f = gzip.open(dataset_path, "rt", encoding="utf-8")
        else:
            f = open(dataset_path, "r", encoding="utf-8")

        with f:
            for line in f:
                documents.append(json.loads(line))

        # 2. 是否打乱 document 顺序
        if shuffle:
            random.shuffle(documents)

        # 3. 把所有 document 转成 token
        all_tokens = []

        for item in documents:
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{item['prompt']}\n\n"
                "### Response:\n"
                f"{item['response']}"
            )

            tokens = tokenizer(
                text,
                add_special_tokens=True,
            )["input_ids"]

            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)

        # 4. 切成 seq_length + 1 的连续 chunks
        self.sequences = []

        chunk_length = seq_length + 1

        for i in range(
            0,
            len(all_tokens) - chunk_length + 1,
            seq_length,
        ):
            self.sequences.append(
                all_tokens[i : i + chunk_length]
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        tokens = torch.tensor(
            self.sequences[i],
            dtype=torch.long,
        )

        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }

def run_iterate_batches(
    dataset,
    batch_size,
    shuffle,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader