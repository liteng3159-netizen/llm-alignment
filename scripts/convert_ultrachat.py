import json
import gzip
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# 路径
# ============================================================

INPUT_DIR = Path(
    "/mnt/data/lt/modelscope-cache/"
    "datasets/swift--ultrachat_200k/snapshots/master/data"
)

OUTPUT_FILE = Path(
    "/mnt/data/lt/homework5/data/ultrachat_test.jsonl.gz"
)


# ============================================================
# 找到所有 train_sft parquet
# ============================================================

parquet_files = sorted(INPUT_DIR.glob("test_sft-*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"没有找到 train_sft parquet 文件：{INPUT_DIR}"
    )

print("找到以下文件：")
for path in parquet_files:
    print(f"  {path}")

print(f"\n共 {len(parquet_files)} 个 parquet 文件")


# ============================================================
# 创建输出目录
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 转换
# ============================================================

total = 0

with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as fout:

    for parquet_file in parquet_files:

        print(f"\n正在处理：{parquet_file.name}")

        df = pd.read_parquet(parquet_file)

        print(f"  rows = {len(df)}")
        print(f"  columns = {list(df.columns)}")

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=parquet_file.name,
        ):
            prompt = row["prompt"]

            messages = row["messages"]

            # 找到第一个 assistant 回复
            response = None

            for message in messages:
                if message["role"] == "assistant":
                    response = message["content"]
                    break

            if response is None:
                print("WARNING: 没有找到 assistant message，跳过")
                continue

            example = {
                "prompt": prompt,
                "response": response,
            }

            fout.write(json.dumps(
                example,
                ensure_ascii=False,
            ) + "\n")

            total += 1


# ============================================================
# 完成
# ============================================================

print("\n========================================")
print("转换完成！")
print(f"总样本数: {total}")
print(f"输出文件: {OUTPUT_FILE}")
print(f"文件大小: {OUTPUT_FILE.stat().st_size / 1024**2:.2f} MB")
print("========================================")