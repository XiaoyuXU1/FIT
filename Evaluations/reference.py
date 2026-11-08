"""Compute common endpoint metrics once, outside each method's online budget."""
import argparse
import gc

import torch

from utils import load_model, load_tokenizer, manifest, pch_records, write_json
from Evaluations.evaluator import GENERATION_BATCH_SIZE, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--retain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = pch_records(args.seed)
    result = {"protocol": "pch-v2", "metric_generation_batch_size":GENERATION_BATCH_SIZE, **manifest(records)}
    for label, name in (("initial", args.initial), ("retain", args.retain)):
        model = load_model(name, args.device).eval()
        tokenizer = load_tokenizer(name)
        result[label] = {"checkpoint": name,
                         "forget": evaluate(model, tokenizer, records[:300]),
                         "retain": evaluate(model, tokenizer, records[300:])}
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output, result)


if __name__ == "__main__":
    main()
