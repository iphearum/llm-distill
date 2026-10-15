"""Sample, clean, deduplicate and split the Khmer corpus.

Changes from the original:
- Samples through a streaming shuffle buffer. Taking the first N rows of a
  stream gives whatever order the dataset was uploaded in (often one source
  or one site at a time), which biases both the vocabulary and the eval set.
- Drops rows that are mostly non-Khmer (URLs, English boilerplate, numbers),
  which otherwise inflate character counts and dilute discovery.
- Collapses whitespace before hashing, so trivially different copies dedupe.
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import regex
from datasets import load_dataset

KHMER_CHAR = regex.compile(r'[\p{Script=Khmer}]')
WS = re.compile(r'\s+')

p = argparse.ArgumentParser()
p.add_argument('--sources', nargs='+',
               default=['nphearum/khmer-raw-text-3M', 'nphearum/khmer-raw-text-3M-v2'])
p.add_argument('--max-per-dataset', type=int, default=12000)
p.add_argument('--shuffle-buffer', type=int, default=50000)
p.add_argument('--min-chars', type=int, default=40)
p.add_argument('--min-khmer-ratio', type=float, default=0.6,
               help='Minimum share of non-space characters that are Khmer script')
p.add_argument('--validation-ratio', type=float, default=.05)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()

random.seed(a.seed)
out = Path('data')
out.mkdir(exist_ok=True)
seen, rows = set(), []
for source in a.sources:
    ds = load_dataset(source, split='train', streaming=True)
    ds = ds.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer)
    added = skipped_short = skipped_lang = skipped_dup = 0
    for row in ds:
        t = row.get('text')
        if not isinstance(t, str):
            continue
        t = WS.sub(' ', unicodedata.normalize('NFC', t)).strip()
        if len(t) < a.min_chars:
            skipped_short += 1
            continue
        non_space = len(t) - t.count(' ')
        if len(KHMER_CHAR.findall(t)) / max(non_space, 1) < a.min_khmer_ratio:
            skipped_lang += 1
            continue
        digest = hashlib.sha256(t.encode()).digest()
        if digest in seen:
            skipped_dup += 1
            continue
        seen.add(digest)
        rows.append({'text': t, 'source': source})
        added += 1
        if a.max_per_dataset and added >= a.max_per_dataset:
            break
    print(f'{source}: kept {added}, short {skipped_short}, '
          f'non-Khmer {skipped_lang}, duplicate {skipped_dup}')

random.shuffle(rows)
cut = max(1, round(len(rows) * a.validation_ratio))
for name, subset in [('valid', rows[:cut]), ('train', rows[cut:])]:
    with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
        for obj in subset:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(name, len(subset))