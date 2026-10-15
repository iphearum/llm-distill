"""Discover Khmer words worth adding as single tokens.

Why the original score was wrong
--------------------------------
Hugging Face added tokens are not used only where the segmenter put a word.
They are matched on the raw string, leftmost-longest, BEFORE the BPE model
runs, and they ignore word and cluster boundaries. So:

1. A candidate fires inside other words, including in the middle of a Khmer
   cluster: an added token for a word ending in RO will also match where RO
   carries a following sign such as MUUSIKATOAN, leaving that sign to start
   the next token; a token beginning with KA will match a subscript KA after
   COENG. Those matches give the new embedding an unrelated meaning.
2. Overlapping candidates compete: a longer added word swallows a shorter
   one, so segmenter frequencies double count.

This version keeps segmenter frequencies to build a candidate pool, then
simulates the real matching on the corpus, rejects candidates that often cut
clusters, and re-scores by the occurrences that actually survive.
"""
import argparse
import collections
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import regex
from khmer_segmenter import Tokenizer
from tqdm import tqdm

from common import KHMER_WORD, MARK, splits_cluster

_WORKER_SEGMENTER = None
_MAX_CHARS = 18


def init_worker(max_chars):
    global _WORKER_SEGMENTER, _MAX_CHARS
    _WORKER_SEGMENTER = Tokenizer(seg_type='mor')
    _MAX_CHARS = max_chars


def count_chunk(texts):
    counts = collections.Counter()
    for text in texts:
        # khmer-segmenter returns a space-separated string, not a token list.
        for word in _WORKER_SEGMENTER.tokenize(text).split():
            if (KHMER_WORD.fullmatch(word) and 2 <= len(word) <= _MAX_CHARS
                    and not MARK.match(word[0])):
                counts[word] += 1
    return counts


def chunks(items, size):
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def simulate(words, texts):
    """Leftmost-longest matching, as the HF added-token matcher does.

    Returns per-word (matches, cluster-splitting matches).
    """
    hits = collections.Counter()
    bad = collections.Counter()
    if not words:
        return hits, bad
    pattern = regex.compile('|'.join(regex.escape(w)
                                     for w in sorted(words, key=len, reverse=True)))
    for text in texts:
        for m in pattern.finditer(text):
            w = m.group()
            hits[w] += 1
            if splits_cluster(text, m.start(), m.end()):
                bad[w] += 1
    return hits, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--top-k', type=int, default=256)
    ap.add_argument('--pool-factor', type=int, default=4,
                    help='Candidate pool size = top_k * pool_factor')
    ap.add_argument('--min-frequency', type=int, default=8)
    ap.add_argument('--max-split-rate', type=float, default=0.02,
                    help='Reject a word if more than this share of its matches cut a cluster')
    ap.add_argument('--max-chars', type=int, default=18)
    ap.add_argument('--max-rows', type=int, default=20000)
    ap.add_argument('--refine-rounds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--chunk-size', type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)

    with open('data/train.jsonl', encoding='utf-8') as f:
        texts = [json.loads(line)['text'] for line in islice(f, args.max_rows)]
    n_batches = math.ceil(len(texts) / args.chunk_size)

    counts = collections.Counter()
    if args.workers == 1:
        init_worker(args.max_chars)
        results = map(count_chunk, chunks(texts, args.chunk_size))
        for r in tqdm(results, total=n_batches, desc='Segmenting'):
            counts.update(r)
    else:
        with ProcessPoolExecutor(args.workers, initializer=init_worker,
                                 initargs=(args.max_chars,)) as pool:
            for r in tqdm(pool.map(count_chunk, chunks(texts, args.chunk_size)),
                          total=n_batches, desc='Segmenting'):
                counts.update(r)

    words = [w for w, c in counts.items() if c >= args.min_frequency]
    old_len = {}
    for batch in chunks(words, 2048):
        for w, ids in zip(batch, tok(batch, add_special_tokens=False)['input_ids']):
            if len(ids) > 1:
                old_len[w] = len(ids)
    pool = sorted(old_len, key=lambda w: -(old_len[w] - 1) * counts[w])
    pool = pool[:args.top_k * args.pool_factor]
    print(f'{len(counts)} segmented types, {len(old_len)} multi-token, pool {len(pool)}')

    # Pass 1: judge every pool word under competition from the whole pool.
    hits, bad = simulate(pool, texts)
    rejected = {}
    for w in pool:
        if hits[w] and bad[w] / hits[w] > args.max_split_rate:
            rejected[w] = f'splits clusters in {bad[w]}/{hits[w]} matches'
    ranked = [w for w in pool if w not in rejected]

    # Refinement: re-simulate the chosen set alone, drop words that rarely win
    # their matches once longer words compete, refill from the ranked pool.
    def score(w, h):
        return (old_len[w] - 1) * h[w]

    hits, _ = simulate(ranked, texts)
    ranked.sort(key=lambda w: -score(w, hits))
    selected = ranked[:args.top_k]
    reserve = ranked[args.top_k:]
    for _ in range(args.refine_rounds):
        hits, bad = simulate(selected, texts)
        keep, dropped = [], 0
        for w in selected:
            if hits[w] < args.min_frequency:
                rejected[w] = f'only {hits[w]} winning matches'
                dropped += 1
            elif hits[w] and bad[w] / hits[w] > args.max_split_rate:
                rejected[w] = f'splits clusters in {bad[w]}/{hits[w]} matches (final set)'
                dropped += 1
            else:
                keep.append(w)
        selected = keep + reserve[:dropped]
        reserve = reserve[dropped:]
        if not dropped:
            break

    hits, bad = simulate(selected, texts)
    out = [{'token': w,
            'segmenter_frequency': counts[w],
            'matched_occurrences': hits[w],
            'cluster_split_matches': bad[w],
            'old_token_count': old_len[w],
            'estimated_saved_tokens': score(w, hits)}
           for w in selected]
    out.sort(key=lambda d: -d['estimated_saved_tokens'])

    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/candidates.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    Path('artifacts/rejected.json').write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding='utf-8')
    total = sum(d['estimated_saved_tokens'] for d in out)
    print(f'Selected {len(out)}, rejected {len(rejected)}, '
          f'estimated tokens saved on sample: {total}')
    for d in out[:10]:
        print(d)


if __name__ == '__main__':
    main()